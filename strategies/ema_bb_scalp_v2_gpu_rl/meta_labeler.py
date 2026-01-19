"""
Layer 2: Meta-Labeler Training

This module provides training for the Meta-Labeler, which predicts
P(success) for each candidate entry signal.

Supports two model types:
1. XGBoost GPU: Fast, interpretable, tree-based
2. Transformer: Sequence-aware, captures temporal patterns

The meta-labeler is trained on Triple Barrier labels and used
to guide the RL agent's entry decisions.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict
from tqdm import tqdm

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not installed. XGBoost meta-labeler unavailable.")

from .config import MetaLabelerConfig
from .model import TransformerMetaLabeler


def prepare_meta_labeler_data(
    df: pd.DataFrame,
    config: MetaLabelerConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for meta-labeler training.

    Only uses bars where a signal was generated (base_signal != 0).

    Returns:
        X_train, y_train, X_val, y_val
    """
    # Filter to signal bars only
    signal_mask = df['base_signal'] != 0
    signal_df = df[signal_mask].copy()

    print(f"Total signals for meta-labeler training: {len(signal_df)}")

    # Extract features
    feature_cols = config.feature_columns
    X = signal_df[feature_cols].values.astype(np.float32)

    # Handle any remaining NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Labels
    y = signal_df['meta_label'].values.astype(np.float32)

    # Chronological split (not random - prevents look-ahead bias)
    split_idx = int(len(X) * (1 - config.train_val_split))

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    print(f"Train win rate: {y_train.mean()*100:.1f}%")
    print(f"Val win rate: {y_val.mean()*100:.1f}%")

    return X_train, y_train, X_val, y_val


def train_xgboost_meta_labeler(
    df: pd.DataFrame,
    config: MetaLabelerConfig,
    save_path: Optional[str] = None
) -> xgb.Booster:
    """
    Train XGBoost meta-labeler on GPU.

    Args:
        df: DataFrame with features and meta_label column
        config: MetaLabelerConfig
        save_path: Optional path to save model

    Returns:
        Trained XGBoost Booster
    """
    if not HAS_XGB:
        raise RuntimeError("XGBoost not installed")

    # Prepare data
    X_train, y_train, X_val, y_val = prepare_meta_labeler_data(df, config)

    # Create DMatrix objects
    dtrain = xgb.DMatrix(
        X_train, label=y_train,
        feature_names=config.feature_columns
    )
    dval = xgb.DMatrix(
        X_val, label=y_val,
        feature_names=config.feature_columns
    )

    # XGBoost parameters (GPU-accelerated)
    params = {
        'objective': 'binary:logistic',
        'tree_method': config.xgb_tree_method,
        'device': config.xgb_device,
        'max_depth': config.xgb_max_depth,
        'learning_rate': config.xgb_learning_rate,
        'eval_metric': config.xgb_eval_metric,
        'seed': config.random_seed,
        # Additional regularization
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    }

    print("\nTraining XGBoost meta-labeler...")
    print(f"Parameters: {params}")

    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=config.xgb_n_estimators,
        evals=evals,
        early_stopping_rounds=config.xgb_early_stopping_rounds,
        verbose_eval=50
    )

    # Evaluate final performance
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)

    train_acc = ((train_pred > 0.5) == y_train).mean() * 100
    val_acc = ((val_pred > 0.5) == y_val).mean() * 100

    print(f"\nFinal Performance:")
    print(f"  Train Accuracy: {train_acc:.1f}%")
    print(f"  Val Accuracy: {val_acc:.1f}%")

    # Feature importance
    importance = model.get_score(importance_type='gain')
    print("\nFeature Importance (gain):")
    for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {score:.2f}")

    # Save model
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(save_path))
        print(f"\nModel saved to: {save_path}")

    return model


def load_xgboost_meta_labeler(model_path: str) -> Optional[xgb.Booster]:
    """Load trained XGBoost meta-labeler."""
    if not HAS_XGB:
        return None

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None

    model = xgb.Booster()
    model.load_model(str(model_path))
    return model


def predict_xgboost(
    model: xgb.Booster,
    features: np.ndarray,
    feature_names: list
) -> np.ndarray:
    """Get predictions from XGBoost model."""
    dmatrix = xgb.DMatrix(features, feature_names=feature_names)
    return model.predict(dmatrix)


# =============================================================================
# Transformer Meta-Labeler Training
# =============================================================================

def prepare_sequence_data(
    df: pd.DataFrame,
    config: MetaLabelerConfig,
    seq_len: int = 20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare sequence data for Transformer meta-labeler.

    Creates sliding windows of features for sequence classification.
    """
    feature_cols = config.feature_columns
    features = df[feature_cols].values.astype(np.float32)
    labels = df['meta_label'].values.astype(np.float32)
    signals = df['base_signal'].values

    # Normalize features
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(features.shape[1]):
        col = features[:, i]
        mean, std = col.mean(), col.std() + 1e-8
        features[:, i] = (col - mean) / std

    # Create sequences only for signal bars
    X_list = []
    y_list = []

    for i in range(seq_len, len(df)):
        if signals[i] != 0:  # Only signal bars
            X_list.append(features[i-seq_len:i])
            y_list.append(labels[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Split
    split_idx = int(len(X) * (1 - config.train_val_split))

    X_train = torch.tensor(X[:split_idx])
    y_train = torch.tensor(y[:split_idx])
    X_val = torch.tensor(X[split_idx:])
    y_val = torch.tensor(y[split_idx:])

    print(f"Sequence data: Train={len(X_train)}, Val={len(X_val)}")
    return X_train, y_train, X_val, y_val


def train_transformer_meta_labeler(
    df: pd.DataFrame,
    config: MetaLabelerConfig,
    device: str = "cuda",
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    save_path: Optional[str] = None
) -> TransformerMetaLabeler:
    """
    Train Transformer meta-labeler.
    """
    # Prepare sequence data
    X_train, y_train, X_val, y_val = prepare_sequence_data(
        df, config, config.transformer_seq_len
    )

    # Move to device
    device = torch.device(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # Create model
    model = TransformerMetaLabeler(
        input_dim=len(config.feature_columns),
        d_model=config.transformer_d_model,
        nhead=config.transformer_nhead,
        num_layers=config.transformer_num_layers,
        seq_len=config.transformer_seq_len,
        dropout=config.transformer_dropout
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.BCELoss()

    best_val_acc = 0
    best_model_state = None

    print("\nTraining Transformer meta-labeler...")

    for epoch in range(epochs):
        model.train()

        # Shuffle training data
        perm = torch.randperm(len(X_train), device=device)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        total_loss = 0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_x = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            val_acc = ((val_pred > 0.5) == y_val).float().mean().item() * 100

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={total_loss/n_batches:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val Acc={val_acc:.1f}%")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"\nBest Val Accuracy: {best_val_acc:.1f}%")

    # Save model
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    return model


def train_meta_labeler(
    df: pd.DataFrame,
    config: MetaLabelerConfig,
    save_dir: str = "models"
) -> Dict:
    """
    Main function to train meta-labeler.

    Automatically selects XGBoost or Transformer based on config.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if config.model_type == "xgboost":
        save_path = save_dir / "meta_labeler.json"
        model = train_xgboost_meta_labeler(
            df, config, save_path=str(save_path)
        )
        results["xgboost_model"] = model
        results["xgboost_path"] = str(save_path)

    elif config.model_type == "transformer":
        save_path = save_dir / "meta_labeler_transformer.pt"
        model = train_transformer_meta_labeler(
            df, config, save_path=str(save_path)
        )
        results["transformer_model"] = model
        results["transformer_path"] = str(save_path)

    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    return results
