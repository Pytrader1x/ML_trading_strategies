"""
Model wrapper with activation hooks for visualization.

Supports loading models from different strategies.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path
import sys


def get_device() -> str:
    """Determine the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class ActorCriticWithActivations(nn.Module):
    """
    Wrapper for ActorCritic models with activation hooks.

    Loads the base model from the strategy directory and adds
    forward hooks to capture intermediate activations for visualization.
    """

    def __init__(self, base_model: nn.Module):
        """
        Initialize with a base model.

        Args:
            base_model: The underlying ActorCritic model
        """
        super().__init__()
        self.model = base_model
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        # Hook into encoder layers
        if hasattr(self.model, 'encoder'):
            for i, layer in enumerate(self.model.encoder):
                if isinstance(layer, nn.Linear):
                    layer.register_forward_hook(get_activation(f'encoder_{i}'))

        # Hook into actor/critic heads
        if hasattr(self.model, 'actor') and len(self.model.actor) > 0:
            self.model.actor[0].register_forward_hook(get_activation('actor_hidden'))
        if hasattr(self.model, 'critic') and len(self.model.critic) > 0:
            self.model.critic[0].register_forward_hook(get_activation('critic_hidden'))

    def forward(self, state: torch.Tensor):
        """Forward pass through the model."""
        return self.model.forward(state)

    def get_action_with_details(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Dict:
        """
        Get action with full details for visualization.

        Args:
            state: Input state tensor of shape (batch_size, state_dim)
            deterministic: If True, return argmax action instead of sampling

        Returns:
            Dictionary containing:
                - action: Selected action index
                - probs: Probability distribution over actions
                - value: Value estimate
                - entropy: Policy entropy
                - confidence: Max probability (confidence score)
                - activations: 2D array of encoder activations for heatmap
        """
        self.activations = {}

        with torch.no_grad():
            action_logits, value = self.forward(state)
            probs = torch.softmax(action_logits, dim=-1)

            # Compute entropy
            log_probs = torch.log_softmax(action_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)

            if deterministic:
                action = action_logits.argmax(dim=-1)
            else:
                action = torch.multinomial(probs, 1).squeeze(-1)

            # Confidence = max probability
            confidence = probs.max(dim=-1).values

        # Format activations for heatmap (4 rows x 32 cols)
        activations_2d = None
        if 'encoder_2' in self.activations:
            act = self.activations['encoder_2'][0].cpu().numpy()
            n = len(act)
            rows = 4
            cols = min(32, n // rows)
            if rows * cols > 0:
                activations_2d = act[:rows*cols].reshape(rows, cols).tolist()

        return {
            'action': action.item(),
            'probs': probs.cpu().numpy()[0].tolist(),
            'value': value.item(),
            'entropy': entropy.item(),
            'confidence': confidence.item(),
            'activations': activations_2d
        }


def load_model(
    model_path: Path,
    strategy_dir: Path,
    device: str = "cpu"
) -> ActorCriticWithActivations:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to the model checkpoint
        strategy_dir: Path to the strategy directory (for importing model class)
        device: Device to load model onto

    Returns:
        Loaded model with activation hooks
    """
    # Add strategy directory to path for importing
    strategy_dir = Path(strategy_dir)
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    # Import the model class and config from the strategy
    try:
        from model import ActorCritic
        from config import PPOConfig
    except ImportError as e:
        raise ImportError(
            f"Could not import model/config from strategy directory: {strategy_dir}\n"
            f"Make sure model.py and config.py exist in the strategy directory.\n"
            f"Error: {e}"
        )

    # Create the model
    config = PPOConfig(device=device)
    base_model = ActorCritic(config)

    # Load weights
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
        print(f"  Trained for {checkpoint.get('timestep', 'unknown')} steps")
    else:
        print(f"WARNING: Model not found at {model_path}, using random policy")

    base_model.to(device)
    base_model.eval()

    # Wrap with activation hooks
    return ActorCriticWithActivations(base_model)
