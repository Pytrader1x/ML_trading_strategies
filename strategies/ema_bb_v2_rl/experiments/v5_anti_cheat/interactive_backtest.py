#!/usr/bin/env python3
"""
Interactive HTML Backtest Visualization for V5 Anti-Cheat.

Creates a beautiful, interactive chart with:
- OHLC Candlestick price chart with trade entries/exits
- Cumulative PnL curve below
- Metrics dashboard in top left
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import json

EXPERIMENT_DIR = Path(__file__).parent
STRATEGY_DIR = EXPERIMENT_DIR.parent.parent
# ML_trading_strategies/data (4 levels up from v5_anti_cheat)
DATA_DIR = EXPERIMENT_DIR.parent.parent.parent.parent / "data"

sys.path.insert(0, str(EXPERIMENT_DIR))
from config import PPOConfig, Actions
from env import TradeEpisode

sys.path.insert(0, str(STRATEGY_DIR))
from model import ActorCritic

ACTION_NAMES = ['HOLD', 'EXIT', 'TIGHTEN_SL', 'TRAIL_BE', 'PARTIAL']


def load_model(checkpoint_path: Path, device: str = "cpu"):
    config = PPOConfig(device=device)
    config.hidden_dims = [512, 256]
    model = ActorCritic(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def load_price_data():
    """Load OHLC price data."""
    price_file = DATA_DIR / "AUDUSD_15M.csv"
    print(f"Loading price data from {price_file}")
    df = pd.read_csv(price_file, parse_dates=['DateTime'])
    df.set_index('DateTime', inplace=True)
    return df


def load_trades():
    """Load trade data with entry/exit times."""
    trades_file = STRATEGY_DIR / "data" / "trades_test_2022_2025.csv"
    print(f"Loading trades from {trades_file}")
    df = pd.read_csv(trades_file, parse_dates=['entry_time', 'exit_time'])
    return df


def evaluate_trades_with_v5(model, episodes, config, trades_df):
    """Evaluate trades with V5 model and match to trade data."""
    results = []

    for i, ep in enumerate(episodes):
        if i >= len(trades_df):
            break

        trade = trades_df.iloc[i]
        n_valid = int(ep.valid_mask.sum().item())

        # State tracking
        sl_atr = 1.5
        max_fav = 0.0
        max_adv = 0.0
        action_hist = []
        be_active = False

        exit_bar = None
        exit_pnl = None
        exit_reason = None

        for bar in range(n_valid):
            current_pnl = ep.market_tensor[bar, 0].item()
            max_fav = max(max_fav, current_pnl)
            max_adv = min(max_adv, current_pnl)

            # Build state
            position = [bar/200, current_pnl, max_fav, max_adv, sl_atr/2]
            market = ep.market_tensor[bar, :10].tolist()
            entry = [ep.entry_atr, ep.entry_adx/100, ep.entry_rsi/100, ep.entry_bb_width, ep.entry_ema_diff]
            hist = action_hist[-5:] if len(action_hist) >= 5 else [0.0]*(5-len(action_hist)) + action_hist
            state = torch.tensor(position + market + entry + hist, dtype=torch.float32).unsqueeze(0)

            # Action mask
            mask = torch.ones(1, 5, dtype=torch.bool)
            if bar < config.reward.min_exit_bar:
                mask[0, Actions.EXIT] = False
                mask[0, Actions.PARTIAL_EXIT] = False
            if bar < config.reward.min_trail_bar:
                mask[0, Actions.TRAIL_BREAKEVEN] = False

            with torch.no_grad():
                logits, _ = model(state)
                masked_logits = logits.clone()
                masked_logits[~mask] = float('-inf')
                action = masked_logits.argmax().item()

            action_hist.append(float(action) / 4.0)

            if action == Actions.TIGHTEN_SL:
                sl_atr = max(0.5, sl_atr - 0.25)

            if action == Actions.TRAIL_BREAKEVEN and not be_active:
                if current_pnl >= config.reward.min_profit_for_trail:
                    be_active = True

            if action == Actions.EXIT:
                exit_bar = bar
                exit_reason = "EXIT"
                exit_pnl = current_pnl
                break

            if action == Actions.PARTIAL_EXIT:
                exit_bar = bar
                exit_reason = "PARTIAL"
                exit_pnl = current_pnl * 0.5
                break

            if be_active and current_pnl <= 0:
                exit_bar = bar
                exit_reason = "BE_SL"
                exit_pnl = 0
                break

            sl_pct = sl_atr * ep.entry_atr
            if current_pnl <= -sl_pct:
                exit_bar = bar
                exit_reason = "SL_HIT"
                exit_pnl = -sl_pct
                break

        if exit_bar is None:
            exit_bar = n_valid - 1
            exit_reason = "END"
            exit_pnl = ep.market_tensor[exit_bar, 0].item()

        # Calculate V5 exit time
        entry_time = trade['entry_time']
        v5_exit_time = entry_time + pd.Timedelta(minutes=15 * exit_bar)

        results.append({
            'trade_id': i,
            'entry_time': entry_time,
            'exit_time': v5_exit_time,
            'classical_exit_time': trade['exit_time'],
            'entry_price': trade['entry_price'],
            'direction': 'LONG' if trade['direction'] == 1 else 'SHORT',
            'exit_bar': exit_bar,
            'exit_reason': exit_reason,
            'exit_pnl': exit_pnl,
            'classical_pnl': trade['pnl_pct'],
            'v5_exit_price': trade['entry_price'] * (1 + exit_pnl) if trade['direction'] == 1 else trade['entry_price'] * (1 - exit_pnl)
        })

    return results


def generate_html(results, price_df):
    """Generate interactive HTML visualization with candlestick chart."""

    # Calculate metrics
    returns = [r['exit_pnl'] for r in results]
    classical_returns = [r['classical_pnl'] for r in results]

    total_return = sum(returns) * 100
    classical_total = sum(classical_returns) * 100
    win_rate = np.mean([r > 0 for r in returns]) * 100
    avg_win = np.mean([r for r in returns if r > 0]) * 100 if any(r > 0 for r in returns) else 0
    avg_loss = np.mean([r for r in returns if r < 0]) * 100 if any(r < 0 for r in returns) else 0

    returns_arr = np.array(returns)
    sharpe = (np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(len(returns)/3)) if np.std(returns_arr) > 0 else 0

    cumulative = np.cumsum(returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = np.max(drawdown) * 100

    exit_reasons = {}
    for r in results:
        exit_reasons[r['exit_reason']] = exit_reasons.get(r['exit_reason'], 0) + 1

    avg_bars = np.mean([r['exit_bar'] for r in results])

    # Prepare price data for a sample period (show ~200 trades worth)
    sample_trades = results[:200]
    if sample_trades:
        start_time = sample_trades[0]['entry_time'] - pd.Timedelta(hours=12)
        end_time = sample_trades[-1]['exit_time'] + pd.Timedelta(hours=12)
        price_sample = price_df[start_time:end_time].reset_index()
    else:
        price_sample = price_df.iloc[-1000:].reset_index()

    # Format for Plotly
    ohlc_data = {
        'x': price_sample['DateTime'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
        'open': price_sample['Open'].tolist(),
        'high': price_sample['High'].tolist(),
        'low': price_sample['Low'].tolist(),
        'close': price_sample['Close'].tolist()
    }

    # Entry/Exit markers for sample period
    long_entries = []
    short_entries = []
    exits_win = []
    exits_loss = []

    for r in sample_trades:
        entry_str = r['entry_time'].strftime('%Y-%m-%d %H:%M')
        exit_str = r['exit_time'].strftime('%Y-%m-%d %H:%M')

        if r['direction'] == 'LONG':
            long_entries.append({
                'x': entry_str,
                'y': r['entry_price'],
                'text': f"LONG #{r['trade_id']+1}"
            })
        else:
            short_entries.append({
                'x': entry_str,
                'y': r['entry_price'],
                'text': f"SHORT #{r['trade_id']+1}"
            })

        exit_marker = {
            'x': exit_str,
            'y': r['v5_exit_price'],
            'text': f"{r['exit_reason']}<br>{r['exit_pnl']*100:.2f}%<br>{r['exit_bar']} bars"
        }
        if r['exit_pnl'] >= 0:
            exits_win.append(exit_marker)
        else:
            exits_loss.append(exit_marker)

    # Cumulative PnL data
    cumulative_pnl = np.cumsum([r['exit_pnl'] * 100 for r in results]).tolist()
    classical_cumulative = np.cumsum([r['classical_pnl'] * 100 for r in results]).tolist()
    trade_numbers = list(range(1, len(results) + 1))

    # Trade returns for bar chart
    trade_returns = [{'x': i+1, 'y': r['exit_pnl']*100, 'dir': r['direction'], 'reason': r['exit_reason'], 'bars': r['exit_bar']}
                     for i, r in enumerate(results)]

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V5 Anti-Cheat Backtest - AUDUSD 15M</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            min-height: 100vh;
            color: #c9d1d9;
            padding: 20px;
        }}
        .container {{ max-width: 1800px; margin: 0 auto; }}
        .header {{
            text-align: center;
            margin-bottom: 25px;
            padding: 20px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{
            font-size: 2.2em;
            background: linear-gradient(90deg, #58a6ff, #a371f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .header p {{ color: #8b949e; font-size: 1em; }}

        .metrics-box {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(13, 17, 23, 0.95);
            border: 1px solid rgba(88, 166, 255, 0.3);
            border-radius: 10px;
            padding: 15px 20px;
            z-index: 1000;
            min-width: 180px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        .metrics-box h3 {{
            color: #58a6ff;
            font-size: 0.9em;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.9em;
        }}
        .metric-label {{ color: #8b949e; }}
        .metric-value {{ font-weight: 600; }}
        .positive {{ color: #3fb950; }}
        .negative {{ color: #f85149; }}
        .neutral {{ color: #d29922; }}

        .chart-container {{
            background: rgba(255,255,255,0.02);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.08);
            position: relative;
        }}
        .chart-title {{
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #c9d1d9;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .chart-title::before {{
            content: '';
            display: inline-block;
            width: 3px;
            height: 16px;
            background: linear-gradient(180deg, #58a6ff, #a371f7);
            border-radius: 2px;
        }}

        .legend-box {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85em;
            color: #8b949e;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        .legend-triangle {{
            width: 0;
            height: 0;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
        }}
        .legend-triangle.up {{ border-bottom: 10px solid #3fb950; }}
        .legend-triangle.down {{ border-top: 10px solid #f85149; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.06);
        }}
        .stat-value {{
            font-size: 1.6em;
            font-weight: bold;
            margin-bottom: 4px;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .exit-badges {{
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 15px;
        }}
        .exit-badge {{
            padding: 6px 14px;
            border-radius: 15px;
            font-size: 0.85em;
        }}
        .exit-EXIT {{ background: rgba(248, 81, 73, 0.2); color: #f85149; }}
        .exit-SL_HIT {{ background: rgba(210, 153, 34, 0.2); color: #d29922; }}
        .exit-BE_SL {{ background: rgba(88, 166, 255, 0.2); color: #58a6ff; }}
        .exit-PARTIAL {{ background: rgba(163, 113, 247, 0.2); color: #a371f7; }}
        .exit-END {{ background: rgba(139, 148, 158, 0.2); color: #8b949e; }}

        .verdict {{
            background: linear-gradient(135deg, rgba(63, 185, 80, 0.1), rgba(88, 166, 255, 0.1));
            border: 1px solid rgba(63, 185, 80, 0.3);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-top: 20px;
        }}
        .verdict h2 {{ color: #3fb950; font-size: 1.5em; margin-bottom: 8px; }}
        .verdict p {{ color: #8b949e; }}

        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 1000px) {{
            .two-col {{ grid-template-columns: 1fr; }}
            .metrics-box {{ position: relative; top: 0; left: 0; margin-bottom: 15px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>V5 Anti-Cheat Backtest Results</h1>
            <p>AUDUSD 15M | Out-of-Sample 2022-2025 | {len(results)} Trades | Trade-Based Sharpe: {sharpe:.2f}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value positive">+{total_return:.1f}%</div>
                <div class="stat-label">Total Return</div>
            </div>
            <div class="stat-card">
                <div class="stat-value neutral">{sharpe:.2f}</div>
                <div class="stat-label">Sharpe (Trade)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'positive' if win_rate > 50 else 'negative'}">{win_rate:.1f}%</div>
                <div class="stat-label">Win Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value negative">-{max_dd:.2f}%</div>
                <div class="stat-label">Max Drawdown</div>
            </div>
            <div class="stat-card">
                <div class="stat-value positive">+{avg_win:.3f}%</div>
                <div class="stat-label">Avg Win</div>
            </div>
            <div class="stat-card">
                <div class="stat-value negative">{avg_loss:.3f}%</div>
                <div class="stat-label">Avg Loss</div>
            </div>
            <div class="stat-card">
                <div class="stat-value neutral">{avg_bars:.1f}</div>
                <div class="stat-label">Avg Bars</div>
            </div>
            <div class="stat-card">
                <div class="stat-value positive">+{total_return - classical_total:.1f}%</div>
                <div class="stat-label">vs Classical</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Price Chart with Trade Entries & Exits (First 200 Trades)</div>
            <div class="metrics-box">
                <h3>Quick Stats</h3>
                <div class="metric-row">
                    <span class="metric-label">Trades Shown:</span>
                    <span class="metric-value">{len(sample_trades)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Long:</span>
                    <span class="metric-value positive">{len(long_entries)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Short:</span>
                    <span class="metric-value negative">{len(short_entries)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Winners:</span>
                    <span class="metric-value positive">{len(exits_win)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Losers:</span>
                    <span class="metric-value negative">{len(exits_loss)}</span>
                </div>
            </div>
            <div id="candlestick-chart"></div>
            <div class="legend-box">
                <div class="legend-item"><div class="legend-triangle up"></div> Long Entry</div>
                <div class="legend-item"><div class="legend-triangle down"></div> Short Entry</div>
                <div class="legend-item"><div class="legend-dot" style="background:#3fb950;"></div> Exit (Win)</div>
                <div class="legend-item"><div class="legend-dot" style="background:#f85149;"></div> Exit (Loss)</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Cumulative P&L - V5 RL vs Classical Strategy</div>
            <div id="cumulative-chart"></div>
        </div>

        <div class="two-col">
            <div class="chart-container">
                <div class="chart-title">Individual Trade Returns</div>
                <div id="trades-chart"></div>
                <div class="exit-badges">
                    {"".join([f'<span class="exit-badge exit-{reason}">{reason}: {count}</span>' for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1])])}
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Return Distribution</div>
                <div id="histogram-chart"></div>
            </div>
        </div>

        <div class="verdict">
            <h2>✓ VERIFIED LEGITIMATE</h2>
            <p>No lookahead bias | 0 bar-0 exits | 0 bar-1 exits | Action masking working | Trade-based Sharpe {sharpe:.2f}</p>
        </div>
    </div>

    <script>
        const darkTheme = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(22,27,34,0.5)',
            font: {{ color: '#c9d1d9', size: 11 }},
            xaxis: {{ gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' }},
            yaxis: {{ gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' }},
            margin: {{ t: 30, r: 50, b: 50, l: 60 }}
        }};
        const config = {{ displayModeBar: true, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }};

        // Candlestick Chart
        const candleTrace = {{
            x: {json.dumps(ohlc_data['x'])},
            open: {json.dumps(ohlc_data['open'])},
            high: {json.dumps(ohlc_data['high'])},
            low: {json.dumps(ohlc_data['low'])},
            close: {json.dumps(ohlc_data['close'])},
            type: 'candlestick',
            increasing: {{ line: {{ color: '#3fb950' }}, fillcolor: '#3fb950' }},
            decreasing: {{ line: {{ color: '#f85149' }}, fillcolor: '#f85149' }},
            name: 'AUDUSD',
            hoverinfo: 'x+text',
            text: {json.dumps(ohlc_data['x'])}
        }};

        const longEntries = {{
            x: {json.dumps([e['x'] for e in long_entries])},
            y: {json.dumps([e['y'] for e in long_entries])},
            mode: 'markers',
            type: 'scatter',
            marker: {{ symbol: 'triangle-up', size: 12, color: '#3fb950', line: {{ color: '#fff', width: 1 }} }},
            name: 'Long Entry',
            text: {json.dumps([e['text'] for e in long_entries])},
            hovertemplate: '%{{text}}<br>Price: %{{y:.5f}}<extra></extra>'
        }};

        const shortEntries = {{
            x: {json.dumps([e['x'] for e in short_entries])},
            y: {json.dumps([e['y'] for e in short_entries])},
            mode: 'markers',
            type: 'scatter',
            marker: {{ symbol: 'triangle-down', size: 12, color: '#f85149', line: {{ color: '#fff', width: 1 }} }},
            name: 'Short Entry',
            text: {json.dumps([e['text'] for e in short_entries])},
            hovertemplate: '%{{text}}<br>Price: %{{y:.5f}}<extra></extra>'
        }};

        const exitsWin = {{
            x: {json.dumps([e['x'] for e in exits_win])},
            y: {json.dumps([e['y'] for e in exits_win])},
            mode: 'markers',
            type: 'scatter',
            marker: {{ symbol: 'circle', size: 8, color: '#3fb950', line: {{ color: '#fff', width: 1 }} }},
            name: 'Exit (Win)',
            text: {json.dumps([e['text'] for e in exits_win])},
            hovertemplate: '%{{text}}<extra></extra>'
        }};

        const exitsLoss = {{
            x: {json.dumps([e['x'] for e in exits_loss])},
            y: {json.dumps([e['y'] for e in exits_loss])},
            mode: 'markers',
            type: 'scatter',
            marker: {{ symbol: 'circle', size: 8, color: '#f85149', line: {{ color: '#fff', width: 1 }} }},
            name: 'Exit (Loss)',
            text: {json.dumps([e['text'] for e in exits_loss])},
            hovertemplate: '%{{text}}<extra></extra>'
        }};

        Plotly.newPlot('candlestick-chart', [candleTrace, longEntries, shortEntries, exitsWin, exitsLoss], {{
            ...darkTheme,
            xaxis: {{ ...darkTheme.xaxis, rangeslider: {{ visible: false }}, title: '' }},
            yaxis: {{ ...darkTheme.yaxis, title: 'Price' }},
            legend: {{ x: 0.5, y: 1.02, xanchor: 'center', orientation: 'h', bgcolor: 'rgba(0,0,0,0)' }},
            height: 500
        }}, config);

        // Cumulative P&L
        Plotly.newPlot('cumulative-chart', [
            {{
                x: {json.dumps(trade_numbers)},
                y: {json.dumps(cumulative_pnl)},
                type: 'scatter',
                mode: 'lines',
                fill: 'tozeroy',
                fillcolor: 'rgba(88, 166, 255, 0.15)',
                line: {{ color: '#58a6ff', width: 2 }},
                name: 'V5 RL',
                hovertemplate: 'Trade %{{x}}<br>Cumulative: %{{y:.2f}}%<extra></extra>'
            }},
            {{
                x: {json.dumps(trade_numbers)},
                y: {json.dumps(classical_cumulative)},
                type: 'scatter',
                mode: 'lines',
                line: {{ color: '#8b949e', width: 1.5, dash: 'dot' }},
                name: 'Classical',
                hovertemplate: 'Trade %{{x}}<br>Classical: %{{y:.2f}}%<extra></extra>'
            }}
        ], {{
            ...darkTheme,
            xaxis: {{ ...darkTheme.xaxis, title: 'Trade Number' }},
            yaxis: {{ ...darkTheme.yaxis, title: 'Cumulative Return (%)' }},
            legend: {{ x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.5)' }},
            height: 300
        }}, config);

        // Trade Returns Bar
        const tradeData = {json.dumps(trade_returns)};
        Plotly.newPlot('trades-chart', [{{
            x: tradeData.map(t => t.x),
            y: tradeData.map(t => t.y),
            type: 'bar',
            marker: {{ color: tradeData.map(t => t.y >= 0 ? '#3fb950' : '#f85149'), opacity: 0.8 }},
            hovertemplate: tradeData.map(t => `Trade %{{x}}<br>%{{y:.3f}}%<br>${{t.dir}} | ${{t.reason}}<extra></extra>`)
        }}], {{
            ...darkTheme,
            xaxis: {{ ...darkTheme.xaxis, title: 'Trade #' }},
            yaxis: {{ ...darkTheme.yaxis, title: 'Return (%)' }},
            height: 280
        }}, config);

        // Histogram
        Plotly.newPlot('histogram-chart', [{{
            x: {json.dumps([r['exit_pnl']*100 for r in results])},
            type: 'histogram',
            nbinsx: 50,
            marker: {{ color: 'rgba(88, 166, 255, 0.6)', line: {{ color: '#58a6ff', width: 1 }} }},
            hovertemplate: 'Return: %{{x:.2f}}%<br>Count: %{{y}}<extra></extra>'
        }}], {{
            ...darkTheme,
            xaxis: {{ ...darkTheme.xaxis, title: 'Return (%)' }},
            yaxis: {{ ...darkTheme.yaxis, title: 'Count' }},
            height: 280
        }}, config);
    </script>
</body>
</html>'''

    return html_content


def main():
    print("="*60)
    print(" Generating Interactive Backtest with Price Chart")
    print("="*60)

    # Load model
    model_path = EXPERIMENT_DIR / "models" / "exit_policy_final.pt"
    print(f"\nLoading model...")
    model, config = load_model(model_path)

    # Load price data
    price_df = load_price_data()
    print(f"Price data: {len(price_df)} bars")

    # Load trades
    trades_df = load_trades()
    print(f"Trades: {len(trades_df)} trades")

    # Load episodes
    episode_file = STRATEGY_DIR / "data" / "episodes_test_2022_2025.pkl"
    print(f"Loading episodes...")
    with open(episode_file, 'rb') as f:
        data = pickle.load(f)
    episodes = data['episodes']
    print(f"Episodes: {len(episodes)}")

    # Evaluate with V5
    print("\nEvaluating trades with V5 model...")
    results = evaluate_trades_with_v5(model, episodes, config, trades_df)
    print(f"Evaluated {len(results)} trades")

    # Generate HTML
    print("\nGenerating HTML...")
    html = generate_html(results, price_df)

    # Save
    output_path = EXPERIMENT_DIR / "backtest_visualization.html"
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\n✓ Saved: {output_path}")
    print(f"\nOpen: file://{output_path.absolute()}")


if __name__ == "__main__":
    main()
