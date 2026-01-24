#!/usr/bin/env python3
"""
Real-Time RL Trade Visualizer V2

Beautiful visualization of RL agent making exit decisions on OOS data.
Features:
- Large, visible entry/exit markers with PnL labels
- Rolling trade history table
- Neural network activation heatmaps
- Model output details (value, entropy, confidence)
- Real metrics that update properly

Usage:
    python visualizer.py --speed 10  # 10x speed
    python visualizer.py --speed 1   # Real-time

Then open: http://localhost:8765
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import webbrowser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import ActorCritic
from config import PPOConfig, Actions

# ============================================================================
# Configuration
# ============================================================================

STRATEGY_DIR = Path(__file__).parent
DATA_DIR = STRATEGY_DIR / "data"
MODEL_PATH = STRATEGY_DIR / "models" / "exit_policy_insample_2005_2021.pt"

# Visualization settings
WINDOW_BARS = 96  # 96 bars = 24 hours of 15M data
PORT = 8765

# FX PnL Calculation Constants (Production Backtest Engine compatible)
POSITION_SIZE = 2_000_000  # $2M notional position
PIP_SIZE = 0.0001  # Standard pip for AUDUSD
PIP_VALUE = POSITION_SIZE * PIP_SIZE  # $200 per pip for 2M position
OOS_START = '2022-01-01'
OOS_END = '2025-12-31'


def format_datetime_pretty(dt_str: str) -> str:
    """
    Format datetime string to human-readable format.
    e.g., "2024-01-23 11:30:00" -> "Tue 23rd Jan 11:30am"
    """
    try:
        if isinstance(dt_str, str):
            # Handle common datetime string formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S']:
                try:
                    dt = datetime.strptime(dt_str[:19], fmt[:len(dt_str[:19])+2])
                    break
                except ValueError:
                    continue
            else:
                return dt_str  # Return original if parsing fails
        else:
            dt = pd.to_datetime(dt_str)

        # Get day suffix (1st, 2nd, 3rd, etc.)
        day = dt.day
        if 10 <= day % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')

        # Format: "Tue 23rd Jan 11:30am"
        weekday = dt.strftime('%a')
        month = dt.strftime('%b')
        time_str = dt.strftime('%I:%M%p').lower().lstrip('0')

        return f"{weekday} {day}{suffix} {month} {time_str}"
    except Exception:
        return str(dt_str)  # Return original on any error

# ============================================================================
# HTML Template - MAJOR IMPROVEMENTS
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL Exit Optimizer - Live Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
            color: #e6edf3;
            min-height: 100vh;
            overflow-x: hidden;
        }
        .header {
            background: rgba(22, 27, 34, 0.95);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #30363d;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header h1 {
            font-size: 20px;
            font-weight: 600;
            background: linear-gradient(90deg, #58a6ff, #a371f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.connected { background: #3fb950; box-shadow: 0 0 8px #3fb950; }
        .status-dot.disconnected { background: #f85149; }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 340px;
            grid-template-rows: 1fr auto;
            gap: 12px;
            padding: 12px;
            height: calc(100vh - 56px);
        }
        .main-panel {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .chart-container {
            background: rgba(22, 27, 34, 0.8);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 12px;
            flex: 1;
            min-height: 380px;
        }
        .side-panel {
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow-y: auto;
        }
        .trade-log-container {
            grid-column: 1 / -1;
            background: rgba(22, 27, 34, 0.95);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 14px;
            max-height: 450px;
            overflow-y: auto;
            overflow-x: auto;
        }
        .trade-log-container h3 {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #58a6ff;
            margin-bottom: 10px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        /* Trade Card Layout */
        .trade-card {
            background: rgba(13, 17, 23, 0.8);
            border: 1px solid #30363d;
            border-radius: 8px;
            margin-bottom: 12px;
            overflow: hidden;
        }
        .trade-card.active {
            border: 2px solid #a371f7;
            box-shadow: 0 0 15px rgba(163, 113, 247, 0.3);
            animation: card-pulse 2s infinite;
        }
        @keyframes card-pulse {
            0%, 100% { box-shadow: 0 0 15px rgba(163, 113, 247, 0.3); }
            50% { box-shadow: 0 0 25px rgba(163, 113, 247, 0.5); }
        }
        .trade-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            background: rgba(48, 54, 61, 0.5);
            border-bottom: 1px solid #30363d;
        }
        .trade-entry-info {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .trade-direction {
            font-weight: 700;
            font-size: 13px;
            padding: 4px 10px;
            border-radius: 4px;
        }
        .trade-direction.long { background: rgba(63, 185, 80, 0.2); color: #3fb950; }
        .trade-direction.short { background: rgba(248, 81, 73, 0.2); color: #f85149; }
        .trade-details {
            font-size: 11px;
            color: #c9d1d9;
        }
        .trade-details .label { color: #8b949e; }
        .trade-details .value { font-weight: 600; font-family: 'SF Mono', monospace; }
        .trade-levels {
            display: flex;
            gap: 12px;
            font-size: 10px;
        }
        .trade-levels .sl { color: #f85149; }
        .trade-levels .tp { color: #3fb950; }
        .trade-result {
            text-align: right;
        }
        .trade-result .pnl {
            font-size: 16px;
            font-weight: 700;
            font-family: 'SF Mono', monospace;
        }
        .trade-result .pnl.positive { color: #3fb950; }
        .trade-result .pnl.negative { color: #f85149; }
        .trade-result .pips {
            font-size: 11px;
            color: #8b949e;
        }
        .active-badge {
            display: inline-block;
            padding: 3px 8px;
            background: linear-gradient(135deg, #a371f7, #8957e5);
            color: #fff;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.5px;
            animation: pulse-badge 1.5s infinite;
        }
        @keyframes pulse-badge {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        /* Action Timeline */
        .action-timeline {
            display: flex;
            overflow-x: auto;
            padding: 10px 14px;
            gap: 8px;
            background: rgba(13, 17, 23, 0.5);
        }
        .action-item {
            flex: 0 0 auto;
            min-width: 130px;
            max-width: 160px;
            background: rgba(48, 54, 61, 0.6);
            border-radius: 6px;
            padding: 8px 10px;
            border-left: 3px solid #30363d;
            font-size: 10px;
        }
        .action-item.hold { border-left-color: #8b949e; }
        .action-item.partial { border-left-color: #a371f7; background: rgba(163, 113, 247, 0.1); }
        .action-item.tighten { border-left-color: #d29922; background: rgba(210, 153, 34, 0.1); }
        .action-item.trail { border-left-color: #58a6ff; background: rgba(88, 166, 255, 0.1); }
        .action-item.exit { border-left-color: #f85149; background: rgba(248, 81, 73, 0.1); }
        .action-item.total-win { border-left-color: #3fb950; background: rgba(63, 185, 80, 0.15); border-width: 4px; }
        .action-item.total-loss { border-left-color: #f85149; background: rgba(248, 81, 73, 0.15); border-width: 4px; }
        .action-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        .action-type {
            font-weight: 700;
            text-transform: uppercase;
            font-size: 9px;
            letter-spacing: 0.5px;
        }
        .action-type.hold { color: #8b949e; }
        .action-type.partial { color: #a371f7; }
        .action-type.tighten { color: #d29922; }
        .action-type.trail { color: #58a6ff; }
        .action-type.exit { color: #f85149; }
        .action-bar {
            font-size: 9px;
            color: #6e7681;
            background: rgba(48, 54, 61, 0.8);
            padding: 1px 5px;
            border-radius: 3px;
        }
        .action-time {
            font-size: 9px;
            color: #8b949e;
            margin-bottom: 3px;
        }
        .action-price {
            font-family: 'SF Mono', monospace;
            color: #c9d1d9;
        }
        .action-pnl {
            font-weight: 600;
            font-family: 'SF Mono', monospace;
        }
        .action-pnl.positive { color: #3fb950; }
        .action-pnl.negative { color: #f85149; }
        .action-size {
            color: #8b949e;
            font-size: 9px;
        }
        .action-note {
            color: #6e7681;
            font-size: 9px;
            margin-top: 3px;
            font-style: italic;
        }
        .no-actions {
            color: #6e7681;
            font-style: italic;
            padding: 10px;
        }
        .pnl-breakdown {
            background: rgba(163, 113, 247, 0.08);
            border-left: 3px solid #a371f7;
            padding: 6px 10px;
            margin: 4px 0;
            font-size: 10px;
        }
        .panel {
            background: rgba(22, 27, 34, 0.8);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 14px;
        }
        .panel h3 {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #8b949e;
            margin-bottom: 12px;
            font-weight: 600;
        }

        /* Action Display */
        .action-display {
            text-align: center;
            padding: 10px;
        }
        .action-name {
            font-size: 36px;
            font-weight: 800;
            margin-bottom: 8px;
            transition: all 0.2s ease;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .action-HOLD { color: #8b949e; }
        .action-EXIT { color: #f85149; text-shadow: 0 0 20px rgba(248,81,73,0.5); }
        .action-TIGHTEN_SL { color: #d29922; text-shadow: 0 0 20px rgba(210,153,34,0.5); }
        .action-TRAIL_BREAKEVEN { color: #58a6ff; text-shadow: 0 0 20px rgba(88,166,255,0.5); }
        .action-PARTIAL_EXIT { color: #a371f7; text-shadow: 0 0 20px rgba(163,113,247,0.5); }

        .action-probs {
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-top: 12px;
        }
        .prob-bar {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .prob-label {
            width: 85px;
            font-size: 10px;
            text-align: right;
            color: #8b949e;
            font-weight: 500;
        }
        .prob-fill-container {
            flex: 1;
            height: 18px;
            background: rgba(48, 54, 61, 0.8);
            border-radius: 4px;
            overflow: hidden;
        }
        .prob-fill {
            height: 100%;
            transition: width 0.2s ease;
            border-radius: 4px;
        }
        .prob-value {
            width: 50px;
            font-size: 12px;
            font-weight: 700;
            font-family: 'SF Mono', monospace;
        }

        /* Model Output Panel */
        .model-output {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
        }
        .model-stat {
            background: rgba(48, 54, 61, 0.5);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .model-stat-value {
            font-size: 18px;
            font-weight: 700;
            font-family: 'SF Mono', monospace;
        }
        .model-stat-label {
            font-size: 9px;
            color: #8b949e;
            margin-top: 2px;
            text-transform: uppercase;
        }

        /* Position State */
        .state-features {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .feature {
            background: rgba(48, 54, 61, 0.5);
            padding: 10px;
            border-radius: 8px;
        }
        .feature-name {
            font-size: 9px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .feature-value {
            font-size: 18px;
            font-weight: 700;
            margin-top: 4px;
            font-family: 'SF Mono', monospace;
        }
        .feature-value.positive { color: #3fb950; }
        .feature-value.negative { color: #f85149; }
        .feature-value.neutral { color: #8b949e; }

        /* Trade History Table */
        .trade-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }
        .trade-table th {
            text-align: left;
            padding: 8px 6px;
            border-bottom: 1px solid #30363d;
            color: #8b949e;
            font-weight: 600;
            font-size: 9px;
            text-transform: uppercase;
        }
        .trade-table td {
            padding: 8px 6px;
            border-bottom: 1px solid rgba(48, 54, 61, 0.5);
            font-family: 'SF Mono', monospace;
        }
        .trade-table tr:hover {
            background: rgba(48, 54, 61, 0.3);
        }
        .trade-table .pnl-positive { color: #3fb950; font-weight: 700; }
        .trade-table .pnl-negative { color: #f85149; font-weight: 700; }
        .trade-table .direction-long { color: #3fb950; }
        .trade-table .direction-short { color: #f85149; }

        /* Session Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
        }
        .stat-box {
            text-align: center;
            padding: 12px 8px;
            background: rgba(48, 54, 61, 0.5);
            border-radius: 8px;
        }
        .stat-value {
            font-size: 22px;
            font-weight: 800;
            font-family: 'SF Mono', monospace;
        }
        .stat-label {
            font-size: 9px;
            color: #8b949e;
            margin-top: 4px;
            text-transform: uppercase;
        }

        /* Activations */
        .activations-container {
            height: 100px;
        }

        /* Controls */
        .controls-panel {
            background: rgba(22, 27, 34, 0.8);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 12px 20px;
        }
        .controls {
            display: flex;
            gap: 12px;
            justify-content: center;
            align-items: center;
        }
        .btn {
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-primary {
            background: linear-gradient(135deg, #238636, #2ea043);
            color: white;
        }
        .btn-primary:hover {
            background: linear-gradient(135deg, #2ea043, #3fb950);
            transform: translateY(-1px);
        }
        .btn-secondary {
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
        }
        .btn-secondary:hover {
            background: #30363d;
        }
        .speed-control {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #8b949e;
            font-size: 13px;
        }
        .speed-slider {
            width: 120px;
            accent-color: #58a6ff;
        }
        .speed-value {
            font-family: 'SF Mono', monospace;
            font-weight: 600;
            color: #58a6ff;
            min-width: 40px;
        }

        #candlestick-chart, #activations-chart {
            width: 100%;
            height: 100%;
        }

        /* Current Trade Highlight */
        .current-trade {
            background: linear-gradient(135deg, rgba(88,166,255,0.1), rgba(163,113,247,0.1));
            border: 1px solid rgba(88,166,255,0.3);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .current-trade-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .current-trade-direction {
            font-size: 14px;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 4px;
        }
        .current-trade-direction.long { background: rgba(63,185,80,0.2); color: #3fb950; }
        .current-trade-direction.short { background: rgba(248,81,73,0.2); color: #f85149; }
        .current-trade-pnl {
            font-size: 24px;
            font-weight: 800;
            font-family: 'SF Mono', monospace;
        }
        .no-trade {
            text-align: center;
            color: #8b949e;
            padding: 20px;
            font-style: italic;
        }
        /* Comprehensive Metrics Panel */
        .metrics-panel {
            background: rgba(22, 27, 34, 0.95) !important;
        }
        .metrics-panel h3 {
            color: #58a6ff;
            margin-bottom: 10px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
        }
        .metric-box {
            background: rgba(48, 54, 61, 0.5);
            border-radius: 6px;
            padding: 8px 6px;
            text-align: center;
        }
        .metric-box.highlight {
            background: rgba(88, 166, 255, 0.1);
            border: 1px solid rgba(88, 166, 255, 0.3);
        }
        .metric-value {
            font-size: 14px;
            font-weight: 700;
            font-family: 'SF Mono', monospace;
            color: #e6edf3;
        }
        .metric-value.positive { color: #3fb950; }
        .metric-value.negative { color: #f85149; }
        .metric-value.long-color { color: #3fb950; }
        .metric-value.short-color { color: #f85149; }
        .metric-value.rl-color { color: #a371f7; }
        .metric-value.sl-color { color: #d29922; }
        .metric-label {
            font-size: 9px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            margin-top: 2px;
        }
        /* Controls panel fix - ensure visibility */
        .controls-panel {
            background: rgba(22, 27, 34, 0.95);
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 10px 15px;
            margin-top: 8px;
            z-index: 50;
        }
        .controls {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>RL Exit Optimizer - Live Visualization</h1>
        <div class="status">
            <div class="status-item">
                <div class="status-dot" id="connection-status"></div>
                <span id="connection-text">Connecting...</span>
            </div>
            <div class="status-item">
                <span id="timestamp" style="font-family: 'SF Mono', monospace;">--</span>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="main-panel">
            <div class="chart-container">
                <div id="candlestick-chart"></div>
            </div>
            <div class="controls-panel">
                <div class="controls">
                    <button class="btn btn-primary" id="play-btn" onclick="togglePlay()">
                        <span id="play-icon">&#9658;</span> Play
                    </button>
                    <button class="btn btn-secondary" onclick="stepForward()">Step &rarr;</button>
                    <button class="btn btn-secondary" onclick="nextTrade()" style="background: linear-gradient(135deg, #1f6feb, #388bfd); border: none;">
                        Next Trade &#9654;&#9654;
                    </button>
                    <button class="btn btn-secondary" onclick="resetPlayback()">Reset</button>
                    <div class="speed-control">
                        <span>Speed:</span>
                        <input type="range" class="speed-slider" id="speed-slider" min="1" max="100" value="10">
                        <span class="speed-value" id="speed-value">10x</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="side-panel">
            <!-- Current Trade -->
            <div class="panel">
                <h3>Active Position</h3>
                <div id="current-trade-container">
                    <div class="no-trade">No active trade</div>
                </div>
            </div>

            <!-- RL Action -->
            <div class="panel">
                <h3>RL Agent Decision</h3>
                <div class="action-display">
                    <div class="action-name" id="action-name">WAITING</div>
                    <div class="action-probs" id="action-probs">
                        <div class="prob-bar">
                            <span class="prob-label">HOLD</span>
                            <div class="prob-fill-container"><div class="prob-fill" id="prob-0" style="width:0%; background:#8b949e;"></div></div>
                            <span class="prob-value" id="prob-val-0">0.0%</span>
                        </div>
                        <div class="prob-bar">
                            <span class="prob-label">EXIT</span>
                            <div class="prob-fill-container"><div class="prob-fill" id="prob-1" style="width:0%; background:#f85149;"></div></div>
                            <span class="prob-value" id="prob-val-1">0.0%</span>
                        </div>
                        <div class="prob-bar">
                            <span class="prob-label">TIGHTEN SL</span>
                            <div class="prob-fill-container"><div class="prob-fill" id="prob-2" style="width:0%; background:#d29922;"></div></div>
                            <span class="prob-value" id="prob-val-2">0.0%</span>
                        </div>
                        <div class="prob-bar">
                            <span class="prob-label">TRAIL BE</span>
                            <div class="prob-fill-container"><div class="prob-fill" id="prob-3" style="width:0%; background:#58a6ff;"></div></div>
                            <span class="prob-value" id="prob-val-3">0.0%</span>
                        </div>
                        <div class="prob-bar">
                            <span class="prob-label">PARTIAL</span>
                            <div class="prob-fill-container"><div class="prob-fill" id="prob-4" style="width:0%; background:#a371f7;"></div></div>
                            <span class="prob-value" id="prob-val-4">0.0%</span>
                        </div>
                    </div>
                    <div class="model-output">
                        <div class="model-stat">
                            <div class="model-stat-value" id="model-value">0.00</div>
                            <div class="model-stat-label">Value Est.</div>
                        </div>
                        <div class="model-stat">
                            <div class="model-stat-value" id="model-entropy">0.00</div>
                            <div class="model-stat-label">Entropy</div>
                        </div>
                        <div class="model-stat">
                            <div class="model-stat-value" id="model-confidence">0%</div>
                            <div class="model-stat-label">Confidence</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Neural Activations -->
            <div class="panel">
                <h3>Neural Activations (Encoder Layer)</h3>
                <div class="activations-container">
                    <div id="activations-chart"></div>
                </div>
            </div>

            <!-- Comprehensive Metrics Panel -->
            <div class="panel metrics-panel" style="flex: 1;">
                <h3>📊 Performance Metrics</h3>
                <div class="metrics-grid">
                    <!-- Row 1: Key Metrics -->
                    <div class="metric-box highlight">
                        <div class="metric-value" id="metric-sharpe">0.00</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric-box highlight">
                        <div class="metric-value" id="metric-winrate">0%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-box highlight">
                        <div class="metric-value positive" id="metric-pnl">$0</div>
                        <div class="metric-label">Total P&L</div>
                    </div>
                    <div class="metric-box highlight">
                        <div class="metric-value" id="metric-return">0%</div>
                        <div class="metric-label">Return %</div>
                    </div>
                    <!-- Row 2: Trade Stats -->
                    <div class="metric-box">
                        <div class="metric-value" id="metric-trades">0</div>
                        <div class="metric-label">Trades</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="metric-pf">0.00</div>
                        <div class="metric-label">Profit Factor</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="metric-avgwin">$0</div>
                        <div class="metric-label">Avg Win</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="metric-avgloss">$0</div>
                        <div class="metric-label">Avg Loss</div>
                    </div>
                    <!-- Row 3: Long/Short -->
                    <div class="metric-box">
                        <div class="metric-value long-color" id="metric-longpct">0%</div>
                        <div class="metric-label">Long %</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value short-color" id="metric-shortpct">0%</div>
                        <div class="metric-label">Short %</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="metric-longwr">0%</div>
                        <div class="metric-label">Long WR</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="metric-shortwr">0%</div>
                        <div class="metric-label">Short WR</div>
                    </div>
                    <!-- Row 4: Duration & Drawdown -->
                    <div class="metric-box">
                        <div class="metric-value" id="metric-avgdur">0</div>
                        <div class="metric-label">Avg Bars</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value negative" id="metric-maxdd">$0</div>
                        <div class="metric-label">Max DD</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="metric-maxwin">$0</div>
                        <div class="metric-label">Best Trade</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="metric-maxloss">$0</div>
                        <div class="metric-label">Worst Trade</div>
                    </div>
                    <!-- Row 5: Exit Types -->
                    <div class="metric-box">
                        <div class="metric-value rl-color" id="metric-rlexit">0%</div>
                        <div class="metric-label">RL Exit %</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value sl-color" id="metric-slhit">0%</div>
                        <div class="metric-label">SL Hit %</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" id="metric-partials">0.0</div>
                        <div class="metric-label">Partials/Trade</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value positive" id="metric-pips">+0</div>
                        <div class="metric-label">Total Pips</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Detailed Trade Log (Full Width Below Chart) -->
        <div class="trade-log-container">
            <h3>
                <span style="color:#58a6ff;">📊</span> DETAILED TRADE LOG
                <span style="font-size:10px; color:#8b949e; font-weight:400; margin-left:auto;">
                    PIP VALUE: $200/pip | POSITION: 2M AUDUSD | Actions → scroll horizontally
                </span>
            </h3>
            <div id="trade-log-body">
                <div style="text-align:center; color:#8b949e; padding:30px;">Waiting for trades...</div>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let isPlaying = false;

        const actionNames = ['HOLD', 'EXIT', 'TIGHTEN_SL', 'TRAIL_BREAKEVEN', 'PARTIAL_EXIT'];
        const actionColors = ['#8b949e', '#f85149', '#d29922', '#58a6ff', '#a371f7'];

        // Initialize charts
        function initCharts() {
            // Candlestick chart with better styling
            Plotly.newPlot('candlestick-chart', [{
                type: 'candlestick',
                x: [],
                open: [],
                high: [],
                low: [],
                close: [],
                increasing: {line: {color: '#3fb950', width: 1}, fillcolor: '#238636'},
                decreasing: {line: {color: '#f85149', width: 1}, fillcolor: '#da3633'},
                name: 'AUDUSD'
            }], {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(13,17,23,0.5)',
                font: {color: '#8b949e', family: 'SF Pro Display'},
                xaxis: {
                    gridcolor: 'rgba(48,54,61,0.5)',
                    linecolor: '#30363d',
                    rangeslider: {visible: false},
                    tickfont: {size: 10}
                },
                yaxis: {
                    gridcolor: 'rgba(48,54,61,0.5)',
                    linecolor: '#30363d',
                    side: 'right',
                    tickfont: {size: 10},
                    tickformat: '.5f'
                },
                margin: {l: 10, r: 70, t: 10, b: 40},
                showlegend: false
            }, {responsive: true});

            // Activations heatmap
            Plotly.newPlot('activations-chart', [{
                type: 'heatmap',
                z: [new Array(32).fill(0), new Array(32).fill(0), new Array(32).fill(0), new Array(32).fill(0)],
                colorscale: [
                    [0, '#0d1117'],
                    [0.3, '#1f6feb'],
                    [0.6, '#a371f7'],
                    [1, '#f85149']
                ],
                showscale: false
            }], {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: {l: 5, r: 5, t: 5, b: 5},
                xaxis: {visible: false},
                yaxis: {visible: false}
            }, {responsive: true});
        }

        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8765/ws');

            ws.onopen = () => {
                document.getElementById('connection-status').className = 'status-dot connected';
                document.getElementById('connection-text').textContent = 'Connected';
            };

            ws.onclose = () => {
                document.getElementById('connection-status').className = 'status-dot disconnected';
                document.getElementById('connection-text').textContent = 'Disconnected';
                setTimeout(connectWebSocket, 2000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateVisualization(data);
            };
        }

        function updateVisualization(data) {
            // Update timestamp
            document.getElementById('timestamp').textContent = data.timestamp;

            // Update candlestick chart with markers and annotations
            if (data.candles) {
                let traces = [{
                    type: 'candlestick',
                    x: data.candles.timestamps,
                    open: data.candles.open,
                    high: data.candles.high,
                    low: data.candles.low,
                    close: data.candles.close,
                    increasing: {line: {color: '#3fb950', width: 1}, fillcolor: '#238636'},
                    decreasing: {line: {color: '#f85149', width: 1}, fillcolor: '#da3633'}
                }];

                // Entry markers - Direction-based colors and symbols
                // LONG = Green Up Triangle, SHORT = Red Down Triangle
                if (data.entries && data.entries.length > 0) {
                    traces.push({
                        type: 'scatter',
                        mode: 'markers+text',
                        x: data.entries.map(e => e.time),
                        y: data.entries.map(e => e.price),
                        marker: {
                            size: 20,
                            color: data.entries.map(e => e.direction === 1 ? '#3fb950' : '#f85149'),
                            symbol: data.entries.map(e => e.direction === 1 ? 'triangle-up' : 'triangle-down'),
                            line: {color: '#ffffff', width: 2}
                        },
                        text: data.entries.map(e => e.direction === 1 ? 'LONG' : 'SHORT'),
                        textposition: data.entries.map(e => e.direction === 1 ? 'top center' : 'bottom center'),
                        textfont: {
                            size: 10,
                            color: data.entries.map(e => e.direction === 1 ? '#3fb950' : '#f85149'),
                            family: 'SF Mono, monospace',
                            weight: 700
                        },
                        name: 'Entry',
                        hovertemplate: '<b>%{text}</b><br>%{x}<br>Price: %{y:.5f}<extra></extra>'
                    });
                }

                // Exit markers - Color based on P&L, symbol based on direction
                // Exit Long = triangle-down, Exit Short = triangle-up
                if (data.exits && data.exits.length > 0) {
                    traces.push({
                        type: 'scatter',
                        mode: 'markers+text',
                        x: data.exits.map(e => e.time),
                        y: data.exits.map(e => e.price),
                        marker: {
                            size: 20,
                            color: data.exits.map(e => e.pnl >= 0 ? '#3fb950' : '#f85149'),
                            symbol: data.exits.map(e => e.direction === 1 ? 'triangle-down' : 'triangle-up'),
                            line: {color: '#ffffff', width: 2}
                        },
                        text: data.exits.map(e => {
                            const sign = e.pnl >= 0 ? '+' : '';
                            const pips = e.pnl.toFixed(1);
                            const dollars = e.pnl_dollars ? (e.pnl_dollars >= 0 ? '+$' : '-$') + Math.abs(e.pnl_dollars).toFixed(0) : '';
                            return sign + pips + ' pips<br>' + dollars;
                        }),
                        textposition: data.exits.map(e => e.direction === 1 ? 'bottom center' : 'top center'),
                        textfont: {
                            size: 11,
                            color: data.exits.map(e => e.pnl >= 0 ? '#3fb950' : '#f85149'),
                            family: 'SF Mono, monospace',
                            weight: 700
                        },
                        name: 'Exit',
                        hovertemplate: '<b>EXIT</b><br>%{x}<br>Price: %{y:.5f}<br>%{text}<extra></extra>'
                    });
                }

                let layout = {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(13,17,23,0.5)',
                    font: {color: '#8b949e', family: 'SF Pro Display'},
                    xaxis: {
                        gridcolor: 'rgba(48,54,61,0.5)',
                        linecolor: '#30363d',
                        rangeslider: {visible: false},
                        tickfont: {size: 10}
                    },
                    yaxis: {
                        gridcolor: 'rgba(48,54,61,0.5)',
                        linecolor: '#30363d',
                        side: 'right',
                        tickfont: {size: 10},
                        tickformat: '.5f'
                    },
                    margin: {l: 10, r: 70, t: 30, b: 40},
                    showlegend: false,
                    shapes: [],
                    annotations: []
                };

                // Add SL/TP zone for active trade
                if (data.trade_zone) {
                    const tz = data.trade_zone;
                    // Profit/Loss zone fill
                    layout.shapes.push({
                        type: 'rect',
                        xref: 'x', yref: 'y',
                        x0: tz.x0, x1: tz.x1,
                        y0: tz.sl, y1: tz.tp,
                        fillcolor: 'rgba(88,166,255,0.08)',
                        line: {width: 0}
                    });
                    // Original SL line (dotted gray - initial position)
                    layout.shapes.push({
                        type: 'line',
                        xref: 'x', yref: 'y',
                        x0: tz.x0, x1: tz.x1,
                        y0: tz.original_sl, y1: tz.original_sl,
                        line: {color: '#6e7681', width: 1, dash: 'dot'}
                    });
                    // Current SL line (solid red - may be tightened by RL)
                    layout.shapes.push({
                        type: 'line',
                        xref: 'x', yref: 'y',
                        x0: tz.x0, x1: tz.x1,
                        y0: tz.sl, y1: tz.sl,
                        line: {color: '#f85149', width: 2, dash: tz.sl_changed ? 'solid' : 'dash'}
                    });
                    // TP line (green)
                    layout.shapes.push({
                        type: 'line',
                        xref: 'x', yref: 'y',
                        x0: tz.x0, x1: tz.x1,
                        y0: tz.tp, y1: tz.tp,
                        line: {color: '#3fb950', width: 2, dash: 'dash'}
                    });
                    // Entry price line (blue dotted)
                    layout.shapes.push({
                        type: 'line',
                        xref: 'x', yref: 'y',
                        x0: tz.x0, x1: tz.x1,
                        y0: tz.entry_price, y1: tz.entry_price,
                        line: {color: '#58a6ff', width: 1, dash: 'dot'}
                    });
                    // Add labels for SL/TP
                    layout.annotations.push({
                        x: tz.x1, y: tz.tp,
                        text: 'TP', showarrow: false,
                        font: {color: '#3fb950', size: 10}, xanchor: 'left'
                    });
                    layout.annotations.push({
                        x: tz.x1, y: tz.sl,
                        text: tz.sl_changed ? 'SL (modified)' : 'SL',
                        showarrow: false,
                        font: {color: '#f85149', size: 10}, xanchor: 'left'
                    });
                }

                // Add PnL annotations for exits
                if (data.exits && data.exits.length > 0) {
                    data.exits.forEach(e => {
                        layout.annotations.push({
                            x: e.time,
                            y: e.price,
                            xref: 'x',
                            yref: 'y',
                            text: '<b>' + (e.pnl >= 0 ? '+' : '') + e.pnl.toFixed(1) + ' pips</b><br>' +
                                  (e.pnl_dollars ? (e.pnl_dollars >= 0 ? '+$' : '-$') + Math.abs(e.pnl_dollars).toFixed(0) : ''),
                            showarrow: true,
                            arrowhead: 0,
                            arrowsize: 1,
                            arrowwidth: 1,
                            arrowcolor: e.pnl >= 0 ? '#3fb950' : '#f85149',
                            ax: 0,
                            ay: e.pnl >= 0 ? -50 : 50,
                            bgcolor: 'rgba(13,17,23,0.9)',
                            bordercolor: e.pnl >= 0 ? '#3fb950' : '#f85149',
                            borderwidth: 2,
                            borderpad: 4,
                            font: {
                                color: e.pnl >= 0 ? '#3fb950' : '#f85149',
                                size: 11,
                                family: 'SF Mono, monospace'
                            }
                        });
                    });
                }

                Plotly.react('candlestick-chart', traces, layout);
            }

            // Update current trade panel
            const tradeContainer = document.getElementById('current-trade-container');
            if (data.current_trade) {
                const t = data.current_trade;
                const dirClass = t.direction === 1 ? 'long' : 'short';
                const dirText = t.direction === 1 ? 'LONG' : 'SHORT';
                const pnlColor = t.pnl_pips >= 0 ? '#3fb950' : '#f85149';

                // Build action counts summary
                const ac = t.action_counts || {};
                const actionParts = [];
                if (ac.HOLD > 0) actionParts.push(`H:${ac.HOLD}`);
                if (ac.TIGHTEN_SL > 0) actionParts.push(`<span style="color:#d29922">T:${ac.TIGHTEN_SL}</span>`);
                if (ac.TRAIL_BE > 0) actionParts.push(`<span style="color:#58a6ff">BE:${ac.TRAIL_BE}</span>`);
                if (ac.EXIT > 0) actionParts.push(`<span style="color:#f85149">X:${ac.EXIT}</span>`);
                if (ac.PARTIAL > 0) actionParts.push(`<span style="color:#a371f7">P:${ac.PARTIAL}</span>`);
                const actionSummary = actionParts.join(' ') || 'None';

                // SL status
                const slTightened = t.sl_tightened ? '<span style="color:#d29922;">[SL TIGHTENED]</span>' : '';
                const trailedBE = t.trailed_to_be ? '<span style="color:#58a6ff;">[TRAILING BE]</span>' : '';

                // Position size display
                const posSize = t.position_size !== undefined ? t.position_size : 1.0;
                const posSizePercent = (posSize * 100).toFixed(0);
                const posValueM = t.position_value_m !== undefined ? t.position_value_m.toFixed(1) : '2.0';
                const posColor = posSize < 1.0 ? '#a371f7' : '#3fb950';  // Purple if partial taken
                const partialInfo = t.partial_exits > 0 ? `<span style="color:#a371f7;">(${t.partial_exits} partial${t.partial_exits > 1 ? 's' : ''})</span>` : '';

                // Show realized vs unrealized
                const realizedPnl = t.realized_pnl !== undefined ? t.realized_pnl : 0;
                const unrealizedPnl = t.unrealized_pnl !== undefined ? t.unrealized_pnl : t.pnl_dollars;

                tradeContainer.innerHTML = `
                    <div class="current-trade">
                        <div class="current-trade-header">
                            <span class="current-trade-direction ${dirClass}">${dirText}</span>
                            <span style="color:${posColor}; font-size:12px; font-weight:600;">${posValueM}M (${posSizePercent}%)</span>
                        </div>
                        <div class="current-trade-pnl" style="color:${pnlColor}">
                            ${t.pnl_pips >= 0 ? '+' : ''}${t.pnl_pips.toFixed(1)} pips
                        </div>
                        <div style="color:${pnlColor}; font-size:16px; font-weight:600; font-family:'SF Mono',monospace;">
                            ${t.pnl_dollars >= 0 ? '+' : ''}$${t.pnl_dollars.toFixed(0)} ${partialInfo}
                        </div>
                        ${realizedPnl !== 0 ? `
                        <div style="display:flex; justify-content:space-between; font-size:10px; margin-top:4px;">
                            <span style="color:#a371f7;">Realized: $${realizedPnl.toFixed(0)}</span>
                            <span style="color:#8b949e;">Unrealized: $${unrealizedPnl.toFixed(0)}</span>
                        </div>` : ''}
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px;">
                            <div class="feature">
                                <div class="feature-name">Entry</div>
                                <div class="feature-value neutral">${t.entry_price.toFixed(5)}</div>
                            </div>
                            <div class="feature">
                                <div class="feature-name">Current</div>
                                <div class="feature-value neutral">${t.current_price.toFixed(5)}</div>
                            </div>
                            <div class="feature">
                                <div class="feature-name">MFE</div>
                                <div class="feature-value positive">+${t.mfe.toFixed(1)} pips</div>
                            </div>
                            <div class="feature">
                                <div class="feature-name">MAE</div>
                                <div class="feature-value negative">${t.mae.toFixed(1)} pips</div>
                            </div>
                        </div>
                        <div style="margin-top:10px; padding-top:8px; border-top:1px solid #30363d;">
                            <div style="display:flex; justify-content:space-between; font-size:11px; color:#8b949e;">
                                <span>SL: ${t.current_sl ? t.current_sl.toFixed(5) : 'N/A'}</span>
                                <span>${slTightened}${trailedBE}</span>
                            </div>
                            <div style="margin-top:6px; font-size:11px;">
                                <span style="color:#8b949e;">Actions:</span> ${actionSummary}
                            </div>
                        </div>
                    </div>
                `;
            } else {
                tradeContainer.innerHTML = '<div class="no-trade">No active trade</div>';
            }

            // Update action display
            if (data.action !== undefined) {
                const actionName = actionNames[data.action];
                const actionEl = document.getElementById('action-name');
                actionEl.textContent = actionName;
                actionEl.className = 'action-name action-' + actionName;

                // Update probability bars
                for (let i = 0; i < 5; i++) {
                    const prob = data.probs[i] * 100;
                    document.getElementById('prob-' + i).style.width = prob + '%';
                    document.getElementById('prob-val-' + i).textContent = prob.toFixed(1) + '%';
                }

                // Update model outputs
                if (data.model_output) {
                    document.getElementById('model-value').textContent = data.model_output.value.toFixed(3);
                    document.getElementById('model-entropy').textContent = data.model_output.entropy.toFixed(3);
                    document.getElementById('model-confidence').textContent = (data.model_output.confidence * 100).toFixed(0) + '%';
                }
            }

            // Update neural activations
            if (data.activations) {
                Plotly.react('activations-chart', [{
                    type: 'heatmap',
                    z: data.activations,
                    colorscale: [
                        [0, '#0d1117'],
                        [0.3, '#1f6feb'],
                        [0.6, '#a371f7'],
                        [1, '#f85149']
                    ],
                    showscale: false
                }], {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    margin: {l: 5, r: 5, t: 5, b: 5},
                    xaxis: {visible: false},
                    yaxis: {visible: false}
                });
            }

            // Update comprehensive metrics panel
            if (data.metrics) {
                const m = data.metrics;
                const s = data.stats;

                // Key metrics
                document.getElementById('metric-sharpe').textContent = m.sharpe_ratio.toFixed(2);
                document.getElementById('metric-winrate').textContent = m.win_rate + '%';

                const pnlEl = document.getElementById('metric-pnl');
                pnlEl.textContent = (s.total_pnl >= 0 ? '+$' : '-$') + Math.abs(s.total_pnl).toFixed(0);
                pnlEl.className = 'metric-value ' + (s.total_pnl >= 0 ? 'positive' : 'negative');

                const retEl = document.getElementById('metric-return');
                retEl.textContent = (m.return_pct >= 0 ? '+' : '') + m.return_pct.toFixed(3) + '%';
                retEl.className = 'metric-value ' + (m.return_pct >= 0 ? 'positive' : 'negative');

                // Trade stats
                document.getElementById('metric-trades').textContent = s.trades;
                document.getElementById('metric-pf').textContent = m.profit_factor.toFixed(2);
                document.getElementById('metric-avgwin').textContent = '+$' + m.avg_win.toFixed(0);
                document.getElementById('metric-avgloss').textContent = '-$' + m.avg_loss.toFixed(0);

                // Long/Short
                document.getElementById('metric-longpct').textContent = m.long_pct + '%';
                document.getElementById('metric-shortpct').textContent = m.short_pct + '%';
                document.getElementById('metric-longwr').textContent = m.long_win_rate + '%';
                document.getElementById('metric-shortwr').textContent = m.short_win_rate + '%';

                // Duration & Drawdown
                document.getElementById('metric-avgdur').textContent = m.avg_duration.toFixed(1);
                document.getElementById('metric-maxdd').textContent = '-$' + m.max_drawdown.toFixed(0);
                document.getElementById('metric-maxwin').textContent = '+$' + m.max_win.toFixed(0);
                document.getElementById('metric-maxloss').textContent = '-$' + Math.abs(m.max_loss).toFixed(0);

                // Exit types
                document.getElementById('metric-rlexit').textContent = m.rl_exit_pct + '%';
                document.getElementById('metric-slhit').textContent = m.sl_hit_pct + '%';
                document.getElementById('metric-partials').textContent = m.partial_per_trade.toFixed(2);

                const pipsEl = document.getElementById('metric-pips');
                pipsEl.textContent = (s.total_pips >= 0 ? '+' : '') + s.total_pips.toFixed(1);
                pipsEl.className = 'metric-value ' + (s.total_pips >= 0 ? 'positive' : 'negative');
            }

            // === UPDATE DETAILED TRADE LOG (Card-Based Layout) ===
                const logBody = document.getElementById('trade-log-body');
                const PIP_VALUE = 200;  // $200 per pip for 2M position

                logBody.innerHTML = data.trade_history.map((t, idx) => {
                    const isActive = t.is_active === true;
                    const dirClass = t.direction === 1 ? 'long' : 'short';
                    const dirText = t.direction === 1 ? '🟢 LONG' : '🔴 SHORT';
                    const pnlClass = t.pnl_dollars >= 0 ? 'positive' : 'negative';

                    // Entry info
                    const entryTime = t.entry_time_pretty || t.entry_time || '-';
                    const entryPrice = t.entry_price ? t.entry_price.toFixed(5) : '-';
                    const positionM = isActive
                        ? (t.position_size_pct / 100 * 2).toFixed(1)
                        : '2.0';

                    // SL/TP
                    const slPrice = t.sl_price ? t.sl_price.toFixed(5) : (t.original_sl ? t.original_sl.toFixed(5) : '-');
                    const tpPrice = t.tp_price ? t.tp_price.toFixed(5) : '-';
                    const slChanged = t.sl_tightened || t.trailed_to_be;

                    // Result
                    const pnlSign = t.pnl_dollars >= 0 ? '+' : '';
                    const pipsSign = t.pnl_pips >= 0 ? '+' : '';

                    // Build action timeline
                    const actions = t.action_history || [];
                    let actionsHtml = '';
                    if (actions.length === 0) {
                        actionsHtml = '<div class="no-actions">No significant actions yet...</div>';
                    } else {
                        // Calculate running totals for meaningful actions (PARTIAL, EXIT)
                        let realizedPnl = 0;
                        let totalPips = 0;

                        actionsHtml = actions.map(a => {
                            const actionType = a.action.toLowerCase().replace('_', '');
                            const actionClass = actionType === 'tighten_sl' ? 'tighten' :
                                               actionType === 'trail_be' ? 'trail' : actionType;
                            const pnlSign = a.pnl_dollars >= 0 ? '+' : '';
                            const pnlClass = a.pnl_dollars >= 0 ? 'positive' : 'negative';

                            // Track cumulative P&L for PARTIAL and EXIT actions
                            if (a.action === 'PARTIAL' || a.action === 'EXIT') {
                                realizedPnl += a.pnl_dollars;
                                totalPips += a.pnl_pips * (a.position_size_pct / 100);
                            }

                            let actionContent = '';
                            if (a.action === 'PARTIAL') {
                                actionContent = `
                                    <div class="action-time">${a.time_pretty || ''}</div>
                                    <div class="action-price">@ ${a.price.toFixed(5)}</div>
                                    <div class="action-size">${a.position_size_pct.toFixed(0)}% (${a.position_m.toFixed(1)}M)</div>
                                    <div class="action-pnl ${pnlClass}">${pnlSign}${a.pnl_pips.toFixed(1)} pips</div>
                                    <div class="action-pnl ${pnlClass}">${pnlSign}$${a.pnl_dollars.toFixed(0)}</div>
                                `;
                            } else if (a.action === 'EXIT') {
                                actionContent = `
                                    <div class="action-time">${a.time_pretty || ''}</div>
                                    <div class="action-price">@ ${a.price.toFixed(5)}</div>
                                    <div class="action-size">${a.position_size_pct.toFixed(0)}% (${a.position_m.toFixed(1)}M)</div>
                                    <div class="action-pnl ${pnlClass}">${pnlSign}${a.pnl_pips.toFixed(1)} pips</div>
                                    <div class="action-pnl ${pnlClass}">${pnlSign}$${a.pnl_dollars.toFixed(0)}</div>
                                `;
                            } else if (a.action === 'TIGHTEN_SL' || a.action === 'TRAIL_BE') {
                                actionContent = `
                                    <div class="action-time">${a.time_pretty || ''}</div>
                                    <div class="action-price">@ ${a.price.toFixed(5)}</div>
                                    <div class="action-note">${a.note || ''}</div>
                                    <div class="action-pnl ${pnlClass}">PnL: ${pnlSign}$${a.pnl_dollars.toFixed(0)}</div>
                                `;
                            } else {
                                // HOLD
                                actionContent = `
                                    <div class="action-time">${a.time_pretty || ''}</div>
                                    <div class="action-price">@ ${a.price.toFixed(5)}</div>
                                    <div class="action-pnl ${pnlClass}">PnL: ${pnlSign}$${a.pnl_dollars.toFixed(0)}</div>
                                `;
                            }

                            return `
                                <div class="action-item ${actionClass}">
                                    <div class="action-header">
                                        <span class="action-type ${actionClass}">${a.action}</span>
                                        <span class="action-bar">Bar ${a.bar}</span>
                                    </div>
                                    ${actionContent}
                                </div>
                            `;
                        }).join('');

                        // Add TOTAL summary at the end for completed trades
                        if (!isActive && t.pnl_dollars !== undefined) {
                            const totalPnl = t.pnl_dollars;
                            const totalClass = totalPnl >= 0 ? 'positive' : 'negative';
                            const totalSign = totalPnl >= 0 ? '+' : '';
                            const totalPipsVal = t.pnl_pips || 0;
                            const totalPipsSign = totalPipsVal >= 0 ? '+' : '';

                            actionsHtml += `
                                <div class="action-item" style="border-left: 3px solid ${totalPnl >= 0 ? '#3fb950' : '#f85149'}; background: ${totalPnl >= 0 ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)'}; min-width: 140px;">
                                    <div class="action-header">
                                        <span class="action-type" style="color: ${totalPnl >= 0 ? '#3fb950' : '#f85149'}; font-size: 10px;">TOTAL</span>
                                        <span class="action-bar" style="background: ${totalPnl >= 0 ? 'rgba(63, 185, 80, 0.3)' : 'rgba(248, 81, 73, 0.3)'}">${t.exit_reason || 'Closed'}</span>
                                    </div>
                                    <div style="font-size: 10px; color: #8b949e; margin-bottom: 2px;">Full Trade Result</div>
                                    <div class="action-pnl ${totalClass}" style="font-size: 13px;">${totalPipsSign}${totalPipsVal.toFixed(1)} pips</div>
                                    <div class="action-pnl ${totalClass}" style="font-size: 15px; font-weight: 800;">${totalSign}$${Math.abs(totalPnl).toFixed(0)}</div>
                                </div>
                            `;
                        }
                    }

                    // Build the card
                    return `
                        <div class="trade-card ${isActive ? 'active' : ''}">
                            <div class="trade-card-header">
                                <div class="trade-entry-info">
                                    <span class="trade-direction ${dirClass}">${dirText}</span>
                                    <div class="trade-details">
                                        <span class="label">Entry:</span>
                                        <span class="value">${positionM}M @ ${entryPrice}</span>
                                        <span style="color:#6e7681;margin-left:8px;">${entryTime}</span>
                                    </div>
                                    <div class="trade-levels">
                                        <span class="sl" style="${slChanged ? 'font-weight:600;' : ''}">SL: ${slPrice}${slChanged ? ' ✓' : ''}</span>
                                        <span class="tp">TP: ${tpPrice}</span>
                                    </div>
                                </div>
                                <div class="trade-result">
                                    ${isActive ? '<span class="active-badge">⚡ ACTIVE</span>' : ''}
                                    <div class="pnl ${pnlClass}">${pnlSign}$${Math.abs(t.pnl_dollars).toFixed(0)}</div>
                                    <div class="pips">${pipsSign}${t.pnl_pips.toFixed(1)} pips</div>
                                </div>
                            </div>
                            <div class="action-timeline">
                                ${actionsHtml}
                            </div>
                        </div>
                    `;
                }).join('');
            }
        }

        function togglePlay() {
            isPlaying = !isPlaying;
            const btn = document.getElementById('play-btn');
            btn.innerHTML = isPlaying ? '<span>&#10074;&#10074;</span> Pause' : '<span>&#9658;</span> Play';
            ws.send(JSON.stringify({command: isPlaying ? 'play' : 'pause'}));
        }

        function stepForward() {
            ws.send(JSON.stringify({command: 'step'}));
        }

        function nextTrade() {
            // Jump to next trade entry or current trade's exit
            ws.send(JSON.stringify({command: 'next_trade'}));
        }

        function resetPlayback() {
            ws.send(JSON.stringify({command: 'reset'}));
            isPlaying = false;
            document.getElementById('play-btn').innerHTML = '<span>&#9658;</span> Play';
        }

        // Speed slider
        document.getElementById('speed-slider').addEventListener('input', (e) => {
            const speed = e.target.value;
            document.getElementById('speed-value').textContent = speed + 'x';
            ws.send(JSON.stringify({command: 'speed', value: parseInt(speed)}));
        });

        // Initialize on load
        window.onload = () => {
            initCharts();
            connectWebSocket();
        };
    </script>
</body>
</html>
"""


# ============================================================================
# Model with Activation Hooks
# ============================================================================

class ActorCriticWithActivations(ActorCritic):
    """ActorCritic with hooks to capture intermediate activations."""

    def __init__(self, config: PPOConfig):
        super().__init__(config)
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        # Hook into encoder layers
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(get_activation(f'encoder_{i}'))

        # Hook into actor/critic heads
        self.actor[0].register_forward_hook(get_activation('actor_hidden'))
        self.critic[0].register_forward_hook(get_activation('critic_hidden'))

    def get_action_with_details(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Dict:
        """Get action with full details for visualization."""
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
            activations_2d = act[:rows*cols].reshape(rows, cols).tolist()

        return {
            'action': action.item(),
            'probs': probs.cpu().numpy()[0].tolist(),
            'value': value.item(),
            'entropy': entropy.item(),
            'confidence': confidence.item(),
            'activations': activations_2d
        }


# ============================================================================
# Trade Simulator
# ============================================================================

@dataclass
class Trade:
    """Represents an active or completed trade."""
    entry_time: str
    entry_price: float
    entry_idx: int
    direction: int  # 1 = long, -1 = short
    sl: float
    tp: float
    entry_atr: float
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_idx: Optional[int] = None
    pnl_pips: Optional[float] = None
    pnl_dollars: Optional[float] = None
    max_favorable_pips: float = 0.0
    max_adverse_pips: float = 0.0
    # Position size tracking (1.0 = full, 0.5 = half after partial)
    position_size: float = 1.0  # Fraction of original position remaining
    realized_pnl_pips: float = 0.0  # PnL already realized from partials
    realized_pnl_dollars: float = 0.0
    partial_exits: List = field(default_factory=list)  # List of {price, pips, dollars, size}
    # Action tracking
    action_counts: Dict = field(default_factory=lambda: {
        'HOLD': 0, 'EXIT': 0, 'TIGHTEN_SL': 0, 'TRAIL_BE': 0, 'PARTIAL': 0
    })
    # Complete action history - each action with full details
    action_history: List = field(default_factory=list)  # List of action dicts
    exit_reason: str = ""  # RL_EXIT, SL_HIT, TRAIL_BE_HIT, TIGHTEN_SL_HIT, TP_HIT
    sl_tightened: bool = False
    trailed_to_be: bool = False


class TradeSimulator:
    """Simulates trading on historical data with RL exit decisions."""

    def __init__(
        self,
        price_data: pd.DataFrame,
        trades_data: pd.DataFrame,
        model: ActorCriticWithActivations,
        device: str = "mps"
    ):
        self.price_data = price_data.copy()
        self.trades_data = trades_data
        self.model = model
        self.device = torch.device(device)

        # Create local index mapping
        self.price_data['local_idx'] = range(len(price_data))

        # Create timestamp to local index mapping
        self.ts_to_idx = {str(ts): idx for idx, ts in enumerate(price_data.index)}

        # Pre-compute trade entries by LOCAL index (using entry_time)
        self.trade_entries = {}
        for _, trade in trades_data.iterrows():
            entry_time = str(trade['entry_time'])
            if entry_time in self.ts_to_idx:
                local_idx = self.ts_to_idx[entry_time]
                self.trade_entries[local_idx] = trade

        # Current state
        self.current_idx = 0
        self.active_trade: Optional[Trade] = None

        # Track completed trades
        self.completed_trades: List[Trade] = []
        self.entries: List[Dict] = []
        self.exits: List[Dict] = []

        # State tracking for RL
        self.bars_held = 0
        self.max_favorable = 0.0
        self.max_adverse = 0.0
        self.action_history = [0] * 5
        self.current_sl_price = 0.0  # Current SL level (from classical, can be tightened by RL)

        # Session stats - comprehensive metrics
        self.stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pips': 0.0,
            'total_pnl': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'long_trades': 0,
            'short_trades': 0,
            'long_wins': 0,
            'short_wins': 0,
            'long_pnl': 0.0,
            'short_pnl': 0.0,
            'total_bars_held': 0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'pnl_history': [],  # For Sharpe calculation
            'partial_count': 0,
            'sl_count': 0,
            'trail_be_count': 0,
            'tighten_sl_count': 0,
            'rl_exit_count': 0
        }

        # Add technical indicators
        self._add_indicators()

    def _add_indicators(self):
        """Add technical indicators for state computation."""
        df = self.price_data

        # ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ADX (simplified - using directional movement)
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()

        # BB position
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['bb_upper'] = sma + 2 * std
        df['bb_lower'] = sma - 2 * std
        df['bb_pos'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # EMA diff
        df['ema_fast'] = df['Close'].ewm(span=30).mean()
        df['ema_slow'] = df['Close'].ewm(span=50).mean()
        df['ema_diff'] = (df['ema_fast'] - df['ema_slow']) / df['Close']

        # Fill NaN
        df.bfill(inplace=True)
        df.fillna(0, inplace=True)

    def reset(self):
        """Reset simulator to beginning."""
        self.current_idx = 0
        self.active_trade = None
        self.completed_trades = []
        self.entries = []
        self.exits = []
        self.bars_held = 0
        self.max_favorable = 0.0
        self.max_adverse = 0.0
        self.action_history = [0] * 5
        self.current_sl_price = 0.0  # Will be set from classical strategy's SL
        self.stats = {'trades': 0, 'wins': 0, 'total_pips': 0.0, 'total_pnl': 0.0}

    def build_state(self) -> torch.Tensor:
        """Build state tensor for RL model."""
        row = self.price_data.iloc[self.current_idx]

        if self.active_trade is None:
            return torch.zeros(1, 25, device=self.device)

        # Compute unrealized PnL
        current_price = row['Close']
        entry_price = self.active_trade.entry_price
        direction = self.active_trade.direction
        pnl = (current_price - entry_price) / entry_price * direction

        # Update MFE/MAE
        self.max_favorable = max(self.max_favorable, pnl)
        self.max_adverse = min(self.max_adverse, pnl)

        # Position features (5)
        bars_held_norm = self.bars_held / 200.0
        # Compute SL distance as fraction of entry price
        sl_distance = abs(entry_price - self.current_sl_price) / entry_price if entry_price > 0 else 0

        # Market features (10)
        market_features = [
            pnl,
            self.max_favorable,
            self.max_adverse,
            row['atr'] / current_price if current_price > 0 else 0,
            row['adx'] / 100.0 if not np.isnan(row['adx']) else 0.25,
            row['rsi'] / 100.0 if not np.isnan(row['rsi']) else 0.5,
            row['bb_pos'] if not np.isnan(row['bb_pos']) else 0.5,
            row['ema_diff'] if not np.isnan(row['ema_diff']) else 0,
            (row['Close'] - row['Open']) / row['Open'] if row['Open'] > 0 else 0,
            0.0  # volume placeholder
        ]

        # Entry context (5)
        entry_context = [
            self.active_trade.entry_atr / entry_price if entry_price > 0 else 0,
            row['adx'] / 100.0 if not np.isnan(row['adx']) else 0.25,
            row['rsi'] / 100.0 if not np.isnan(row['rsi']) else 0.5,
            (row['bb_upper'] - row['bb_lower']) / current_price if current_price > 0 else 0,
            row['ema_diff'] if not np.isnan(row['ema_diff']) else 0
        ]

        # Action history (5)
        action_hist = [a / 4.0 for a in self.action_history]

        # Combine all features
        state = [bars_held_norm, pnl, self.max_favorable, self.max_adverse, sl_distance]
        state.extend(market_features)
        state.extend(entry_context)
        state.extend(action_hist)

        # Clean any NaN/Inf
        state = [0.0 if (np.isnan(x) or np.isinf(x)) else x for x in state]

        return torch.tensor([state], dtype=torch.float32, device=self.device)

    def step_to_next_trade(self) -> Optional[Dict]:
        """
        Jump to next significant event:
        - If in a trade: step until exit
        - If not in trade: step until next entry
        Returns the final state after the jump.
        """
        result = None
        max_steps = 5000  # Safety limit
        steps = 0

        if self.active_trade is not None:
            # In a trade - step until it exits
            while self.active_trade is not None and steps < max_steps:
                result = self.step()
                if result is None:
                    break
                steps += 1
        else:
            # Not in trade - step until we enter one
            while self.active_trade is None and steps < max_steps:
                result = self.step()
                if result is None:
                    break
                steps += 1
                # Check if we just entered a trade
                if self.active_trade is not None:
                    break

        return result

    def step(self) -> Optional[Dict]:
        """Advance one bar and return visualization data."""
        if self.current_idx >= len(self.price_data) - 1:
            return None

        row = self.price_data.iloc[self.current_idx]
        timestamp = str(row.name)

        action = 0
        probs = [1.0, 0, 0, 0, 0]
        model_output = {'value': 0, 'entropy': 0, 'confidence': 1.0}
        activations = None

        # Check for new trade entry
        if self.active_trade is None and self.current_idx in self.trade_entries:
            trade_row = self.trade_entries[self.current_idx]
            entry_atr = row['atr'] if not np.isnan(row['atr']) else 0.001
            self.active_trade = Trade(
                entry_time=timestamp,
                entry_price=trade_row['entry_price'],
                entry_idx=self.current_idx,
                direction=int(trade_row['direction']),
                sl=trade_row['sl'],
                tp=trade_row['tp'],
                entry_atr=entry_atr
            )
            self.entries.append({
                'time': timestamp,
                'price': float(trade_row['entry_price']),
                'direction': int(trade_row['direction'])  # 1=Long, -1=Short
            })
            self.bars_held = 0
            self.max_favorable = 0.0
            self.max_adverse = 0.0
            # Use the CLASSICAL strategy's SL as initial stop loss
            self.current_sl_price = trade_row['sl']

        # If in trade, get RL decision
        current_trade_info = None
        if self.active_trade is not None:
            state_tensor = self.build_state()
            result = self.model.get_action_with_details(state_tensor, deterministic=True)

            action = result['action']
            probs = result['probs']
            model_output = {
                'value': result['value'],
                'entropy': result['entropy'],
                'confidence': result['confidence']
            }
            activations = result['activations']

            # Update action history
            self.action_history = self.action_history[1:] + [action]

            # Track action counts for this trade
            action_names = ['HOLD', 'EXIT', 'TIGHTEN_SL', 'TRAIL_BE', 'PARTIAL']
            self.active_trade.action_counts[action_names[action]] += 1

            # Current trade metrics
            current_price = row['Close']
            pnl_pips = (current_price - self.active_trade.entry_price) * 10000 * self.active_trade.direction
            pnl_dollars = pnl_pips * POSITION_SIZE / 10000

            # Update MFE/MAE in pips
            self.active_trade.max_favorable_pips = max(self.active_trade.max_favorable_pips, pnl_pips)
            self.active_trade.max_adverse_pips = min(self.active_trade.max_adverse_pips, pnl_pips)

            # Calculate unrealized PnL for remaining position
            unrealized_pnl_dollars = pnl_dollars * self.active_trade.position_size
            # Total PnL = realized (from partials) + unrealized (remaining position)
            total_pnl_dollars = self.active_trade.realized_pnl_dollars + unrealized_pnl_dollars

            current_trade_info = {
                'direction': int(self.active_trade.direction),
                'entry_price': float(self.active_trade.entry_price),
                'current_price': float(current_price),
                'pnl_pips': float(pnl_pips),
                'pnl_dollars': float(total_pnl_dollars),
                'unrealized_pnl': float(unrealized_pnl_dollars),
                'realized_pnl': float(self.active_trade.realized_pnl_dollars),
                'bars_held': int(self.bars_held),
                'mfe': float(self.active_trade.max_favorable_pips),
                'mae': float(self.active_trade.max_adverse_pips),
                'action_counts': {k: int(v) for k, v in self.active_trade.action_counts.items()},
                'sl_tightened': bool(self.active_trade.sl_tightened),
                'trailed_to_be': bool(self.active_trade.trailed_to_be),
                'current_sl': float(self.current_sl_price),
                'original_sl': float(self.active_trade.sl),
                'position_size': float(self.active_trade.position_size),
                'position_value_m': float(POSITION_SIZE * self.active_trade.position_size / 1_000_000),
                'partial_exits': int(len(self.active_trade.partial_exits))
            }

            # Process action and record to action_history
            exit_trade = False
            exit_price = current_price
            exit_reason = ""
            old_sl = self.current_sl_price

            if action == Actions.EXIT:
                exit_trade = True
                exit_reason = "RL_EXIT"
                # Record EXIT action
                self.active_trade.action_history.append({
                    'action': 'EXIT',
                    'bar': self.bars_held,
                    'time': timestamp,
                    'time_pretty': format_datetime_pretty(timestamp),
                    'price': float(current_price),
                    'pnl_pips': float(pnl_pips),
                    'pnl_dollars': float(pnl_dollars * self.active_trade.position_size),
                    'position_size_pct': float(self.active_trade.position_size * 100),
                    'position_m': float(POSITION_SIZE * self.active_trade.position_size / 1_000_000),
                    'cumulative_pnl': float(total_pnl_dollars),
                    'sl': float(self.current_sl_price),
                    'note': f"Full exit @ {current_price:.5f}"
                })
            elif action == Actions.PARTIAL_EXIT:
                # Close 50% of remaining position, keep the rest
                if self.active_trade.position_size > 0.25:  # Don't partial if too small
                    partial_size = self.active_trade.position_size * 0.5
                    partial_pnl_pips = pnl_pips  # PnL per pip at this moment
                    partial_pnl_dollars = partial_pnl_pips * POSITION_SIZE * partial_size / 10000

                    # Record PARTIAL action BEFORE updating position
                    self.active_trade.action_history.append({
                        'action': 'PARTIAL',
                        'bar': self.bars_held,
                        'time': timestamp,
                        'time_pretty': format_datetime_pretty(timestamp),
                        'price': float(current_price),
                        'pnl_pips': float(partial_pnl_pips),
                        'pnl_dollars': float(partial_pnl_dollars),
                        'position_size_pct': float(partial_size * 100),
                        'position_m': float(POSITION_SIZE * partial_size / 1_000_000),
                        'remaining_pct': float((self.active_trade.position_size - partial_size) * 100),
                        'cumulative_pnl': float(self.active_trade.realized_pnl_dollars + partial_pnl_dollars),
                        'sl': float(self.current_sl_price),
                        'note': f"Partial {partial_size*100:.0f}% @ {current_price:.5f}"
                    })

                    # Record the partial exit
                    self.active_trade.partial_exits.append({
                        'price': current_price,
                        'pips': partial_pnl_pips,
                        'dollars': partial_pnl_dollars,
                        'size': partial_size,
                        'time': timestamp
                    })

                    # Accumulate realized PnL
                    self.active_trade.realized_pnl_pips += partial_pnl_pips * partial_size
                    self.active_trade.realized_pnl_dollars += partial_pnl_dollars

                    # Reduce position size
                    self.active_trade.position_size -= partial_size

                    # Add to stats (partial profit taking)
                    self.stats['total_pips'] += partial_pnl_pips * partial_size
                    self.stats['total_pnl'] += partial_pnl_dollars
                else:
                    # Position too small, treat as full exit
                    exit_trade = True
                    exit_reason = "RL_PARTIAL_FINAL"
                    self.active_trade.action_history.append({
                        'action': 'EXIT',
                        'bar': self.bars_held,
                        'time': timestamp,
                        'time_pretty': format_datetime_pretty(timestamp),
                        'price': float(current_price),
                        'pnl_pips': float(pnl_pips),
                        'pnl_dollars': float(pnl_dollars * self.active_trade.position_size),
                        'position_size_pct': float(self.active_trade.position_size * 100),
                        'position_m': float(POSITION_SIZE * self.active_trade.position_size / 1_000_000),
                        'cumulative_pnl': float(total_pnl_dollars),
                        'sl': float(self.current_sl_price),
                        'note': f"Final exit (too small to partial) @ {current_price:.5f}"
                    })
            elif action == Actions.TIGHTEN_SL:
                # Tighten the SL closer to current price
                if self.active_trade.direction == 1:  # Long
                    new_sl = current_price - (current_price - self.current_sl_price) * 0.5
                    if new_sl > self.current_sl_price:  # Only tighten
                        self.current_sl_price = new_sl
                        self.active_trade.sl_tightened = True
                else:  # Short
                    new_sl = current_price + (self.current_sl_price - current_price) * 0.5
                    if new_sl < self.current_sl_price:  # Only tighten
                        self.current_sl_price = new_sl
                        self.active_trade.sl_tightened = True
                # Record TIGHTEN_SL action
                self.active_trade.action_history.append({
                    'action': 'TIGHTEN_SL',
                    'bar': self.bars_held,
                    'time': timestamp,
                    'time_pretty': format_datetime_pretty(timestamp),
                    'price': float(current_price),
                    'pnl_pips': float(pnl_pips),
                    'pnl_dollars': float(unrealized_pnl_dollars),
                    'position_size_pct': float(self.active_trade.position_size * 100),
                    'position_m': float(POSITION_SIZE * self.active_trade.position_size / 1_000_000),
                    'cumulative_pnl': float(total_pnl_dollars),
                    'old_sl': float(old_sl),
                    'new_sl': float(self.current_sl_price),
                    'sl': float(self.current_sl_price),
                    'note': f"SL {old_sl:.5f} → {self.current_sl_price:.5f}"
                })
            elif action == Actions.TRAIL_BREAKEVEN:
                if pnl_pips > 0:
                    # Move SL to entry (breakeven)
                    self.current_sl_price = self.active_trade.entry_price
                    self.active_trade.trailed_to_be = True
                    # Record TRAIL_BE action
                    self.active_trade.action_history.append({
                        'action': 'TRAIL_BE',
                        'bar': self.bars_held,
                        'time': timestamp,
                        'time_pretty': format_datetime_pretty(timestamp),
                        'price': float(current_price),
                        'pnl_pips': float(pnl_pips),
                        'pnl_dollars': float(unrealized_pnl_dollars),
                        'position_size_pct': float(self.active_trade.position_size * 100),
                        'position_m': float(POSITION_SIZE * self.active_trade.position_size / 1_000_000),
                        'cumulative_pnl': float(total_pnl_dollars),
                        'old_sl': float(old_sl),
                        'new_sl': float(self.active_trade.entry_price),
                        'sl': float(self.current_sl_price),
                        'note': f"Trail to BE @ {self.active_trade.entry_price:.5f}"
                    })
            else:
                # HOLD action - only record significant ones (every 5 bars or first)
                if self.bars_held == 0 or self.bars_held % 5 == 0:
                    self.active_trade.action_history.append({
                        'action': 'HOLD',
                        'bar': self.bars_held,
                        'time': timestamp,
                        'time_pretty': format_datetime_pretty(timestamp),
                        'price': float(current_price),
                        'pnl_pips': float(pnl_pips),
                        'pnl_dollars': float(unrealized_pnl_dollars),
                        'position_size_pct': float(self.active_trade.position_size * 100),
                        'position_m': float(POSITION_SIZE * self.active_trade.position_size / 1_000_000),
                        'cumulative_pnl': float(total_pnl_dollars),
                        'sl': float(self.current_sl_price),
                        'note': f"Hold @ bar {self.bars_held}"
                    })

            # Check SL hit - ONLY after the entry bar (bars_held > 0)
            # Use the classical strategy's SL as initial, tracked by current_sl_price
            if self.bars_held > 0 and not exit_trade:  # Don't check SL on entry bar or if already exiting
                if self.active_trade.direction == 1:  # Long
                    if row['Low'] <= self.current_sl_price:
                        exit_trade = True
                        exit_price = self.current_sl_price
                        # Determine SL type
                        if self.active_trade.trailed_to_be:
                            exit_reason = "TRAIL_BE_HIT"
                        elif self.active_trade.sl_tightened:
                            exit_reason = "TIGHTEN_SL_HIT"
                        else:
                            exit_reason = "SL_HIT"
                else:  # Short
                    if row['High'] >= self.current_sl_price:
                        exit_trade = True
                        exit_price = self.current_sl_price
                        if self.active_trade.trailed_to_be:
                            exit_reason = "TRAIL_BE_HIT"
                        elif self.active_trade.sl_tightened:
                            exit_reason = "TIGHTEN_SL_HIT"
                        else:
                            exit_reason = "SL_HIT"

            if exit_trade:
                # Close remaining position
                exit_pnl_pips = (exit_price - self.active_trade.entry_price) * 10000 * self.active_trade.direction
                remaining_size = self.active_trade.position_size
                exit_pnl_dollars = exit_pnl_pips * POSITION_SIZE * remaining_size / 10000

                # Total trade PnL = realized from partials + final exit PnL
                total_pnl_pips = self.active_trade.realized_pnl_pips + (exit_pnl_pips * remaining_size)
                total_pnl_dollars = self.active_trade.realized_pnl_dollars + exit_pnl_dollars

                self.active_trade.exit_time = timestamp
                self.active_trade.exit_price = exit_price
                self.active_trade.exit_idx = self.current_idx
                self.active_trade.pnl_pips = total_pnl_pips
                self.active_trade.pnl_dollars = total_pnl_dollars
                self.active_trade.exit_reason = exit_reason

                self.exits.append({
                    'time': str(timestamp),
                    'price': float(exit_price),
                    'pnl': float(total_pnl_pips),
                    'pnl_dollars': float(total_pnl_dollars),
                    'exit_reason': str(exit_reason),
                    'direction': int(self.active_trade.direction),  # 1=Long, -1=Short
                    'had_partials': bool(len(self.active_trade.partial_exits) > 0)
                })

                self.completed_trades.append(self.active_trade)

                # Update comprehensive stats
                self.stats['trades'] += 1
                self.stats['total_pips'] += exit_pnl_pips * remaining_size  # Only add remaining (partials already added)
                self.stats['total_pnl'] += exit_pnl_dollars
                self.stats['pnl_history'].append(total_pnl_dollars)
                self.stats['total_bars_held'] += self.bars_held

                # Win/Loss tracking
                if total_pnl_dollars > 0:
                    self.stats['wins'] += 1
                    self.stats['gross_profit'] += total_pnl_dollars
                    self.stats['max_win'] = max(self.stats['max_win'], total_pnl_dollars)
                    self.stats['consecutive_wins'] += 1
                    self.stats['consecutive_losses'] = 0
                    self.stats['max_consecutive_wins'] = max(self.stats['max_consecutive_wins'], self.stats['consecutive_wins'])
                else:
                    self.stats['losses'] += 1
                    self.stats['gross_loss'] += abs(total_pnl_dollars)
                    self.stats['max_loss'] = min(self.stats['max_loss'], total_pnl_dollars)
                    self.stats['consecutive_losses'] += 1
                    self.stats['consecutive_wins'] = 0
                    self.stats['max_consecutive_losses'] = max(self.stats['max_consecutive_losses'], self.stats['consecutive_losses'])

                # Long/Short tracking
                if self.active_trade.direction == 1:
                    self.stats['long_trades'] += 1
                    self.stats['long_pnl'] += total_pnl_dollars
                    if total_pnl_dollars > 0:
                        self.stats['long_wins'] += 1
                else:
                    self.stats['short_trades'] += 1
                    self.stats['short_pnl'] += total_pnl_dollars
                    if total_pnl_dollars > 0:
                        self.stats['short_wins'] += 1

                # Exit reason tracking
                if 'SL_HIT' in exit_reason:
                    self.stats['sl_count'] += 1
                elif 'TRAIL_BE' in exit_reason:
                    self.stats['trail_be_count'] += 1
                elif 'TIGHTEN' in exit_reason:
                    self.stats['tighten_sl_count'] += 1
                elif 'RL_EXIT' in exit_reason or 'RL_PARTIAL' in exit_reason:
                    self.stats['rl_exit_count'] += 1

                # Partial count
                if self.active_trade.partial_exits:
                    self.stats['partial_count'] += len(self.active_trade.partial_exits)

                current_trade_info = None
                self.active_trade = None
            else:
                self.bars_held += 1

        # Get candle window
        start_idx = max(0, self.current_idx - WINDOW_BARS)
        window = self.price_data.iloc[start_idx:self.current_idx + 1]

        # Build candles data
        candles = {
            'timestamps': [str(t) for t in window.index],
            'open': window['Open'].tolist(),
            'high': window['High'].tolist(),
            'low': window['Low'].tolist(),
            'close': window['Close'].tolist()
        }

        # Filter entries/exits to window
        window_start = str(window.index[0])
        entries_in_window = [e for e in self.entries if e['time'] >= window_start]
        exits_in_window = [e for e in self.exits if e['time'] >= window_start]

        # Trade zone - show CURRENT SL (which may have been tightened by RL)
        trade_zone = None
        if self.active_trade:
            trade_zone = {
                'x0': str(self.active_trade.entry_time),
                'x1': str(timestamp),
                'entry_price': float(self.active_trade.entry_price),
                'sl': float(self.current_sl_price),  # Dynamic SL (can be tightened by RL)
                'original_sl': float(self.active_trade.sl),  # Original SL for comparison
                'tp': float(self.active_trade.tp),
                'sl_changed': bool(abs(self.current_sl_price - self.active_trade.sl) > 0.00001),
                'direction': int(self.active_trade.direction)
            }

        # Trade history for table with detailed PnL breakdown
        # ACTIVE trade first, then completed trades (most recent first)
        trade_history = []

        # Add ACTIVE trade first (if any) - shows real-time updates
        if self.active_trade is not None:
            current_price = self.price_data.iloc[self.current_idx]['Close']
            current_pnl_pips = (current_price - self.active_trade.entry_price) * 10000 * self.active_trade.direction
            unrealized_pnl = current_pnl_pips * PIP_VALUE / 10000 * self.active_trade.position_size
            total_pnl = self.active_trade.realized_pnl_dollars + unrealized_pnl

            # Build partial exit details with times
            active_partial_details = []
            if self.active_trade.partial_exits:
                for i, p in enumerate(self.active_trade.partial_exits):
                    active_partial_details.append({
                        'num': i + 1,
                        'pips': float(p['pips']),
                        'dollars': float(p['dollars']),
                        'size_pct': float(p['size'] * 100),
                        'time': str(p.get('time', '')),
                        'time_pretty': format_datetime_pretty(p.get('time', ''))
                    })

            trade_history.append({
                'is_active': True,
                'direction': int(self.active_trade.direction),
                'entry_price': float(self.active_trade.entry_price),
                'exit_price': None,  # Not exited yet
                'current_price': float(current_price),
                'pnl_pips': float(current_pnl_pips),
                'pnl_dollars': float(total_pnl),
                'unrealized_pnl': float(unrealized_pnl),
                'exit_reason': '',
                'action_counts': {k: int(v) for k, v in self.active_trade.action_counts.items()},
                'sl_tightened': bool(self.active_trade.sl_tightened),
                'trailed_to_be': bool(self.active_trade.trailed_to_be),
                'num_partials': int(len(self.active_trade.partial_exits)) if self.active_trade.partial_exits else 0,
                'partial_pnl': float(self.active_trade.realized_pnl_dollars),
                'partial_details': active_partial_details,
                'position_size_pct': float(self.active_trade.position_size * 100),
                'bars_held': int(self.bars_held),
                'sl_price': float(self.current_sl_price),
                'original_sl': float(self.active_trade.sl),
                'tp_price': float(self.active_trade.tp),
                'mfe': float(self.active_trade.max_favorable_pips),
                'mae': float(self.active_trade.max_adverse_pips),
                # DateTime fields
                'entry_time': str(self.active_trade.entry_time),
                'entry_time_pretty': format_datetime_pretty(self.active_trade.entry_time),
                'exit_time': None,
                'exit_time_pretty': None,
                # Full action history
                'action_history': list(self.active_trade.action_history)
            })

        # Add completed trades - REVERSED (most recent first)
        max_completed = 14 if self.active_trade else 15
        for t in reversed(self.completed_trades[-max_completed:]):
            # Calculate individual partial PnL breakdown with times
            partial_details = []
            if t.partial_exits:
                for i, p in enumerate(t.partial_exits):
                    partial_details.append({
                        'num': i + 1,
                        'pips': float(p['pips']),
                        'dollars': float(p['dollars']),
                        'size_pct': float(p['size'] * 100),
                        'time': str(p.get('time', '')),
                        'time_pretty': format_datetime_pretty(p.get('time', ''))
                    })

            # Final exit PnL (remaining position)
            final_exit_pnl = float(t.pnl_dollars - t.realized_pnl_dollars) if t.pnl_dollars else 0

            trade_history.append({
                'is_active': False,
                'direction': int(t.direction),
                'entry_price': float(t.entry_price),
                'exit_price': float(t.exit_price) if t.exit_price else 0,
                'pnl_pips': float(t.pnl_pips) if t.pnl_pips is not None else 0,
                'pnl_dollars': float(t.pnl_dollars) if t.pnl_dollars is not None else 0,
                'exit_reason': str(t.exit_reason) if t.exit_reason else '',
                'action_counts': {k: int(v) for k, v in t.action_counts.items()},
                'sl_tightened': bool(t.sl_tightened),
                'trailed_to_be': bool(t.trailed_to_be),
                'num_partials': int(len(t.partial_exits)) if t.partial_exits else 0,
                'partial_details': partial_details,
                'partial_pnl': float(t.realized_pnl_dollars),
                'final_pnl': final_exit_pnl,
                'exit_position_pct': float(t.position_size * 100),
                # DateTime fields
                'entry_time': str(t.entry_time),
                'entry_time_pretty': format_datetime_pretty(t.entry_time),
                'exit_time': str(t.exit_time) if t.exit_time else '',
                'exit_time_pretty': format_datetime_pretty(t.exit_time) if t.exit_time else '',
                # Full action history
                'action_history': list(t.action_history),
                'sl_price': float(t.sl),
                'tp_price': float(t.tp)
            })

        self.current_idx += 1

        # Compute derived metrics
        computed_metrics = self._compute_metrics()

        return {
            'timestamp': timestamp,
            'candles': candles,
            'entries': entries_in_window,
            'exits': exits_in_window,
            'action': action,
            'probs': probs,
            'model_output': model_output,
            'activations': activations,
            'trade_zone': trade_zone,
            'current_trade': current_trade_info,
            'stats': self.stats,
            'metrics': computed_metrics,
            'trade_history': trade_history
        }

    def _compute_metrics(self) -> Dict:
        """Compute derived trading metrics."""
        s = self.stats
        n = s['trades']

        if n == 0:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_trade': 0.0,
                'avg_duration': 0.0,
                'long_pct': 0.0,
                'short_pct': 0.0,
                'long_win_rate': 0.0,
                'short_win_rate': 0.0,
                'expectancy': 0.0,
                'return_pct': 0.0,
                'max_drawdown': 0.0,
                'rl_exit_pct': 0.0,
                'sl_hit_pct': 0.0,
                'partial_per_trade': 0.0
            }

        # Basic rates
        win_rate = (s['wins'] / n * 100) if n > 0 else 0
        long_pct = (s['long_trades'] / n * 100) if n > 0 else 0
        short_pct = (s['short_trades'] / n * 100) if n > 0 else 0
        long_win_rate = (s['long_wins'] / s['long_trades'] * 100) if s['long_trades'] > 0 else 0
        short_win_rate = (s['short_wins'] / s['short_trades'] * 100) if s['short_trades'] > 0 else 0

        # Averages
        avg_win = s['gross_profit'] / s['wins'] if s['wins'] > 0 else 0
        avg_loss = s['gross_loss'] / s['losses'] if s['losses'] > 0 else 0
        avg_trade = s['total_pnl'] / n
        avg_duration = s['total_bars_held'] / n

        # Profit Factor
        profit_factor = s['gross_profit'] / s['gross_loss'] if s['gross_loss'] > 0 else (999.9 if s['gross_profit'] > 0 else 0)

        # Sharpe Ratio (annualized, assuming 15M bars)
        # ~35,000 bars per year for 15M data
        bars_per_year = 35040  # 365.25 * 24 * 4
        if len(s['pnl_history']) >= 2:
            returns = np.array(s['pnl_history'])
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                trades_per_year = bars_per_year / max(1, s['total_bars_held'] / n)
                sharpe_ratio = (mean_return / std_return) * np.sqrt(trades_per_year)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Expectancy (avg profit per $1 risked, assuming 2% risk per trade)
        # expectancy = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_loss if avg_loss > 0
        expectancy = avg_trade

        # Return % (on notional $2M)
        return_pct = (s['total_pnl'] / POSITION_SIZE * 100)

        # Max Drawdown (simple running calculation)
        max_drawdown = 0.0
        if s['pnl_history']:
            running_pnl = 0
            peak = 0
            for pnl in s['pnl_history']:
                running_pnl += pnl
                peak = max(peak, running_pnl)
                drawdown = peak - running_pnl
                max_drawdown = max(max_drawdown, drawdown)

        # Exit type percentages
        rl_exit_pct = (s['rl_exit_count'] / n * 100) if n > 0 else 0
        sl_hit_pct = (s['sl_count'] / n * 100) if n > 0 else 0
        partial_per_trade = s['partial_count'] / n if n > 0 else 0

        return {
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'avg_win': round(avg_win, 0),
            'avg_loss': round(avg_loss, 0),
            'avg_trade': round(avg_trade, 0),
            'avg_duration': round(avg_duration, 1),
            'long_pct': round(long_pct, 1),
            'short_pct': round(short_pct, 1),
            'long_win_rate': round(long_win_rate, 1),
            'short_win_rate': round(short_win_rate, 1),
            'expectancy': round(expectancy, 0),
            'return_pct': round(return_pct, 3),
            'max_drawdown': round(max_drawdown, 0),
            'rl_exit_pct': round(rl_exit_pct, 1),
            'sl_hit_pct': round(sl_hit_pct, 1),
            'partial_per_trade': round(partial_per_trade, 2),
            'max_consec_wins': s['max_consecutive_wins'],
            'max_consec_losses': s['max_consecutive_losses'],
            'max_win': round(s['max_win'], 0),
            'max_loss': round(s['max_loss'], 0),
            'long_pnl': round(s['long_pnl'], 0),
            'short_pnl': round(s['short_pnl'], 0)
        }


# ============================================================================
# WebSocket Server
# ============================================================================

async def run_server():
    """Run the visualization server."""
    import aiohttp
    from aiohttp import web

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon) acceleration")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA acceleration")
    else:
        device = "cpu"
        print("Using CPU")

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    config = PPOConfig(device=device)
    model = ActorCriticWithActivations(config)

    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model (trained for {checkpoint.get('timestep', 'unknown')} steps)")
    else:
        print("WARNING: No trained model found, using random policy")

    model.to(device)
    model.eval()

    # Load data
    print("Loading OOS data...")
    price_file = Path("/Users/williamsmith/Python_local_Mac/01_trading_strategies/ML_trading_strategies/data/AUDUSD_15M.csv")
    trades_file = DATA_DIR / "trades_test_2022_2025.csv"

    price_data = pd.read_csv(price_file, index_col=0, parse_dates=True)
    trades_data = pd.read_csv(trades_file)

    # Filter price data to OOS period (2022-2025)
    price_data = price_data[(price_data.index >= OOS_START) & (price_data.index <= OOS_END)]
    print(f"Loaded {len(price_data):,} bars ({OOS_START} to {OOS_END}), {len(trades_data):,} trades")

    # Create simulator
    simulator = TradeSimulator(price_data, trades_data, model, device)
    print(f"Trade entries mapped: {len(simulator.trade_entries):,}")

    # Server state
    state = {
        'playing': False,
        'speed': 10,
        'clients': set()
    }

    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        state['clients'].add(ws)

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    cmd = data.get('command')

                    if cmd == 'play':
                        state['playing'] = True
                    elif cmd == 'pause':
                        state['playing'] = False
                    elif cmd == 'step':
                        update = simulator.step()
                        if update:
                            await ws.send_str(json.dumps(update))
                    elif cmd == 'next_trade':
                        # Jump to next trade entry or exit
                        update = simulator.step_to_next_trade()
                        if update:
                            await ws.send_str(json.dumps(update))
                    elif cmd == 'reset':
                        simulator.reset()
                        state['playing'] = False
                    elif cmd == 'speed':
                        state['speed'] = data.get('value', 10)
        finally:
            state['clients'].discard(ws)

        return ws

    async def index_handler(request):
        return web.Response(text=HTML_TEMPLATE, content_type='text/html')

    async def playback_loop():
        """Background loop for auto-playback."""
        while True:
            if state['playing'] and state['clients']:
                update = simulator.step()
                if update:
                    msg = json.dumps(update)
                    for client in list(state['clients']):
                        try:
                            await client.send_str(msg)
                        except:
                            state['clients'].discard(client)
                else:
                    state['playing'] = False

            # Sleep based on speed
            await asyncio.sleep(1.0 / max(state['speed'], 1))

    # Create app
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)

    # Start playback loop
    asyncio.create_task(playback_loop())

    # Run server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', PORT)
    await site.start()

    print(f"\n{'='*60}")
    print(f" RL Trade Visualizer V2 Running")
    print(f"{'='*60}")
    print(f" Open: http://localhost:{PORT}")
    print(f" Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    # Open browser
    webbrowser.open(f'http://localhost:{PORT}')

    # Keep running
    while True:
        await asyncio.sleep(3600)


def kill_existing_server(port: int):
    """Kill any existing server on the port."""
    import subprocess
    try:
        result = subprocess.run(
            f"lsof -ti:{port} | xargs kill -9 2>/dev/null",
            shell=True, capture_output=True
        )
        if result.returncode == 0:
            print(f"Killed existing process on port {port}")
            import time
            time.sleep(0.5)
    except Exception:
        pass


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="RL Trade Visualizer")
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    parser.add_argument('--speed', type=int, default=10, help='Initial playback speed')
    args = parser.parse_args()

    globals()['PORT'] = args.port

    # Auto-kill any existing server on this port
    kill_existing_server(args.port)

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
