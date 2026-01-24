"""
Utility functions for the RL Trade Visualizer.
"""

from datetime import datetime
import pandas as pd
import numpy as np


def format_datetime_pretty(dt_str: str) -> str:
    """
    Format datetime string to human-readable format.

    Example: "2024-01-23 11:30:00" -> "Tue 23rd Jan 11:30am"

    Args:
        dt_str: Datetime string in various formats

    Returns:
        Human-readable datetime string
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
        return str(dt_str)


def format_pnl(value: float, include_sign: bool = True) -> str:
    """Format PnL value with optional sign and dollar formatting."""
    if include_sign:
        sign = '+' if value >= 0 else ''
        return f"{sign}${abs(value):,.0f}"
    return f"${abs(value):,.0f}"


def format_pips(value: float, include_sign: bool = True) -> str:
    """Format pip value with optional sign."""
    if include_sign:
        sign = '+' if value >= 0 else ''
        return f"{sign}{value:.1f}"
    return f"{abs(value):.1f}"


def format_percentage(value: float, include_sign: bool = False) -> str:
    """Format percentage value."""
    if include_sign and value > 0:
        return f"+{value:.1f}%"
    return f"{value:.1f}%"


def clean_nan(value, default=0.0):
    """Replace NaN/Inf values with default."""
    if np.isnan(value) or np.isinf(value):
        return default
    return value


def safe_float(val, default=0.0):
    """Safely convert to float, handling NaN/Inf."""
    try:
        if np.isnan(val) or np.isinf(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def kill_existing_server(port: int) -> None:
    """
    Kill any existing server on the specified port.

    Args:
        port: Port number to clear
    """
    import subprocess
    import time

    try:
        result = subprocess.run(
            f"lsof -ti:{port} | xargs kill -9 2>/dev/null",
            shell=True,
            capture_output=True
        )
        if result.returncode == 0:
            print(f"Killed existing process on port {port}")
            time.sleep(0.5)
    except Exception:
        pass
