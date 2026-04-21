"""Canonical backend exports for TRENDOPT."""

from .brute import brute_force
from .data_fetcher import fetch_stock_data, get_price_stats, resolve_ticker
from .dp import dp_cooldown
from .greedy import greedy
from .prediction_model import consensus_forecast, run_all_predictions
from .signal_detector import active_alerts, detect_signals, latest_signal

__all__ = [
    "active_alerts",
    "brute_force",
    "consensus_forecast",
    "detect_signals",
    "dp_cooldown",
    "fetch_stock_data",
    "get_price_stats",
    "greedy",
    "latest_signal",
    "resolve_ticker",
    "run_all_predictions",
]
