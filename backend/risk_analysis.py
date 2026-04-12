"""
backend/risk_analysis.py  —  TRENDOPT
========================================
Investment risk metric computation.

Metrics
-------
annualised_volatility   — std of log-returns × √252
sharpe_ratio            — (mean excess return) / std × √252
sortino_ratio           — only penalises downside deviation
max_drawdown            — peak-to-trough loss
var_95                  — 95 % Value-at-Risk (parametric)
expected_return         — mean annualised log return
beta                    — vs. a benchmark (if provided)
information_ratio
"""

import numpy as np
import pandas as pd
from typing import Optional


def _log_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices.astype(float)))


def annualised_volatility(prices: np.ndarray) -> float:
    lr = _log_returns(prices)
    return float(np.std(lr, ddof=1) * np.sqrt(252))


def sharpe_ratio(prices: np.ndarray, rf: float = 0.06) -> float:
    lr      = _log_returns(prices)
    excess  = lr - rf / 252
    std     = np.std(excess, ddof=1)
    return float(np.mean(excess) / std * np.sqrt(252)) if std > 0 else 0.0


def sortino_ratio(prices: np.ndarray, rf: float = 0.06) -> float:
    lr      = _log_returns(prices)
    excess  = lr - rf / 252
    neg     = excess[excess < 0]
    down_std = np.std(neg, ddof=1) if len(neg) > 1 else 1e-9
    return float(np.mean(excess) / down_std * np.sqrt(252))


def max_drawdown(prices: np.ndarray) -> float:
    p    = prices.astype(float)
    peak = np.maximum.accumulate(p)
    dd   = (p - peak) / peak
    return float(np.min(dd))           # negative value


def var_95(prices: np.ndarray) -> float:
    """Parametric (Gaussian) 1-day 95 % VaR as a fraction of price."""
    lr = _log_returns(prices)
    return float(np.mean(lr) - 1.645 * np.std(lr, ddof=1))


def expected_return(prices: np.ndarray) -> float:
    """Annualised mean log return."""
    lr = _log_returns(prices)
    return float(np.mean(lr) * 252)


def beta(prices: np.ndarray, benchmark: Optional[np.ndarray]) -> Optional[float]:
    """Market beta vs. benchmark (e.g. Nifty50 or S&P500)."""
    if benchmark is None or len(benchmark) != len(prices):
        return None
    r_asset = _log_returns(prices)
    r_bench = _log_returns(benchmark)
    n       = min(len(r_asset), len(r_bench))
    cov     = np.cov(r_asset[:n], r_bench[:n])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else None


def compute_all_risk(prices: np.ndarray, rf: float = 0.06) -> dict:
    """
    Compute all risk metrics for a price array.

    Args:
        prices : 1-D closing price array.
        rf     : Annual risk-free rate (default 6 % for India).

    Returns:
        Dict of named risk metrics.
    """
    vol   = annualised_volatility(prices)
    exp_r = expected_return(prices)
    mdd   = max_drawdown(prices)
    v95   = var_95(prices)
    sh    = sharpe_ratio(prices, rf)
    so    = sortino_ratio(prices, rf)

    # Risk rating
    if vol < 0.20:
        risk_level = "Low 🟢"
    elif vol < 0.35:
        risk_level = "Moderate 🟡"
    else:
        risk_level = "High 🔴"

    return {
        "volatility":         round(vol * 100, 2),        # as %
        "sharpe_ratio":       round(sh, 4),
        "sortino_ratio":      round(so, 4),
        "max_drawdown_pct":   round(abs(mdd) * 100, 2),   # as % (positive)
        "var_95_pct":         round(abs(v95) * 100, 2),   # 1-day VaR %
        "expected_return_pct":round(exp_r * 100, 2),      # annualised %
        "risk_level":         risk_level,
    }
