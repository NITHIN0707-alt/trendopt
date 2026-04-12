"""
backend/signal_detector.py  —  TRENDOPT
==========================================
Multi-strategy buy/sell signal detection.

Strategies
----------
1. Local extrema       — local min → BUY, local max → SELL
2. RSI threshold       — RSI < 30 → BUY, RSI > 70 → SELL
3. MA crossover        — golden cross → BUY, death cross → SELL
4. Bollinger breakout  — price < lower band → BUY, > upper → SELL
5. MACD crossover      — MACD crosses above signal → BUY, below → SELL

Public API
----------
detect_signals(df, strategy='combined') → dict
latest_signal(signals, prices)          → str   ('BUY'|'SELL'|'HOLD')
active_alerts(signals, prices, ticker)  → list[str]
"""

import numpy as np
import pandas as pd
from typing import Tuple
from backend.indicator_module import (
    sma, rsi as calc_rsi, bollinger_bands, macd as calc_macd
)


# ── Individual strategy helpers ───────────────────────────────────────────

def _local_extrema(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    buy, sell = [], []
    for i in range(1, len(c) - 1):
        if c[i] < c[i - 1] and c[i] < c[i + 1]:
            buy.append(i)
        elif c[i] > c[i - 1] and c[i] > c[i + 1]:
            sell.append(i)
    return np.array(buy, dtype=int), np.array(sell, dtype=int)


def _rsi_signals(c: np.ndarray, ob=70, os=30) -> Tuple[np.ndarray, np.ndarray]:
    r    = calc_rsi(c)
    buy  = np.where((r < os) & ~np.isnan(r))[0]
    sell = np.where((r > ob) & ~np.isnan(r))[0]
    return buy, sell


def _ma_crossover(c: np.ndarray, short=20, long=50) -> Tuple[np.ndarray, np.ndarray]:
    ms, ml = sma(c, short), sma(c, long)
    buy, sell = [], []
    for i in range(1, len(c)):
        if np.isnan(ms[i]) or np.isnan(ml[i]):
            continue
        if ms[i - 1] < ml[i - 1] and ms[i] >= ml[i]:
            buy.append(i)
        elif ms[i - 1] > ml[i - 1] and ms[i] <= ml[i]:
            sell.append(i)
    return np.array(buy, dtype=int), np.array(sell, dtype=int)


def _bb_signals(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    upper, _, lower = bollinger_bands(c)
    buy  = np.where(~np.isnan(lower) & (c < lower))[0]
    sell = np.where(~np.isnan(upper) & (c > upper))[0]
    return buy, sell


def _macd_crossover(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ml, sl, _ = calc_macd(c)
    buy, sell = [], []
    for i in range(1, len(c)):
        if ml[i - 1] < sl[i - 1] and ml[i] >= sl[i]:
            buy.append(i)
        elif ml[i - 1] > sl[i - 1] and ml[i] <= sl[i]:
            sell.append(i)
    return np.array(buy, dtype=int), np.array(sell, dtype=int)


# ── Public API ────────────────────────────────────────────────────────────

def detect_signals(df: pd.DataFrame, strategy: str = "combined") -> dict:
    """
    Detect buy/sell signals for a given OHLCV DataFrame.

    Args:
        df       : OHLCV DataFrame.
        strategy : One of 'local_extrema', 'rsi', 'ma_crossover',
                   'bb', 'macd', 'combined'.

    Returns:
        Dict with buy_indices, sell_indices and all indicator arrays.
    """
    c = df["Close"].values.astype(float)

    le_b, le_s   = _local_extrema(c)
    rsi_b, rsi_s = _rsi_signals(c)
    ma_b, ma_s   = _ma_crossover(c)
    bb_b, bb_s   = _bb_signals(c)
    mc_b, mc_s   = _macd_crossover(c)

    if strategy == "local_extrema":
        bi, si = le_b, le_s
    elif strategy == "rsi":
        bi, si = rsi_b, rsi_s
    elif strategy == "ma_crossover":
        bi, si = ma_b, ma_s
    elif strategy == "bb":
        bi, si = bb_b, bb_s
    elif strategy == "macd":
        bi, si = mc_b, mc_s
    else:  # combined — union, deduplicated
        bi = np.unique(np.concatenate([le_b, rsi_b, ma_b, bb_b, mc_b]))
        si = np.unique(np.concatenate([le_s, rsi_s, ma_s, bb_s, mc_s]))

    upper, mid, lower = bollinger_bands(c)
    ml, sl, hist      = calc_macd(c)

    return {
        "buy_indices":  bi,
        "sell_indices": si,
        "rsi":          calc_rsi(c),
        "sma20":        sma(c, 20),
        "sma50":        sma(c, 50),
        "sma200":       sma(c, 200),
        "bb_upper":     upper,
        "bb_mid":       mid,
        "bb_lower":     lower,
        "macd_line":    ml,
        "macd_sig":     sl,
        "macd_hist":    hist,
    }


def latest_signal(signals: dict, prices: np.ndarray, window: int = 10) -> str:
    """
    Determine the most recent directional signal.

    Looks back at the last ``window`` days for buy/sell signals;
    integrates RSI to avoid false reversals.
    """
    n       = len(prices)
    cutoff  = n - window
    rb      = any(i >= cutoff for i in signals["buy_indices"])
    rs      = any(i >= cutoff for i in signals["sell_indices"])
    rsi_now = float(signals["rsi"][-1]) if not np.isnan(signals["rsi"][-1]) else 50.0

    if rb and not rs and rsi_now < 65:
        return "BUY"
    if rs and not rb and rsi_now > 35:
        return "SELL"
    return "HOLD"


def active_alerts(signals: dict, prices: np.ndarray, ticker: str) -> list:
    """
    Generate human-readable alert strings for notable conditions.
    """
    alerts = []
    c       = prices
    rsi_now = float(signals["rsi"][-1]) if not np.isnan(signals["rsi"][-1]) else 50.0
    ma50    = signals["sma50"][-1]
    ma200   = signals["sma200"][-1]
    price   = c[-1]

    if rsi_now < 30:
        alerts.append(f"🟢 {ticker} RSI={rsi_now:.1f} — Oversold, potential reversal BUY")
    if rsi_now > 70:
        alerts.append(f"🔴 {ticker} RSI={rsi_now:.1f} — Overbought, potential reversal SELL")
    if not np.isnan(ma50) and not np.isnan(ma200):
        if ma50 > ma200:
            alerts.append(f"✨ {ticker} Golden Cross — MA50 > MA200 (bullish)")
        else:
            alerts.append(f"⚠️ {ticker} Death Cross — MA50 < MA200 (bearish)")
    bb_upper = signals["bb_upper"][-1]
    bb_lower = signals["bb_lower"][-1]
    if not np.isnan(bb_upper) and price > bb_upper:
        alerts.append(f"🔴 {ticker} Price above Bollinger upper band — overbought")
    if not np.isnan(bb_lower) and price < bb_lower:
        alerts.append(f"🟢 {ticker} Price below Bollinger lower band — oversold")

    return alerts
