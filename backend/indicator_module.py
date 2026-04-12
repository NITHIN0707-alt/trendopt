"""
backend/indicator_module.py  —  TRENDOPT
==========================================
Technical indicator computations.

Functions
---------
sma(prices, window)
ema(prices, span)
rsi(prices, period=14)
macd(prices, fast=12, slow=26, signal=9)
bollinger_bands(prices, window=20, num_std=2)
atr(df, period=14)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def sma(prices: np.ndarray, window: int) -> np.ndarray:
    """Simple Moving Average."""
    return pd.Series(prices.astype(float)).rolling(window).mean().values


def ema(prices: np.ndarray, span: int) -> np.ndarray:
    """Exponential Moving Average."""
    return pd.Series(prices.astype(float)).ewm(span=span, adjust=False).mean().values


def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index using Wilder smoothing.

    Returns array in [0, 100]; first ``period`` values are NaN.
    """
    delta = np.diff(prices.astype(float), prepend=np.nan)
    gain  = np.where(delta > 0,  delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    ag    = pd.Series(gain).ewm(alpha=1 / period, adjust=False).mean().values
    al    = pd.Series(loss).ewm(alpha=1 / period, adjust=False).mean().values
    rs    = np.where(al == 0, np.inf, ag / al)
    r     = 100 - 100 / (1 + rs)
    r[:period] = np.nan
    return r


def macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD indicator.

    Returns:
        (macd_line, signal_line, histogram)
    """
    fast_e  = ema(prices, fast)
    slow_e  = ema(prices, slow)
    line    = fast_e - slow_e
    sig     = pd.Series(line).ewm(span=signal, adjust=False).mean().values
    hist    = line - sig
    return line, sig, hist


def bollinger_bands(
    prices: np.ndarray,
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.

    Returns:
        (upper, middle, lower)
    """
    s   = pd.Series(prices.astype(float))
    mid = s.rolling(window).mean()
    std = s.rolling(window).std()
    return (mid + num_std * std).values, mid.values, (mid - num_std * std).values


def atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """
    Average True Range — volatility indicator.

    Args:
        df     : OHLCV DataFrame.
        period : Smoothing period.

    Returns:
        ATR array.
    """
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr  = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values


def compute_all(df: pd.DataFrame) -> dict:
    """
    Compute all indicators for a given OHLCV DataFrame.

    Returns a dict of named arrays ready for charting.
    """
    c = df["Close"].values.astype(float)
    upper, mid, lower = bollinger_bands(c)
    ml, sl, hist      = macd(c)
    return {
        "close":      c,
        "sma20":      sma(c, 20),
        "sma50":      sma(c, 50),
        "sma200":     sma(c, 200),
        "ema12":      ema(c, 12),
        "ema26":      ema(c, 26),
        "rsi":        rsi(c),
        "macd_line":  ml,
        "macd_sig":   sl,
        "macd_hist":  hist,
        "bb_upper":   upper,
        "bb_mid":     mid,
        "bb_lower":   lower,
        "atr":        atr(df),
    }
