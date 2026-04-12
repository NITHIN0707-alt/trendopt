"""
signals.py — TRENDOPT
Buy / Sell signal detection using local-extrema analysis.

A BUY  signal is generated at a local minimum  (price[i] < price[i-1] and price[i] < price[i+1]).
A SELL signal is generated at a local maximum  (price[i] > price[i-1] and price[i] > price[i+1]).

Additionally, a moving-average crossover filter is applied as an optional
second confirmation layer.
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# Helper — moving average
# ---------------------------------------------------------------------------

def _moving_average(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a simple moving average, padding the head with NaN.

    Args:
        prices : 1-D price array.
        window : Look-back window length.

    Returns:
        1-D array of the same length with the rolling mean.
    """
    ma = np.full(len(prices), np.nan)
    for i in range(window - 1, len(prices)):
        ma[i] = np.mean(prices[i - window + 1: i + 1])
    return ma


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_signals(
    prices: np.ndarray,
    short_window: int = 10,
    long_window: int = 30,
    use_ma_filter: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect buy and sell signal indices in a price series.

    Primary rule   : local-extrema (window-1 neighbourhood).
    Secondary rule : optional MA crossover confirmation.

    Args:
        prices       : 1-D array of closing prices.
        short_window : Short MA period for crossover filter.
        long_window  : Long MA period for crossover filter.
        use_ma_filter: If True, require short MA > long MA for BUY / short < long for SELL.

    Returns:
        Tuple (buy_indices, sell_indices) — 1-D arrays of integer indices.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)

    # Compute MAs if needed
    if use_ma_filter:
        short_ma = _moving_average(prices, short_window)
        long_ma  = _moving_average(prices, long_window)
    else:
        short_ma = long_ma = None

    buy_indices  = []
    sell_indices = []

    for i in range(1, n - 1):
        # --- Local minimum → potential BUY
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            if use_ma_filter and not np.isnan(short_ma[i]) and not np.isnan(long_ma[i]):
                if short_ma[i] > long_ma[i]:          # upward trend confirmation
                    buy_indices.append(i)
            else:
                buy_indices.append(i)

        # --- Local maximum → potential SELL
        elif prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            if use_ma_filter and not np.isnan(short_ma[i]) and not np.isnan(long_ma[i]):
                if short_ma[i] < long_ma[i]:           # downward trend confirmation
                    sell_indices.append(i)
            else:
                sell_indices.append(i)

    return np.array(buy_indices, dtype=int), np.array(sell_indices, dtype=int)


def signals_to_dataframe(
    df: pd.DataFrame,
    buy_indices: np.ndarray,
    sell_indices: np.ndarray,
) -> pd.DataFrame:
    """
    Annotate the original OHLCV DataFrame with signal columns.

    Args:
        df           : Original OHLCV DataFrame (index = dates).
        buy_indices  : Integer positions of BUY signals.
        sell_indices : Integer positions of SELL signals.

    Returns:
        Copy of df with two added boolean columns: 'BUY_SIGNAL', 'SELL_SIGNAL'.
    """
    result = df.copy()
    result["BUY_SIGNAL"]  = False
    result["SELL_SIGNAL"] = False

    closes = df["Close"].values
    for idx in buy_indices:
        if 0 <= idx < len(result):
            result.iloc[idx, result.columns.get_loc("BUY_SIGNAL")] = True

    for idx in sell_indices:
        if 0 <= idx < len(result):
            result.iloc[idx, result.columns.get_loc("SELL_SIGNAL")] = True

    return result


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prices = np.array([10, 8, 12, 11, 15, 13, 16, 14, 18], dtype=float)
    buys, sells = detect_signals(prices, use_ma_filter=False)
    print("Buy  indices:", buys)
    print("Sell indices:", sells)
