"""Legacy signal helpers backed by shared indicator logic."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from backend.indicator_module import sma


def detect_signals(
    prices: np.ndarray,
    short_window: int = 10,
    long_window: int = 30,
    use_ma_filter: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return local-extrema buy/sell indices with an optional MA trend filter."""
    prices = np.asarray(prices, dtype=float)
    buy_indices = []
    sell_indices = []

    if use_ma_filter:
        short_ma = sma(prices, short_window)
        long_ma = sma(prices, long_window)
    else:
        short_ma = None
        long_ma = None

    for i in range(1, len(prices) - 1):
        is_buy = prices[i] < prices[i - 1] and prices[i] < prices[i + 1]
        is_sell = prices[i] > prices[i - 1] and prices[i] > prices[i + 1]

        if is_buy:
            if use_ma_filter and not np.isnan(short_ma[i]) and not np.isnan(long_ma[i]):
                if short_ma[i] > long_ma[i]:
                    buy_indices.append(i)
            else:
                buy_indices.append(i)
        elif is_sell:
            if use_ma_filter and not np.isnan(short_ma[i]) and not np.isnan(long_ma[i]):
                if short_ma[i] < long_ma[i]:
                    sell_indices.append(i)
            else:
                sell_indices.append(i)

    return np.array(buy_indices, dtype=int), np.array(sell_indices, dtype=int)


def signals_to_dataframe(
    df: pd.DataFrame,
    buy_indices: np.ndarray,
    sell_indices: np.ndarray,
) -> pd.DataFrame:
    """Annotate a DataFrame with boolean signal columns."""
    result = df.copy()
    result["BUY_SIGNAL"] = False
    result["SELL_SIGNAL"] = False

    for idx in buy_indices:
        if 0 <= idx < len(result):
            result.iloc[idx, result.columns.get_loc("BUY_SIGNAL")] = True

    for idx in sell_indices:
        if 0 <= idx < len(result):
            result.iloc[idx, result.columns.get_loc("SELL_SIGNAL")] = True

    return result


__all__ = ["detect_signals", "signals_to_dataframe"]
