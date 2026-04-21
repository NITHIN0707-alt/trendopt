"""Compatibility wrapper around the canonical data fetcher."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.data_fetcher import (
    fetch_stock_data as _fetch_stock_data,
    get_price_stats,
    resolve_ticker,
)


def fetch_stock_data(ticker: str = "RELIANCE.NS", period_years: int = 1) -> pd.DataFrame:
    """Return the cleaned OHLCV DataFrame from the canonical backend fetcher."""
    del period_years
    df, _ = _fetch_stock_data(resolve_ticker(ticker))
    return df


def get_closing_prices(df: pd.DataFrame) -> np.ndarray:
    """Extract closing prices as a NumPy array."""
    return df["Close"].to_numpy(dtype=float)


def get_price_summary(df: pd.DataFrame) -> dict:
    """Backwards-compatible alias for the richer backend price summary."""
    stats = get_price_stats(df)
    closes = get_closing_prices(df)
    return {
        "trading_days": stats["trading_days"],
        "min_close": float(closes.min()),
        "max_close": float(closes.max()),
        "mean_close": float(closes.mean()),
        "start_date": stats["start_date"],
        "end_date": stats["end_date"],
    }


__all__ = ["fetch_stock_data", "get_closing_prices", "get_price_summary"]
