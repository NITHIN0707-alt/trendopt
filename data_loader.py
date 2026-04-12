"""
data_loader.py — TRENDOPT
Fetches and preprocesses historical stock data using yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def fetch_stock_data(ticker: str = "RELIANCE.NS", period_years: int = 1) -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE.NS', 'AAPL').
        period_years: Number of years of historical data to fetch.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume.

    Raises:
        ValueError: If no data is returned for the ticker.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * period_years)

    print(f"[DataLoader] Fetching data for {ticker} from {start_date.date()} to {end_date.date()} ...")

    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        raise ConnectionError(f"Failed to download data for '{ticker}': {e}")

    if df is None or df.empty:
        # Try loading from a local CSV fallback (useful for demo / offline mode)
        csv_path = os.path.join(os.path.dirname(__file__), f"synthetic_{ticker.replace('.', '_')}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "synthetic_RELIANCE.csv")
        if os.path.exists(csv_path):
            print(f"[DataLoader] Live download failed — loading synthetic demo data from {csv_path}")
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol and try again.")

    # Flatten MultiIndex columns if present (yfinance ≥ 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only the relevant columns
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()

    # Drop rows with any NaN values
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)

    print(f"[DataLoader] Loaded {len(df)} trading days.")
    return df


def get_closing_prices(df: pd.DataFrame) -> np.ndarray:
    """
    Extract closing prices as a NumPy array.

    Args:
        df: DataFrame with a 'Close' column.

    Returns:
        1-D NumPy array of closing prices.
    """
    return df["Close"].values.astype(float)


def get_price_summary(df: pd.DataFrame) -> dict:
    """
    Return a quick summary of price statistics.

    Args:
        df: OHLCV DataFrame.

    Returns:
        Dict with min, max, mean close price and total trading days.
    """
    closes = get_closing_prices(df)
    return {
        "trading_days": len(df),
        "min_close": float(np.min(closes)),
        "max_close": float(np.max(closes)),
        "mean_close": float(np.mean(closes)),
        "start_date": str(df.index[0].date()),
        "end_date": str(df.index[-1].date()),
    }


if __name__ == "__main__":
    df = fetch_stock_data("RELIANCE.NS")
    print(df.tail())
    print(get_price_summary(df))
