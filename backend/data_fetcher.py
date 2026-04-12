"""
backend/data_fetcher.py  —  TRENDOPT
======================================
Robust stock data acquisition via yfinance.

ROOT CAUSE OF "SYNTHETIC DATA" ISSUE
--------------------------------------
Yahoo Finance requires:
  1. A browser-like User-Agent header
  2. A valid session cookie (obtained by visiting finance.yahoo.com first)
  3. A crumb token for some API calls

This file handles all three automatically.

Strategy (tried in order):
  1. yf.Ticker with cookie session    <- most reliable fix
  2. yf.Ticker plain                  <- simple attempt
  3. yf.download with date range      <- bulk download fallback
  4. Synthetic GBM data               <- ONLY if all three fail

Public API
----------
resolve_ticker(query)      -> str
fetch_stock_data(ticker)   -> (DataFrame, DataSource)
get_price_stats(df)        -> dict
diagnose_ticker(ticker)    -> dict
"""

import warnings
import time
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Tuple, Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ── DataSource result type ────────────────────────────────────────────────
class DataSource:
    def __init__(self, is_live: bool, method: str, error: Optional[str] = None):
        self.is_live = is_live
        self.method  = method
        self.error   = error

    def __repr__(self):
        return f"DataSource(is_live={self.is_live}, method={self.method!r})"


# ── Ticker name → yfinance symbol map ────────────────────────────────────
TICKER_MAP = {
    # Indian (NSE)
    "reliance":    "RELIANCE.NS", "tcs":          "TCS.NS",
    "infosys":     "INFY.NS",     "infy":          "INFY.NS",
    "wipro":       "WIPRO.NS",    "hdfc":          "HDFCBANK.NS",
    "hdfcbank":    "HDFCBANK.NS", "icici":         "ICICIBANK.NS",
    "icicibank":   "ICICIBANK.NS","sbi":           "SBIN.NS",
    "itc":         "ITC.NS",      "bajaj":         "BAJFINANCE.NS",
    "bajajfinance":"BAJFINANCE.NS","hcl":          "HCLTECH.NS",
    "hcltech":     "HCLTECH.NS",  "sunpharma":     "SUNPHARMA.NS",
    "maruti":      "MARUTI.NS",   "titan":         "TITAN.NS",
    "kotak":       "KOTAKBANK.NS","kotakbank":     "KOTAKBANK.NS",
    "axis":        "AXISBANK.NS", "axisbank":      "AXISBANK.NS",
    "ltim":        "LTIM.NS",     "nestleind":     "NESTLEIND.NS",
    "adani":       "ADANIENT.NS", "adanient":      "ADANIENT.NS",
    "powergrid":   "POWERGRID.NS","ntpc":          "NTPC.NS",
    "ongc":        "ONGC.NS",     "tatamotors":    "TATAMOTORS.NS",
    "tatamotor":   "TATAMOTORS.NS","tatasteel":    "TATASTEEL.NS",
    "tatapower":   "TATAPOWER.NS","hindalco":      "HINDALCO.NS",
    "asianpaint":  "ASIANPAINT.NS","drreddy":      "DRREDDY.NS",
    "cipla":       "CIPLA.NS",    "techm":         "TECHM.NS",
    "bajajfinsv":  "BAJAJFINSV.NS","jswsteel":    "JSWSTEEL.NS",
    "nifty50":     "^NSEI",       "sensex":        "^BSESN",
    # US stocks
    "apple":  "AAPL", "aapl":  "AAPL", "microsoft": "MSFT", "msft": "MSFT",
    "google": "GOOGL","googl": "GOOGL","alphabet":  "GOOGL",
    "amazon": "AMZN", "amzn":  "AMZN", "tesla":     "TSLA", "tsla": "TSLA",
    "meta":   "META", "facebook":"META","nvidia":   "NVDA", "nvda": "NVDA",
    "netflix":"NFLX", "nflx":  "NFLX", "amd":       "AMD",  "intel":"INTC",
    "intc":   "INTC", "paypal":"PYPL",  "pypl":      "PYPL", "uber": "UBER",
    "spotify":"SPOT", "jpmorgan":"JPM", "goldman":   "GS",   "visa": "V",
    "mastercard":"MA","pfizer": "PFE",  "exxon":     "XOM",
    "sp500":  "^GSPC","dow":    "^DJI",
}

# Accurate real-world approximate prices (used only for synthetic fallback)
_SEED_PRICES = {
    "TCS.NS": 3200.0,        "RELIANCE.NS": 1280.0,   "INFY.NS": 1590.0,
    "WIPRO.NS": 260.0,       "HDFCBANK.NS": 1750.0,   "ICICIBANK.NS": 1300.0,
    "SBIN.NS": 780.0,        "ITC.NS": 430.0,          "BAJFINANCE.NS": 8800.0,
    "HCLTECH.NS": 1550.0,    "KOTAKBANK.NS": 2200.0,  "AXISBANK.NS": 1150.0,
    "SUNPHARMA.NS": 1800.0,  "MARUTI.NS": 12000.0,    "TITAN.NS": 3200.0,
    "TATAMOTORS.NS": 650.0,  "TATASTEEL.NS": 145.0,   "ASIANPAINT.NS": 2200.0,
    "BAJAJFINSV.NS": 1850.0, "DRREDDY.NS": 6400.0,    "NESTLEIND.NS": 2200.0,
    "ADANIENT.NS": 2300.0,   "TECHM.NS": 1450.0,      "JSWSTEEL.NS": 900.0,
    "HINDALCO.NS": 620.0,    "ONGC.NS": 260.0,        "NTPC.NS": 330.0,
    "POWERGRID.NS": 290.0,   "CIPLA.NS": 1500.0,
    "AAPL": 200.0,   "MSFT": 380.0,  "GOOGL": 165.0, "AMZN": 185.0,
    "TSLA": 240.0,   "META": 580.0,  "NVDA":  880.0,  "NFLX": 970.0,
    "AMD":  100.0,   "INTC":  20.0,  "PYPL":   65.0,  "UBER":  75.0,
}

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

# ── Browser-like session (THE KEY FIX) ───────────────────────────────────

def _make_session() -> requests.Session:
    """
    Create a requests.Session with full browser headers.
    Visit finance.yahoo.com first to obtain the required cookies.
    This is the #1 fix for yfinance connection failures.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language":  "en-US,en;q=0.9",
        "Accept-Encoding":  "gzip, deflate, br",
        "Connection":       "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest":   "document",
        "Sec-Fetch-Mode":   "navigate",
        "Sec-Fetch-Site":   "none",
        "Cache-Control":    "max-age=0",
    })
    # Visit Yahoo Finance to get cookies (critical step)
    try:
        session.get("https://finance.yahoo.com", timeout=10)
    except Exception:
        pass  # Even if this fails, the headers alone help
    return session


# ── DataFrame normaliser ──────────────────────────────────────────────────

def _clean(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalise a raw yfinance DataFrame. Returns None if unusable."""
    if raw is None or raw.empty:
        return None
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Close"}, inplace=True)
    keep = [c for c in REQUIRED_COLS if c in df.columns]
    if "Close" not in keep:
        return None
    df = df[keep].copy()
    df.dropna(subset=["Close"], inplace=True)
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df if len(df) >= 10 else None


# ── Fetch methods (ordered best → fallback) ───────────────────────────────

def _method1_session_history(ticker: str) -> Optional[pd.DataFrame]:
    """
    METHOD 1 (BEST): yf.Ticker with a browser-like cookie session.
    This is the primary fix for Yahoo Finance connection issues.
    """
    try:
        session = _make_session()
        t   = yf.Ticker(ticker, session=session)
        raw = t.history(period="1y", auto_adjust=True, actions=False)
        return _clean(raw)
    except Exception:
        return None


def _method2_plain_history(ticker: str) -> Optional[pd.DataFrame]:
    """METHOD 2: yf.Ticker without custom session."""
    try:
        raw = yf.Ticker(ticker).history(period="1y", auto_adjust=True, actions=False)
        return _clean(raw)
    except Exception:
        return None


def _method3_download(ticker: str) -> Optional[pd.DataFrame]:
    """METHOD 3: yf.download with explicit date range."""
    try:
        end   = datetime.today()
        start = end - timedelta(days=370)
        raw   = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True, threads=False,
        )
        return _clean(raw)
    except Exception:
        return None


def _method4_download_period(ticker: str) -> Optional[pd.DataFrame]:
    """METHOD 4: yf.download with period string."""
    try:
        raw = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        return _clean(raw)
    except Exception:
        return None


# ── Synthetic fallback ────────────────────────────────────────────────────

def _synthetic(ticker: str, n: int = 252) -> pd.DataFrame:
    """
    GBM-based OHLCV data seeded with real approximate prices.
    ONLY used when ALL live methods fail.
    """
    seed   = abs(hash(ticker)) % (2 ** 31)
    rng    = np.random.default_rng(seed)
    s0     = _SEED_PRICES.get(ticker,
             1200.0 if ticker.endswith(".NS") else 150.0)
    prices = s0 * np.cumprod(1 + rng.normal(0.08 / n, 0.015, n))
    dates  = pd.bdate_range(end=datetime.today(), periods=n)
    nx     = lambda lo, hi: rng.uniform(lo, hi, n)
    df = pd.DataFrame({
        "Open":   (prices * nx(0.997, 1.003)).round(2),
        "High":   (prices * nx(1.003, 1.018)).round(2),
        "Low":    (prices * nx(0.982, 0.997)).round(2),
        "Close":  prices.round(2),
        "Volume": rng.integers(500_000, 8_000_000, n).astype(float),
    }, index=dates)
    df.index.name = "Date"
    return df


# ── Public API ────────────────────────────────────────────────────────────

def resolve_ticker(query: str) -> str:
    """Map a friendly name or partial ticker to a yfinance symbol."""
    key = query.strip().lower().replace(" ", "")
    return TICKER_MAP.get(key, query.strip().upper())


def fetch_stock_data(
    ticker: str,
    period_years: int = 1,
) -> Tuple[pd.DataFrame, DataSource]:
    """
    Download one year of OHLCV data for ticker.

    Tries 4 methods in order. Returns synthetic data ONLY if all fail.

    Returns:
        (DataFrame, DataSource)
        DataSource.is_live = True  -> REAL Yahoo Finance prices
        DataSource.is_live = False -> synthetic demo data (not real)
    """
    if not ticker or not ticker.strip():
        raise ValueError("Ticker symbol cannot be empty.")

    errors = []

    # Method 1 — session with cookies (THE KEY FIX)
    df = _method1_session_history(ticker)
    if df is not None:
        return df, DataSource(True, "session+cookies", None)
    errors.append("session+cookies failed")

    # Method 2 — plain Ticker
    df = _method2_plain_history(ticker)
    if df is not None:
        return df, DataSource(True, "plain_history", None)
    errors.append("plain_history failed")

    # Method 3 — download date range
    df = _method3_download(ticker)
    if df is not None:
        return df, DataSource(True, "download_dates", None)
    errors.append("download_dates failed")

    # Method 4 — download period
    df = _method4_download_period(ticker)
    if df is not None:
        return df, DataSource(True, "download_period", None)
    errors.append("download_period failed")

    # All failed — synthetic fallback
    err_msg = " | ".join(errors)
    print("\n" + "!"*62)
    print("  SYNTHETIC DATA WARNING")
    print(f"  Ticker  : {ticker}")
    print(f"  Reason  : {err_msg}")
    print("  Fix     : pip install --upgrade yfinance requests")
    print("!"*62 + "\n")

    return (
        _synthetic(ticker, int(252 * period_years)),
        DataSource(False, "synthetic", err_msg),
    )


def get_price_stats(df: pd.DataFrame) -> dict:
    """Return summary statistics dict from an OHLCV DataFrame."""
    c   = df["Close"].values.astype(float)
    chg = float((c[-1] - c[-2]) / c[-2] * 100) if len(c) >= 2 else 0.0
    return {
        "current_price": float(c[-1]),
        "prev_close":    float(c[-2]) if len(c) >= 2 else float(c[-1]),
        "change_pct":    round(chg, 2),
        "high_52w":      float(c.max()),
        "low_52w":       float(c.min()),
        "avg_volume":    float(df["Volume"].mean()),
        "start_date":    str(df.index[0].date()),
        "end_date":      str(df.index[-1].date()),
        "trading_days":  int(len(df)),
    }


def diagnose_ticker(ticker: str) -> dict:
    """
    Run all 4 fetch methods and report results.
    Useful for debugging.

    Usage:
        python backend/data_fetcher.py TCS.NS
    """
    results = {}
    methods = [
        ("session+cookies",  _method1_session_history),
        ("plain_history",    _method2_plain_history),
        ("download_dates",   _method3_download),
        ("download_period",  _method4_download_period),
    ]
    for name, fn in methods:
        df = fn(ticker)
        if df is not None:
            results[name] = (
                f"OK — {len(df)} rows | "
                f"last close = {df['Close'].iloc[-1]:,.2f} | "
                f"{df.index[0].date()} → {df.index[-1].date()}"
            )
        else:
            results[name] = "FAILED"
    return results


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ticker = resolve_ticker(sys.argv[1]) if len(sys.argv) > 1 else "TCS.NS"
    print(f"\nDiagnosing: {ticker}\n")
    for method, result in diagnose_ticker(ticker).items():
        icon = "✅" if result.startswith("OK") else "❌"
        print(f"  {icon}  {method:<22}: {result}")

    print(f"\nFull fetch:")
    df, src = fetch_stock_data(ticker)
    print(f"  is_live : {src.is_live}")
    print(f"  method  : {src.method}")
    s = get_price_stats(df)
    print(f"  price   : {s['current_price']:,.2f}")
    print(f"  range   : {s['start_date']} → {s['end_date']}")