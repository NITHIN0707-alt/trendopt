"""
database/db_manager.py  —  TRENDOPT
======================================
SQLite persistence layer.

Tables
------
search_history   — ticker, timestamp, live_data flag
analysis_results — ticker, timestamp, price, recommendation, confidence,
                   profit_greedy, sharpe, volatility, pred_7d, pred_30d
predictions      — ticker, timestamp, model, horizon, pred_json

Public API
----------
init_db()
log_search(ticker, is_live)
save_analysis(ticker, stats, rec, risk, algo)
get_history(limit)
get_analyses(ticker, limit)
export_csv(ticker) → str (path)
"""

import sqlite3
import json
import os
import pandas as pd
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trendopt.db")


def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db() -> None:
    """Create all tables if they don't already exist."""
    with _conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS search_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker    TEXT    NOT NULL,
            timestamp TEXT    NOT NULL,
            is_live   INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS analysis_results (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker        TEXT    NOT NULL,
            timestamp     TEXT    NOT NULL,
            current_price REAL,
            recommendation TEXT,
            confidence    INTEGER,
            score         REAL,
            profit_greedy REAL,
            profit_dp     REAL,
            sharpe        REAL,
            volatility    REAL,
            pred_7d       REAL,
            pred_30d      REAL,
            change_30d    REAL,
            notes         TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker    TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            model     TEXT NOT NULL,
            horizon   INTEGER,
            pred_json TEXT
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker    TEXT NOT NULL UNIQUE,
            added_at  TEXT NOT NULL,
            notes     TEXT
        );
        """)


def log_search(ticker: str, is_live: bool) -> None:
    """Record every user stock search."""
    with _conn() as con:
        con.execute(
            "INSERT INTO search_history (ticker, timestamp, is_live) VALUES (?,?,?)",
            (ticker, datetime.now().isoformat(timespec="seconds"), int(is_live)),
        )


def save_analysis(
    ticker: str,
    stats:  dict,
    rec:    dict,
    risk:   dict,
    algo:   dict,
) -> None:
    """Persist a full analysis result."""
    with _conn() as con:
        con.execute("""
            INSERT INTO analysis_results
            (ticker, timestamp, current_price, recommendation, confidence, score,
             profit_greedy, profit_dp, sharpe, volatility, pred_7d, pred_30d, change_30d)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ticker,
            datetime.now().isoformat(timespec="seconds"),
            stats.get("current_price"),
            rec.get("recommendation"),
            rec.get("confidence"),
            rec.get("score"),
            algo.get("greedy_profit"),
            algo.get("dp_profit"),
            risk.get("sharpe_ratio"),
            risk.get("volatility"),
            rec.get("pred_7d"),
            rec.get("pred_30d"),
            rec.get("change_30d_pct"),
        ))


def save_predictions(ticker: str, pred_results: list, horizon: int) -> None:
    """Persist ML model predictions as JSON."""
    with _conn() as con:
        ts = datetime.now().isoformat(timespec="seconds")
        for r in pred_results:
            if r.get("predictions"):
                con.execute(
                    "INSERT INTO predictions (ticker, timestamp, model, horizon, pred_json) VALUES (?,?,?,?,?)",
                    (ticker, ts, r["model_name"], horizon, json.dumps(r["predictions"])),
                )


def get_history(limit: int = 50) -> pd.DataFrame:
    """Return the most-recent search history."""
    with _conn() as con:
        return pd.read_sql(
            "SELECT ticker, timestamp, CASE is_live WHEN 1 THEN 'Live' ELSE 'Synthetic' END as source "
            "FROM search_history ORDER BY id DESC LIMIT ?",
            con, params=(limit,)
        )


def get_analyses(ticker: Optional[str] = None, limit: int = 30) -> pd.DataFrame:
    """Return stored analysis results, optionally filtered by ticker."""
    q = "SELECT ticker, timestamp, current_price, recommendation, confidence, " \
        "profit_greedy, sharpe, volatility, pred_30d, change_30d " \
        "FROM analysis_results"
    params: tuple = ()
    if ticker:
        q     += " WHERE ticker = ?"
        params = (ticker,)
    q += " ORDER BY id DESC LIMIT ?"
    params = params + (limit,)
    with _conn() as con:
        return pd.read_sql(q, con, params=params)


def add_to_portfolio(ticker: str, notes: str = "") -> None:
    """Add a ticker to the portfolio watchlist."""
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO portfolio (ticker, added_at, notes) VALUES (?,?,?)",
            (ticker, datetime.now().isoformat(timespec="seconds"), notes),
        )


def get_portfolio() -> list:
    """Return all portfolio tickers."""
    with _conn() as con:
        rows = con.execute("SELECT ticker FROM portfolio ORDER BY added_at").fetchall()
    return [r[0] for r in rows]


def remove_from_portfolio(ticker: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker,))


def export_csv(ticker: str, output_dir: str = ".") -> str:
    """Export all analyses for a ticker to CSV. Returns the file path."""
    df = get_analyses(ticker, limit=1000)
    path = os.path.join(output_dir, f"{ticker}_analysis_history.csv")
    df.to_csv(path, index=False)
    return path


# Initialise on import
init_db()
