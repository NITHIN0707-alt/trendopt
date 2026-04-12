"""
visualize.py — TRENDOPT
All matplotlib-based visualisation routines.

Charts produced:
    1. Historical closing price with buy (▲ green) and sell (▼ red) signals.
    2. Historical vs predicted price trend (multi-model overlay).
    3. Algorithm performance comparison bar chart (profit + execution time).
    4. Combined dashboard (2×2 subplot layout).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from typing import Dict, List, Optional
import os


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

PALETTE = {
    "price":      "#2196F3",   # blue
    "buy":        "#4CAF50",   # green
    "sell":       "#F44336",   # red
    "lr":         "#FF9800",   # orange  — Linear Regression
    "rf":         "#9C27B0",   # purple  — Random Forest
    "lstm":       "#00BCD4",   # cyan    — LSTM
    "bg":         "#0D1117",   # dark background
    "grid":       "#21262D",   # subtle grid
    "text":       "#E6EDF3",   # light text
}

STYLE = {
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "text.color":        PALETTE["text"],
    "legend.facecolor":  "#161B22",
    "legend.edgecolor":  PALETTE["grid"],
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _apply_style(ax):
    """Apply dark-mode style to an Axes object."""
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, color=PALETTE["grid"])
    ax.set_facecolor(PALETTE["bg"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])


# ---------------------------------------------------------------------------
# 1. Historical price + signals
# ---------------------------------------------------------------------------

def plot_price_with_signals(
    df: pd.DataFrame,
    buy_indices: np.ndarray,
    sell_indices: np.ndarray,
    ticker: str = "Stock",
    save_path: Optional[str] = None,
) -> str:
    """
    Plot historical closing price annotated with buy/sell signals.

    Args:
        df           : OHLCV DataFrame indexed by date.
        buy_indices  : Integer positions of BUY signals.
        sell_indices : Integer positions of SELL signals.
        ticker       : Stock ticker label.
        save_path    : File path to save the figure (PNG). Auto-generated if None.

    Returns:
        Path where the figure was saved.
    """
    _ensure_output_dir()
    save_path = save_path or os.path.join(OUTPUT_DIR, f"{ticker}_signals.png")

    closes = df["Close"].values.astype(float)
    dates  = df.index

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor(PALETTE["bg"])

        # Price line
        ax.plot(dates, closes, color=PALETTE["price"], linewidth=1.4,
                label="Close Price", zorder=2)

        # Buy signals
        if len(buy_indices) > 0:
            ax.scatter(
                dates[buy_indices], closes[buy_indices],
                marker="^", s=90, color=PALETTE["buy"], zorder=5,
                label=f"BUY ({len(buy_indices)})",
            )

        # Sell signals
        if len(sell_indices) > 0:
            ax.scatter(
                dates[sell_indices], closes[sell_indices],
                marker="v", s=90, color=PALETTE["sell"], zorder=5,
                label=f"SELL ({len(sell_indices)})",
            )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=30)

        ax.set_title(f"{ticker} — Historical Price & Trading Signals",
                     fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.legend(loc="upper left", fontsize=9)
        _apply_style(ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[Visualize] Signals chart saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# 2. Historical vs predicted prices
# ---------------------------------------------------------------------------

def plot_predictions(
    df: pd.DataFrame,
    prediction_results: List[Dict],
    ticker: str = "Stock",
    save_path: Optional[str] = None,
) -> str:
    """
    Overlay historical prices with multi-model future predictions.

    Args:
        df                  : OHLCV DataFrame.
        prediction_results  : Output list from predictor.run_all_predictions().
        ticker              : Stock ticker label.
        save_path           : Output file path.

    Returns:
        Path where the figure was saved.
    """
    _ensure_output_dir()
    save_path = save_path or os.path.join(OUTPUT_DIR, f"{ticker}_predictions.png")

    closes = df["Close"].values.astype(float)
    dates  = df.index

    # Build future date axis
    last_date    = dates[-1]
    max_horizon  = max(len(r["predictions"]) for r in prediction_results if r["predictions"])
    future_dates = pd.bdate_range(start=last_date, periods=max_horizon + 1)[1:]

    colour_map = {"Linear Regression": PALETTE["lr"],
                  "Random Forest":     PALETTE["rf"],
                  "LSTM":              PALETTE["lstm"]}

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor(PALETTE["bg"])

        # Historical
        ax.plot(dates, closes, color=PALETTE["price"], linewidth=1.4,
                label="Historical Close", zorder=2)

        # Vertical divider at last historical date
        ax.axvline(x=last_date, color=PALETTE["grid"], linestyle="--",
                   linewidth=1, label="Prediction starts")

        # Predictions per model
        for res in prediction_results:
            preds = res.get("predictions", [])
            if not preds:
                continue
            name   = res["model_name"]
            colour = colour_map.get(name, "#FFFFFF")
            fd     = future_dates[:len(preds)]
            ax.plot(fd, preds, color=colour, linewidth=1.6, linestyle="--",
                    label=f"{name} (MAE={res['mae']})", zorder=3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=30)

        ax.set_title(f"{ticker} — Historical vs Predicted Prices",
                     fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.legend(loc="upper left", fontsize=9)
        _apply_style(ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[Visualize] Prediction chart saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# 3. Algorithm performance comparison
# ---------------------------------------------------------------------------

def plot_performance_comparison(
    comparison: List[Dict],
    ticker: str = "Stock",
    save_path: Optional[str] = None,
) -> str:
    """
    Bar chart comparing profit and execution time across algorithms.

    Args:
        comparison : List of dicts with keys: algorithm, profit, time_sec.
        ticker     : Stock ticker label.
        save_path  : Output file path.

    Returns:
        Path where the figure was saved.
    """
    _ensure_output_dir()
    save_path = save_path or os.path.join(OUTPUT_DIR, f"{ticker}_comparison.png")

    names   = [d["algorithm"]  for d in comparison]
    profits = [d["profit"]     for d in comparison]
    times   = [d["time_sec"]   for d in comparison]

    bar_colours = ["#F44336", "#4CAF50", "#2196F3"]   # BF, Greedy, DP

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor(PALETTE["bg"])
        fig.suptitle(f"{ticker} — Algorithm Performance Comparison",
                     fontsize=14, fontweight="bold", color=PALETTE["text"])

        # Profit chart
        bars1 = ax1.bar(names, profits, color=bar_colours, width=0.5,
                        edgecolor=PALETTE["grid"], linewidth=0.8)
        ax1.set_title("Max Profit (₹)", fontsize=11)
        ax1.set_ylabel("Profit (₹)")
        for bar, val in zip(bars1, profits):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"₹{val:.2f}", ha="center", va="bottom",
                     fontsize=9, color=PALETTE["text"])
        _apply_style(ax1)

        # Time chart (log scale for big BF times)
        bars2 = ax2.bar(names, times, color=bar_colours, width=0.5,
                        edgecolor=PALETTE["grid"], linewidth=0.8)
        ax2.set_title("Execution Time (s)", fontsize=11)
        ax2.set_ylabel("Time (seconds)")
        ax2.set_yscale("log")
        for bar, val in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                     f"{val:.4f}s", ha="center", va="bottom",
                     fontsize=9, color=PALETTE["text"])
        _apply_style(ax2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[Visualize] Performance chart saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# 4. Combined 2×2 dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(
    df: pd.DataFrame,
    buy_indices: np.ndarray,
    sell_indices: np.ndarray,
    prediction_results: List[Dict],
    comparison: List[Dict],
    ticker: str = "Stock",
    save_path: Optional[str] = None,
) -> str:
    """
    Render a 2×2 dashboard combining all charts into one figure.

    Args:
        df                  : OHLCV DataFrame.
        buy_indices         : BUY signal positions.
        sell_indices        : SELL signal positions.
        prediction_results  : ML prediction outputs.
        comparison          : Algorithm comparison list.
        ticker              : Stock ticker label.
        save_path           : Output file path.

    Returns:
        Path where the figure was saved.
    """
    _ensure_output_dir()
    save_path = save_path or os.path.join(OUTPUT_DIR, f"{ticker}_dashboard.png")

    closes = df["Close"].values.astype(float)
    dates  = df.index

    last_date   = dates[-1]
    max_horizon = max((len(r["predictions"]) for r in prediction_results if r["predictions"]), default=0)
    future_dates = pd.bdate_range(start=last_date, periods=max_horizon + 1)[1:] if max_horizon else []

    colour_map = {"Linear Regression": PALETTE["lr"],
                  "Random Forest":     PALETTE["rf"],
                  "LSTM":              PALETTE["lstm"]}
    bar_colours = ["#F44336", "#4CAF50", "#2196F3"]

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(18, 11))
        fig.patch.set_facecolor(PALETTE["bg"])
        fig.suptitle(f"TRENDOPT Dashboard — {ticker}",
                     fontsize=17, fontweight="bold", color=PALETTE["text"], y=0.98)

        gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
        ax_sig  = fig.add_subplot(gs[0, :])          # top row — full width
        ax_pred = fig.add_subplot(gs[1, 0])           # bottom left
        ax_bar1 = fig.add_subplot(gs[1, 1])           # bottom right

        # ── Top: signals ──────────────────────────────────────────────────
        ax_sig.plot(dates, closes, color=PALETTE["price"], linewidth=1.2,
                    label="Close Price", zorder=2)
        if len(buy_indices):
            ax_sig.scatter(dates[buy_indices], closes[buy_indices],
                           marker="^", s=70, color=PALETTE["buy"], zorder=5,
                           label=f"BUY ({len(buy_indices)})")
        if len(sell_indices):
            ax_sig.scatter(dates[sell_indices], closes[sell_indices],
                           marker="v", s=70, color=PALETTE["sell"], zorder=5,
                           label=f"SELL ({len(sell_indices)})")
        ax_sig.set_title("Historical Price & Trading Signals", fontsize=11, fontweight="bold")
        ax_sig.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax_sig.xaxis.set_major_locator(mdates.MonthLocator())
        ax_sig.tick_params(axis="x", rotation=20)
        ax_sig.set_ylabel("Price (₹)")
        ax_sig.legend(loc="upper left", fontsize=8)
        _apply_style(ax_sig)

        # ── Bottom left: predictions ───────────────────────────────────────
        ax_pred.plot(dates[-60:], closes[-60:], color=PALETTE["price"],
                     linewidth=1.2, label="Historical")
        ax_pred.axvline(x=last_date, color=PALETTE["grid"], linestyle="--", linewidth=0.8)
        for res in prediction_results:
            preds = res.get("predictions", [])
            if not preds:
                continue
            name   = res["model_name"]
            colour = colour_map.get(name, "#FFFFFF")
            fd     = future_dates[:len(preds)]
            ax_pred.plot(fd, preds, color=colour, linewidth=1.5,
                         linestyle="--", label=f"{name}")
        ax_pred.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax_pred.xaxis.set_major_locator(mdates.MonthLocator())
        ax_pred.tick_params(axis="x", rotation=20)
        ax_pred.set_title("Price Prediction", fontsize=11, fontweight="bold")
        ax_pred.set_ylabel("Price (₹)")
        ax_pred.legend(loc="upper left", fontsize=7)
        _apply_style(ax_pred)

        # ── Bottom right: algorithm profit bars ───────────────────────────
        if comparison:
            names   = [d["algorithm"] for d in comparison]
            profits = [d["profit"]    for d in comparison]
            bars = ax_bar1.bar(names, profits, color=bar_colours[:len(names)],
                               width=0.5, edgecolor=PALETTE["grid"], linewidth=0.8)
            for bar, val in zip(bars, profits):
                ax_bar1.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.5, f"₹{val:.0f}",
                             ha="center", va="bottom", fontsize=8, color=PALETTE["text"])
            ax_bar1.set_title("Algorithm Profit Comparison", fontsize=11, fontweight="bold")
            ax_bar1.set_ylabel("Profit (₹)")
            _apply_style(ax_bar1)

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[Visualize] Dashboard saved → {save_path}")
    return save_path
