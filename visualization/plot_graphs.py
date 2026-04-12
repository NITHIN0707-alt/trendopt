"""
visualization/plot_graphs.py  —  TRENDOPT
============================================
All chart generation functions.

matplotlib charts (static PNG for export):
    plot_price_signals(df, signals, ticker)  → Figure
    plot_rsi(df, signals, ticker)            → Figure
    plot_macd(df, signals, ticker)           → Figure
    plot_algo_comparison(algo, ticker)       → Figure
    plot_equity_curve(bt, df, ticker)        → Figure

plotly charts (interactive, embedded in Streamlit):
    plotly_candlestick(df, signals, ticker)  → go.Figure
    plotly_prediction(df, preds, ticker)     → go.Figure
    plotly_rsi(df, signals, ticker)          → go.Figure
    plotly_algo_bar(algo, ticker)            → go.Figure
    plotly_equity(bt, df, ticker)            → go.Figure
    plotly_risk_radar(risk)                  → go.Figure
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional

# ── Dark palette ─────────────────────────────────────────────────────────
P = dict(
    bg="#0d1117", panel="#161b22", grid="#21262d", text="#e6edf3",
    muted="#8b949e", price="#58a6ff", ma50="#f0883e", ma200="#bc8cff",
    bb="#30363d", buy="#3fb950", sell="#f85149",
    rsi="#ffd700", vol="#00bcd4",
    pred=["#ff9800","#00bcd4","#e040fb","#ff5722"],
)

RC = {
    "figure.facecolor": P["bg"],  "axes.facecolor":  P["panel"],
    "axes.edgecolor":   P["grid"],"axes.labelcolor": P["text"],
    "xtick.color":      P["muted"],"ytick.color":    P["muted"],
    "grid.color":       P["grid"], "text.color":     P["text"],
    "legend.facecolor": P["panel"],"legend.edgecolor":P["grid"],
}

def _ax(ax):
    ax.grid(True, linestyle="--", lw=0.35, alpha=0.5, color=P["grid"])
    ax.set_facecolor(P["panel"])
    for sp in ax.spines.values(): sp.set_edgecolor(P["grid"])


# ══════════════════════════════════════════════════════════════════════════
# PLOTLY (interactive)
# ══════════════════════════════════════════════════════════════════════════

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", size=12),
    xaxis=dict(gridcolor="#21262d", showgrid=True),
    yaxis=dict(gridcolor="#21262d", showgrid=True),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
)


def plotly_candlestick(
    df:      pd.DataFrame,
    signals: dict,
    ticker:  str,
) -> go.Figure:
    """Interactive candlestick with volume, MA50, MA200, Bollinger Bands."""
    closes = df["Close"].values.astype(float)
    dates  = df.index

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=dates, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color=P["buy"],   increasing_fillcolor=P["buy"],
        decreasing_line_color=P["sell"],  decreasing_fillcolor=P["sell"],
    ), row=1, col=1)

    # Bollinger Bands
    ub, mb, lb = signals["bb_upper"], signals["bb_mid"], signals["bb_lower"]
    v = ~(np.isnan(ub) | np.isnan(lb))
    if v.any():
        fig.add_trace(go.Scatter(x=dates[v], y=ub[v], name="BB Upper",
            line=dict(color="rgba(48,54,61,0.9)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates[v], y=lb[v], name="BB Lower",
            line=dict(color="rgba(48,54,61,0.9)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(48,54,61,0.25)"), row=1, col=1)

    # MAs
    for key, colour, lbl in [
        ("sma50",  P["ma50"],  "MA50"),
        ("sma200", P["ma200"], "MA200"),
    ]:
        ma = signals[key]; vm = ~np.isnan(ma)
        if vm.any():
            fig.add_trace(go.Scatter(x=dates[vm], y=ma[vm], name=lbl,
                line=dict(color=colour, width=1.5, dash="dash")), row=1, col=1)

    # Buy/Sell markers
    bi, si = signals["buy_indices"], signals["sell_indices"]
    if len(bi):
        fig.add_trace(go.Scatter(
            x=dates[bi], y=closes[bi] * 0.985,
            mode="markers", name=f"BUY ({len(bi)})",
            marker=dict(symbol="triangle-up", size=10, color=P["buy"]),
        ), row=1, col=1)
    if len(si):
        fig.add_trace(go.Scatter(
            x=dates[si], y=closes[si] * 1.015,
            mode="markers", name=f"SELL ({len(si)})",
            marker=dict(symbol="triangle-down", size=10, color=P["sell"]),
        ), row=1, col=1)

    # Volume bars
    colours = [P["buy"] if df["Close"].iloc[i] >= df["Open"].iloc[i] else P["sell"]
               for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=dates, y=df["Volume"], name="Volume",
        marker_color=colours, opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{ticker} — Price History & Trading Signals",
                   font=dict(size=15)),
        xaxis_rangeslider_visible=False,
        height=560, **_PLOTLY_LAYOUT,
    )
    return fig


def plotly_prediction(
    df:    pd.DataFrame,
    preds: List[Dict],
    ticker: str,
    horizon: int = 30,
) -> go.Figure:
    """Historical close + multi-model future forecasts."""
    closes    = df["Close"].values.astype(float)
    dates     = df.index
    last      = dates[-1]
    max_h     = max((len(r["predictions"]) for r in preds if r.get("predictions")), default=0)
    fdates    = pd.bdate_range(start=last, periods=max_h + 1)[1:] if max_h else []

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=closes, name="Historical Close",
                             line=dict(color=P["price"], width=2)))
    fig.add_vline(x=str(last), line_dash="dash", line_color=P["grid"])

    for i, r in enumerate(preds):
        p = r.get("predictions", [])
        if not p: continue
        mae_str = f"  MAE=₹{r['mae']}" if r.get("mae") else ""
        fig.add_trace(go.Scatter(
            x=fdates[:len(p)], y=p,
            name=f"{r['model_name']}{mae_str}",
            line=dict(color=P["pred"][i % len(P["pred"])], width=2, dash="dash"),
        ))

    fig.update_layout(
        title=f"{ticker} — Historical vs Predicted Prices",
        height=420, **_PLOTLY_LAYOUT,
    )
    return fig


def plotly_rsi(df: pd.DataFrame, signals: dict, ticker: str) -> go.Figure:
    """RSI panel with overbought/oversold bands."""
    r     = signals["rsi"]
    dates = df.index
    v     = ~np.isnan(r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[v], y=r[v], name="RSI (14)",
                             line=dict(color=P["rsi"], width=1.5)))
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.12)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(63,185,80,0.12)", line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color=P["sell"], line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color=P["buy"],  line_width=1)
    fig.add_annotation(x=dates[-1], y=72, text="Overbought",
                       showarrow=False, font=dict(color=P["sell"], size=10))
    fig.add_annotation(x=dates[-1], y=28, text="Oversold",
                       showarrow=False, font=dict(color=P["buy"], size=10))
    fig.update_layout(title=f"{ticker} — RSI (14)", yaxis_range=[0, 100],
                      height=280, **_PLOTLY_LAYOUT)
    return fig


def plotly_macd(df: pd.DataFrame, signals: dict, ticker: str) -> go.Figure:
    """MACD with signal line and histogram."""
    dates = df.index
    ml, sl, hist = signals["macd_line"], signals["macd_sig"], signals["macd_hist"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=hist, name="Histogram",
                         marker_color=np.where(hist >= 0, P["buy"], P["sell"]),
                         opacity=0.6))
    fig.add_trace(go.Scatter(x=dates, y=ml, name="MACD",
                             line=dict(color=P["price"], width=1.5)))
    fig.add_trace(go.Scatter(x=dates, y=sl, name="Signal",
                             line=dict(color=P["ma50"], width=1.5, dash="dash")))
    fig.update_layout(title=f"{ticker} — MACD (12,26,9)",
                      height=280, **_PLOTLY_LAYOUT)
    return fig


def plotly_algo_bar(algo: dict, ticker: str) -> go.Figure:
    """Side-by-side profit and execution-time bars."""
    names   = ["Brute Force", "Greedy", "Dynamic Prog."]
    profits = [algo.get("brute_profit", 0), algo.get("greedy_profit", 0), algo.get("dp_profit", 0)]
    times   = [algo.get("brute_time", 0),   algo.get("greedy_time", 0),   algo.get("dp_time", 0)]
    colours = [P["sell"], P["buy"], P["price"]]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Max Profit (₹)", "Execution Time (s)"))
    for i, (n, p, c) in enumerate(zip(names, profits, colours)):
        fig.add_trace(go.Bar(x=[n], y=[p], name=n, marker_color=c,
                             showlegend=(i == 0)), row=1, col=1)
        fig.add_trace(go.Bar(x=[n], y=[times[i]], name=n, marker_color=c,
                             showlegend=False), row=1, col=2)

    fig.update_layout(title=f"{ticker} — DAA Algorithm Comparison",
                      barmode="group", height=360, **_PLOTLY_LAYOUT)
    return fig


def plotly_equity(bt: dict, df: pd.DataFrame, ticker: str) -> go.Figure:
    """Equity curve vs buy-and-hold benchmark."""
    eq     = bt.get("equity_curve", [bt["initial_cash"]])
    dates  = df.index[:len(eq)]
    closes = df["Close"].values[:len(eq)].astype(float)
    bah    = bt["initial_cash"] * closes / closes[0]   # buy-and-hold

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=eq, name="Strategy",
                             line=dict(color=P["buy"], width=2)))
    fig.add_trace(go.Scatter(x=dates, y=bah, name="Buy-and-Hold",
                             line=dict(color=P["muted"], width=1.5, dash="dash")))
    fig.add_hrect(y0=bt["initial_cash"] * 0.99, y1=bt["initial_cash"] * 1.01,
                  fillcolor="rgba(88,166,255,0.07)", line_width=0)
    fig.update_layout(title=f"{ticker} — Strategy Equity Curve vs Buy-and-Hold",
                      height=360, **_PLOTLY_LAYOUT)
    return fig


def plotly_risk_radar(risk: dict, ticker: str) -> go.Figure:
    """Radar chart of normalised risk metrics."""
    # Normalise each metric to [0, 1] for display
    vol_n  = min(risk.get("volatility", 0) / 80, 1.0)
    sh_n   = max(0, min(risk.get("sharpe_ratio", 0) / 3, 1.0))
    mdd_n  = min(risk.get("max_drawdown_pct", 0) / 50, 1.0)
    var_n  = min(risk.get("var_95_pct", 0) / 10, 1.0)
    er_n   = max(0, min(risk.get("expected_return_pct", 0) / 50, 1.0))

    cats   = ["Volatility", "Sharpe", "Max Drawdown", "VaR 95%", "Exp. Return"]
    values = [vol_n, sh_n, mdd_n, var_n, er_n]
    values += [values[0]]  # close polygon
    cats   += [cats[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=cats, fill="toself",
        fillcolor="rgba(88,166,255,0.2)",
        line=dict(color=P["price"], width=2),
        name=ticker,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=P["panel"],
            radialaxis=dict(visible=True, range=[0, 1], color=P["muted"]),
            angularaxis=dict(color=P["text"]),
        ),
        title=f"{ticker} — Risk Profile",
        height=350, **_PLOTLY_LAYOUT,
    )
    return fig


def plotly_volume_profile(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Horizontal volume-at-price distribution."""
    closes = df["Close"].values.astype(float)
    vols   = df["Volume"].values.astype(float)
    bins   = np.linspace(closes.min(), closes.max(), 40)
    idx    = np.digitize(closes, bins) - 1
    vol_at_price = np.zeros(len(bins))
    for i, v in zip(idx, vols):
        if 0 <= i < len(bins):
            vol_at_price[i] += v

    fig = go.Figure(go.Bar(
        x=vol_at_price, y=bins, orientation="h",
        marker_color=P["vol"], opacity=0.7, name="Volume",
    ))
    fig.update_layout(title=f"{ticker} — Volume Profile",
                      height=420, **_PLOTLY_LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# MATPLOTLIB (static export)
# ══════════════════════════════════════════════════════════════════════════

def save_dashboard_png(
    df: pd.DataFrame,
    signals: dict,
    preds: List[Dict],
    bt: dict,
    algo: dict,
    rec: dict,
    ticker: str,
    output_dir: str = "output",
) -> str:
    """
    Render a 3-row static dashboard PNG and save it to ``output_dir``.

    Returns the file path.
    """
    import os; os.makedirs(output_dir, exist_ok=True)
    closes = df["Close"].values.astype(float)
    dates  = df.index

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(18, 14))
        fig.patch.set_facecolor(P["bg"])
        rec_c = {"BUY": P["buy"], "SELL": P["sell"], "HOLD": P["rsi"]}.get(
            rec["recommendation"], P["text"])
        fig.suptitle(
            f"TRENDOPT  ·  {ticker}  ·  {rec['recommendation']}  ({rec['confidence']}% confidence)",
            fontsize=17, fontweight="bold", color=rec_c, y=0.99,
        )

        gs = fig.add_gridspec(3, 2, hspace=0.48, wspace=0.3,
                              height_ratios=[3, 1.5, 2.5])
        ax_p  = fig.add_subplot(gs[0, :])
        ax_r  = fig.add_subplot(gs[1, :])
        ax_pr = fig.add_subplot(gs[2, 0])
        ax_al = fig.add_subplot(gs[2, 1])

        # ── Price + MA + BB + signals ────────────────────────────────────
        ub, lb = signals["bb_upper"], signals["bb_lower"]
        vv = ~(np.isnan(ub) | np.isnan(lb))
        if vv.any():
            ax_p.fill_between(dates[vv], ub[vv], lb[vv], color=P["bb"], alpha=0.35)
        ax_p.plot(dates, closes, color=P["price"], lw=1.4, label="Close")
        for key, col, lbl in [("sma50", P["ma50"], "MA50"), ("sma200", P["ma200"], "MA200")]:
            ma = signals[key]; vm = ~np.isnan(ma)
            if vm.any():
                ax_p.plot(dates[vm], ma[vm], color=col, lw=1.1, linestyle="--", label=lbl)
        bi, si = signals["buy_indices"], signals["sell_indices"]
        if len(bi): ax_p.scatter(dates[bi], closes[bi], marker="^", s=60,
                                 color=P["buy"],  zorder=6, label=f"BUY ({len(bi)})")
        if len(si): ax_p.scatter(dates[si], closes[si], marker="v", s=60,
                                 color=P["sell"], zorder=6, label=f"SELL ({len(si)})")
        ax_p.set_title("Price + MA50/MA200 + Bollinger Bands + Signals", fontsize=11, fontweight="bold")
        ax_p.set_ylabel("Price"); ax_p.legend(fontsize=7, loc="upper left")
        ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.setp(ax_p.get_xticklabels(), rotation=20); _ax(ax_p)

        # ── RSI ──────────────────────────────────────────────────────────
        r = signals["rsi"]; vr = ~np.isnan(r)
        ax_r.plot(dates[vr], r[vr], color=P["rsi"], lw=1.2)
        ax_r.axhline(70, color=P["sell"], lw=0.8, linestyle="--")
        ax_r.axhline(30, color=P["buy"],  lw=0.8, linestyle="--")
        ax_r.fill_between(dates[vr], r[vr], 70, where=(r[vr] > 70), color=P["sell"], alpha=0.2)
        ax_r.fill_between(dates[vr], r[vr], 30, where=(r[vr] < 30), color=P["buy"],  alpha=0.2)
        ax_r.set_ylim(0, 100); ax_r.set_ylabel("RSI")
        ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.setp(ax_r.get_xticklabels(), rotation=20); _ax(ax_r)

        # ── Predictions ──────────────────────────────────────────────────
        last   = dates[-1]
        max_h  = max((len(r["predictions"]) for r in preds if r.get("predictions")), default=0)
        fdates = pd.bdate_range(start=last, periods=max_h + 1)[1:] if max_h else []
        ax_pr.plot(dates[-60:], closes[-60:], color=P["price"], lw=1.3)
        ax_pr.axvline(x=last, color=P["grid"], linestyle="--", lw=0.8)
        for i, r in enumerate(preds):
            p = r.get("predictions", [])
            if not p: continue
            ax_pr.plot(fdates[:len(p)], p, color=P["pred"][i % len(P["pred"])],
                       lw=1.6, linestyle="--", label=r["model_name"])
        ax_pr.set_title("Price Forecast", fontsize=11, fontweight="bold")
        ax_pr.legend(fontsize=7); ax_pr.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.setp(ax_pr.get_xticklabels(), rotation=20); _ax(ax_pr)

        # ── Algo comparison ──────────────────────────────────────────────
        names   = ["Brute Force", "Greedy", "Dynamic Prog."]
        profits = [algo.get("brute_profit", 0), algo.get("greedy_profit", 0), algo.get("dp_profit", 0)]
        cols    = [P["sell"], P["buy"], P["price"]]
        bars    = ax_al.bar(names, profits, color=cols, width=0.5, edgecolor=P["grid"])
        for b, v in zip(bars, profits):
            ax_al.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                       f"₹{v:,.0f}", ha="center", va="bottom", fontsize=8, color=P["text"])
        ax_al.set_title("Algorithm Profit Comparison", fontsize=11, fontweight="bold")
        ax_al.set_ylabel("Profit (₹)"); _ax(ax_al)

        path = os.path.join(output_dir, f"{ticker}_dashboard.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return path
