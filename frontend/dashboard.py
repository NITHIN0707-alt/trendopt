"""
frontend/dashboard.py  —  TRENDOPT
=====================================
Full interactive Streamlit web dashboard.

Run:
    streamlit run frontend/dashboard.py

Tabs
----
1. 📈 Price & Signals   — candlestick + MA + BB + buy/sell markers
2. 🔮 Predictions       — ML model forecasts chart + accuracy table
3. ⚙️  Algorithms        — DAA comparison with complexity table
4. 📡 Indicators        — RSI + MACD + Bollinger + Volume Profile
5. 🧪 Backtesting       — equity curve + trade log
6. 🎯 Recommendation    — BUY/SELL/HOLD with reasoning + radar
7. 🗂️  Portfolio         — multi-stock watchlist
8. 🕘 History           — search log + past analyses
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from backend.data_fetcher           import resolve_ticker, fetch_stock_data, get_price_stats
from backend.brute                  import brute_force
from backend.greedy                 import greedy
from backend.dp                     import dp_cooldown
from backend.signal_detector        import detect_signals, latest_signal, active_alerts
from backend.prediction_model       import run_all_predictions, consensus_forecast
from backend.backtesting            import backtest
from backend.risk_analysis          import compute_all_risk
from backend.recommendation_engine  import generate_recommendation
from database.db_manager            import (
    log_search, save_analysis, save_predictions,
    get_history, get_analyses,
    add_to_portfolio, get_portfolio, remove_from_portfolio,
    export_csv,
)
from visualization.plot_graphs import (
    plotly_candlestick, plotly_prediction, plotly_rsi, plotly_macd,
    plotly_algo_bar, plotly_equity, plotly_risk_radar, plotly_volume_profile,
    save_dashboard_png,
)


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TRENDOPT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; border-right:1px solid #30363d; }
[data-testid="stHeader"]           { background: transparent; }
[data-testid="stMetric"]           { background: #161b22; border:1px solid #30363d;
                                     border-radius:12px; padding:12px 16px; }
.metric-card {
    background:#161b22; border:1px solid #30363d; border-radius:12px;
    padding:16px 20px; text-align:center; margin-bottom:10px;
}
.mlbl { font-size:12px; color:#8b949e; margin-bottom:4px; }
.mval { font-size:26px; font-weight:600; color:#e6edf3; }
.msub { font-size:12px; margin-top:4px; }

.rec-badge { display:inline-block; padding:10px 32px; border-radius:50px;
             font-size:24px; font-weight:700; letter-spacing:3px; }
.rec-buy   { background:#1a4731; color:#3fb950; border:2px solid #3fb950; }
.rec-sell  { background:#3d1a1a; color:#f85149; border:2px solid #f85149; }
.rec-hold  { background:#3d3010; color:#ffd700; border:2px solid #ffd700; }

.reason-item {
    background:#161b22; border-left:3px solid #58a6ff;
    border-radius:0 8px 8px 0; padding:9px 14px;
    margin-bottom:6px; font-size:13px; color:#c9d1d9;
}
.algo-table { width:100%; border-collapse:collapse; font-size:13px; }
.algo-table th { background:#161b22; color:#8b949e; padding:8px 12px;
                 border-bottom:1px solid #30363d; text-align:left; }
.algo-table td { padding:8px 12px; border-bottom:1px solid #21262d; color:#e6edf3; }
.algo-table tr:hover td { background:#1c2128; }

.alert-box { background:#161b22; border:1px solid #30363d; border-radius:8px;
             padding:10px 14px; margin-bottom:6px; font-size:13px; }
.section-hdr { font-size:15px; font-weight:600; color:#58a6ff;
               border-bottom:1px solid #30363d; padding-bottom:6px;
               margin-bottom:14px; }
.conf-bg { background:#21262d; border-radius:50px; height:10px; margin-top:6px; }
.conf-fg { height:10px; border-radius:50px; }
</style>
""", unsafe_allow_html=True)


# ── Colours ────────────────────────────────────────────────────────────────
REC_COLOUR = {"BUY": "#3fb950", "SELL": "#f85149", "HOLD": "#ffd700"}


# ── Helper: metric card HTML ───────────────────────────────────────────────
def mcard(label, value, sub="", sub_colour="#8b949e"):
    return (f'<div class="metric-card"><div class="mlbl">{label}</div>'
            f'<div class="mval">{value}</div>'
            f'<div class="msub" style="color:{sub_colour}">{sub}</div></div>')


# ── Caching ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_and_analyse(ticker: str, horizon: int, strategy: str):
    df, src = fetch_stock_data(ticker)
    is_live = src.is_live
    fetch_method = src.method
    fetch_error  = src.error
    stats        = get_price_stats(df)
    closes       = df["Close"].values.astype(float)

    # ── Algorithms ────────────────────────────────────────────────────────
    algo = {}
    for name, fn, kwargs in [
        ("brute", brute_force,  {"max_days": 22}),
        ("greedy", greedy,       {}),
        ("dp",     dp_cooldown,  {}),
    ]:
        t0          = time.perf_counter()
        res         = fn(closes, **kwargs)
        dt          = time.perf_counter() - t0
        algo[f"{name}_profit"] = res["profit"]
        algo[f"{name}_time"]   = round(dt, 6)

    # ── Signals ───────────────────────────────────────────────────────────
    signals  = detect_signals(df, strategy=strategy)
    sig_str  = latest_signal(signals, closes)
    alerts   = active_alerts(signals, closes, ticker)

    # ── ML Predictions ────────────────────────────────────────────────────
    lookback    = min(30, len(closes) // 4)
    pred_results = run_all_predictions(closes, horizon=horizon, lookback=lookback)
    consensus   = consensus_forecast(pred_results, horizon)

    # ── Backtesting ───────────────────────────────────────────────────────
    bt = backtest(df, signals)

    # ── Risk ──────────────────────────────────────────────────────────────
    risk = compute_all_risk(closes)

    # ── Recommendation ────────────────────────────────────────────────────
    rec = generate_recommendation(
        stats["current_price"], pred_results, signals, sig_str, risk
    )

    return {
        "df": df, "closes": closes, "stats": stats, "is_live": is_live,
        "fetch_method": src.method,
        "fetch_error":  src.error,
        "algo": algo, "signals": signals, "sig_str": sig_str, "alerts": alerts,
        "pred_results": pred_results, "consensus": consensus,
        "bt": bt, "risk": risk, "rec": rec,
    }


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 TRENDOPT")
    st.markdown("*Intelligent Stock Analyzer*")
    st.divider()

    query = st.text_input("🔍 Stock ticker / name", value="RELIANCE",
                          placeholder="e.g. RELIANCE, TCS, AAPL, MSFT")

    st.markdown("**Quick picks**")
    quick = st.pills("", ["RELIANCE", "TCS", "INFY", "AAPL", "MSFT", "TSLA", "NVDA"],
                     default=None, key="qpick")
    if quick:
        query = quick

    st.divider()
    horizon  = st.slider("Prediction horizon (days)", 7, 60, 30)
    strategy = st.selectbox(
        "Signal strategy",
        ["combined", "local_extrema", "rsi", "ma_crossover", "bb", "macd"],
    )
    show_bb    = st.toggle("Bollinger Bands", value=True)
    show_ma50  = st.toggle("MA 50", value=True)
    show_ma200 = st.toggle("MA 200", value=True)

    st.divider()
    analyze_btn = st.button("🚀 Analyze", use_container_width=True, type="primary")

    st.divider()
    if st.button("💾 Add to Portfolio", use_container_width=True):
        t = resolve_ticker(query)
        add_to_portfolio(t)
        st.success(f"{t} added to portfolio!")

    st.markdown("---")
    st.caption("Data: yfinance  ·  ML: scikit-learn")
    st.caption("⚠️ For educational use only.")


# ══════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════
ticker = resolve_ticker(query or "RELIANCE")
st.markdown(f"# 📊 {ticker}")

if not analyze_btn and "last_run" not in st.session_state:
    st.info("👈 Enter a stock ticker in the sidebar and click **🚀 Analyze**.")
    st.stop()

with st.spinner(f"Analyzing {ticker} — this may take 10–30 seconds…"):
    try:
        d = load_and_analyse(ticker, horizon, strategy)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

st.session_state["last_run"] = ticker

# Persist to DB (async-friendly: outside cache)
log_search(ticker, d["is_live"])
save_analysis(ticker, d["stats"], d["rec"], d["risk"], d["algo"])
save_predictions(ticker, d["pred_results"], horizon)

df       = d["df"]
closes   = d["closes"]
stats    = d["stats"]
signals  = d["signals"]
preds    = d["pred_results"]
rec      = d["rec"]
algo     = d["algo"]
bt       = d["bt"]
risk     = d["risk"]
alerts   = d["alerts"]
dates    = df.index

# ── Data source badge + alerts ─────────────────────────────────────────────
live_lbl = "🟢 Live data" if d["is_live"] else "🔴 SYNTHETIC DATA (prices are NOT real)"
st.caption(f"{live_lbl}  ·  {stats['start_date']} → {stats['end_date']}  " +
           f"·  {stats['trading_days']} trading days")

# Synthetic data warning banner
if not d["is_live"]:
    st.error(
        "⚠️ **SYNTHETIC DATA — Prices are NOT real for " + ticker + "**"
    )
    st.warning(
        """**How to fix this in 2 steps:**

**Step 1** — Open a terminal and run:
```
pip install --upgrade yfinance requests
```

**Step 2** — Stop Streamlit (Ctrl+C) and restart:
```
streamlit run frontend/dashboard.py
```

If the problem persists, your network/firewall may be blocking Yahoo Finance.
Try connecting to a different Wi-Fi or disabling VPN/proxy."""
    )

if alerts:
    with st.expander(f"🔔 {len(alerts)} Active Alert(s)", expanded=False):
        for a in alerts:
            st.markdown(f'<div class="alert-box">{a}</div>', unsafe_allow_html=True)

# ── Top metric row ─────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
chg     = stats["change_pct"]
chg_col = "#3fb950" if chg >= 0 else "#f85149"
chg_sym = "▲" if chg >= 0 else "▼"

c1.markdown(mcard("Current Price", f"₹{stats['current_price']:,.2f}",
                  f"{chg_sym} {abs(chg):.2f}% today", chg_col), unsafe_allow_html=True)
c2.markdown(mcard("52W High",  f"₹{stats['high_52w']:,.2f}"), unsafe_allow_html=True)
c3.markdown(mcard("52W Low",   f"₹{stats['low_52w']:,.2f}"),  unsafe_allow_html=True)
c4.markdown(mcard("Volatility", f"{risk['volatility']:.1f}%",
                  risk["risk_level"],
                  "#f85149" if risk["volatility"] > 35 else "#ffd700" if risk["volatility"] > 20 else "#3fb950"),
            unsafe_allow_html=True)
sh = risk["sharpe_ratio"]
c5.markdown(mcard("Sharpe Ratio", f"{sh:.2f}",
                  "Excellent" if sh > 2 else "Good" if sh > 1 else "Fair" if sh > 0 else "Poor",
                  "#3fb950" if sh > 1 else "#ffd700" if sh > 0 else "#f85149"),
            unsafe_allow_html=True)
c6.markdown(mcard("Recommendation",
                  rec["recommendation"],
                  f"Confidence {rec['confidence']}%",
                  REC_COLOUR[rec["recommendation"]]),
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈 Price & Signals", "🔮 Predictions", "⚙️ Algorithms",
    "📡 Indicators", "🧪 Backtesting", "🎯 Recommendation",
    "🗂️ Portfolio", "🕘 History",
])


# ╔═══════════════════════════════════════╗
# ║  TAB 1 — Price & Signals             ║
# ╚═══════════════════════════════════════╝
with tabs[0]:
    sig_override = signals.copy()
    # Respect sidebar toggles
    if not show_bb:
        sig_override["bb_upper"] = np.full_like(signals["bb_upper"], np.nan)
        sig_override["bb_lower"] = np.full_like(signals["bb_lower"], np.nan)
    if not show_ma50:
        sig_override["sma50"]  = np.full_like(signals["sma50"], np.nan)
    if not show_ma200:
        sig_override["sma200"] = np.full_like(signals["sma200"], np.nan)

    st.plotly_chart(plotly_candlestick(df, sig_override, ticker),
                    use_container_width=True)

    # Volume profile side-by-side
    col_vp, col_stat = st.columns([2, 1])
    with col_vp:
        st.plotly_chart(plotly_volume_profile(df, ticker), use_container_width=True)
    with col_stat:
        st.markdown('<div class="section-hdr">Signal Summary</div>', unsafe_allow_html=True)
        st.metric("BUY signals",  len(signals["buy_indices"]))
        st.metric("SELL signals", len(signals["sell_indices"]))
        st.metric("Strategy",     strategy)
        st.metric("Latest signal", d["sig_str"])
        rsi_now = float(signals["rsi"][-1]) if not np.isnan(signals["rsi"][-1]) else 50.0
        st.metric("RSI now", f"{rsi_now:.1f}")


# ╔═══════════════════════════════════════╗
# ║  TAB 2 — Predictions                 ║
# ╚═══════════════════════════════════════╝
with tabs[1]:
    st.plotly_chart(plotly_prediction(df, preds, ticker, horizon),
                    use_container_width=True)

    st.markdown('<div class="section-hdr">Model Accuracy (held-out test set)</div>',
                unsafe_allow_html=True)
    rows = []
    for r in preds:
        if r.get("predictions"):
            rows.append({
                "Model":  r["model_name"],
                "MAE":    f"₹{r['mae']:,.2f}",
                "RMSE":   f"₹{r['rmse']:,.2f}",
                "R²":     f"{r['r2']:.4f}",
                "MAPE":   f"{r['mape']:.2f}%",
                f"7-Day (₹)":  f"{r['predictions'][6]:,.2f}"  if len(r['predictions']) > 6  else "—",
                f"30-Day (₹)": f"{r['predictions'][29]:,.2f}" if len(r['predictions']) > 29 else "—",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Download predictions CSV
    if preds and preds[0].get("predictions"):
        p0   = preds[0]["predictions"]
        fdates = pd.bdate_range(start=dates[-1], periods=len(p0) + 1)[1:]
        csv_df = pd.DataFrame({"Date": fdates[:len(p0)], "Predicted Close": p0})
        st.download_button("⬇️ Download Forecast CSV",
                           csv_df.to_csv(index=False).encode(),
                           file_name=f"{ticker}_forecast.csv",
                           mime="text/csv")


# ╔═══════════════════════════════════════╗
# ║  TAB 3 — Algorithms                  ║
# ╚═══════════════════════════════════════╝
with tabs[2]:
    a1, a2, a3 = st.columns(3)
    icons = ["🔴 Brute Force", "🟢 Greedy", "🔵 Dynamic Prog."]
    keys  = ["brute", "greedy", "dp"]
    for col, icon, key in zip([a1, a2, a3], icons, keys):
        col.markdown(mcard(icon,
                           f"₹{algo[f'{key}_profit']:,.2f}",
                           f"⏱ {algo[f'{key}_time']:.6f}s"),
                     unsafe_allow_html=True)

    st.plotly_chart(plotly_algo_bar(algo, ticker), use_container_width=True)

    st.markdown('<div class="section-hdr">Complexity Comparison</div>',
                unsafe_allow_html=True)
    st.markdown("""
<table class="algo-table">
<tr><th>Algorithm</th><th>Time</th><th>Space</th><th>Transactions</th><th>Constraint</th><th>Best For</th></tr>
<tr><td>Brute Force</td><td>O(n²)</td><td>O(1)</td><td>Single</td><td>None (capped)</td><td>Baseline / correctness check</td></tr>
<tr><td>Greedy</td><td>O(n)</td><td>O(1)</td><td>Unlimited</td><td>None</td><td>Maximum possible profit</td></tr>
<tr><td>Dynamic Prog.</td><td>O(n)</td><td>O(1)</td><td>Unlimited</td><td>1-day cooldown</td><td>Realistic constrained trading</td></tr>
</table>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ Algorithm Details"):
        st.markdown("""
**Brute Force** — Evaluates every possible (buy_day i, sell_day j) pair where j > i.
Guaranteed optimal for a single transaction but exponential in the uncapped recursive form.

**Greedy** — Single pass: accumulates every positive Δprice.
Equivalent to buying at each local minimum and selling at each local maximum.
Optimal for unlimited transactions with no constraints.

**Dynamic Programming** — Three rolling state variables track the optimal portfolio value
at each day given the cooldown constraint. O(n) time, O(1) space.
""")


# ╔═══════════════════════════════════════╗
# ║  TAB 4 — Indicators                  ║
# ╚═══════════════════════════════════════╝
with tabs[3]:
    st.plotly_chart(plotly_rsi(df, signals, ticker),  use_container_width=True)
    st.plotly_chart(plotly_macd(df, signals, ticker), use_container_width=True)

    # Bollinger Band squeeze meter
    st.markdown('<div class="section-hdr">Bollinger Band Width (Squeeze Meter)</div>',
                unsafe_allow_html=True)
    ub, lb = signals["bb_upper"], signals["bb_lower"]
    mid    = signals["sma20"] if "sma20" in signals else signals["bb_mid"]
    v      = ~(np.isnan(ub) | np.isnan(lb) | np.isnan(mid))
    if v.any():
        bw = (ub[v] - lb[v]) / mid[v] * 100
        bw_df = pd.DataFrame({"Date": dates[v], "Band Width %": bw})
        import plotly.express as px
        fig_bw = px.area(bw_df, x="Date", y="Band Width %",
                         color_discrete_sequence=["#58a6ff"])
        fig_bw.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                              font_color="#e6edf3", height=220,
                              margin=dict(l=40, r=20, t=30, b=30))
        st.plotly_chart(fig_bw, use_container_width=True)

    # Key indicator table
    rsi_now = float(signals["rsi"][-1]) if not np.isnan(signals["rsi"][-1]) else 50.0
    ma50_v  = float(signals["sma50"][-1])  if not np.isnan(signals["sma50"][-1])  else 0.0
    ma200_v = float(signals["sma200"][-1]) if not np.isnan(signals["sma200"][-1]) else 0.0
    bb_u    = float(signals["bb_upper"][-1]) if not np.isnan(signals["bb_upper"][-1]) else 0.0
    bb_l    = float(signals["bb_lower"][-1]) if not np.isnan(signals["bb_lower"][-1]) else 0.0
    macd_v  = float(signals["macd_line"][-1])
    price   = stats["current_price"]

    ind_data = {
        "Indicator": ["RSI (14)", "MA 50", "MA 200", "BB Upper", "BB Lower", "MACD"],
        "Value":     [f"{rsi_now:.1f}", f"₹{ma50_v:,.2f}", f"₹{ma200_v:,.2f}",
                      f"₹{bb_u:,.2f}", f"₹{bb_l:,.2f}", f"{macd_v:.4f}"],
        "Signal":    [
            "🔴 Overbought" if rsi_now > 70 else "🟢 Oversold" if rsi_now < 30 else "⬜ Neutral",
            "🟢 Bullish" if price > ma50_v else "🔴 Bearish",
            "🟢 Bullish" if price > ma200_v else "🔴 Bearish",
            "🔴 Above BB" if price > bb_u else "—",
            "🟢 Below BB" if price < bb_l else "—",
            "🟢 Positive" if macd_v > 0 else "🔴 Negative",
        ],
    }
    st.dataframe(pd.DataFrame(ind_data), use_container_width=True, hide_index=True)


# ╔═══════════════════════════════════════╗
# ║  TAB 5 — Backtesting                 ║
# ╚═══════════════════════════════════════╝
with tabs[4]:
    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(mcard("Total Return",  f"{bt['total_return_pct']:.2f}%",
                      f"₹{bt['total_profit']:,.0f} profit",
                      "#3fb950" if bt["total_return_pct"] >= 0 else "#f85149"),
                unsafe_allow_html=True)
    b2.markdown(mcard("Win Rate",  f"{bt['win_rate']:.1f}%",
                      f"{bt['num_trades']} trades"), unsafe_allow_html=True)
    b3.markdown(mcard("Max Drawdown", f"-{abs(bt['max_drawdown_pct']):.1f}%",
                      "Peak-to-trough"), unsafe_allow_html=True)
    b4.markdown(mcard("Sharpe (BT)", f"{bt['sharpe_ratio']:.4f}",
                      "Annualised"), unsafe_allow_html=True)

    st.plotly_chart(plotly_equity(bt, df, ticker), use_container_width=True)

    st.markdown('<div class="section-hdr">Trade Log</div>', unsafe_allow_html=True)
    if bt["trades"]:
        trade_df = pd.DataFrame(bt["trades"])
        trade_df["buy_date"]  = [str(df.index[i].date()) for i in trade_df["buy_day"]]
        trade_df["sell_date"] = [str(df.index[i].date()) for i in trade_df["sell_day"]]
        show_cols = ["buy_date", "sell_date", "buy_price", "sell_price", "profit", "return_pct"]
        styled = trade_df[show_cols].rename(columns={
            "buy_date": "Buy Date", "sell_date": "Sell Date",
            "buy_price": "Buy ₹", "sell_price": "Sell ₹",
            "profit": "Profit ₹", "return_pct": "Return %",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)
        csv_bytes = styled.to_csv(index=False).encode()
        st.download_button("⬇️ Download Trade Log CSV", csv_bytes,
                           file_name=f"{ticker}_trades.csv", mime="text/csv")
    else:
        st.info("No completed trades in this period.")


# ╔═══════════════════════════════════════╗
# ║  TAB 6 — Recommendation              ║
# ╚═══════════════════════════════════════╝
with tabs[5]:
    rc   = rec["recommendation"]
    rcol = REC_COLOUR[rc]
    rcls = f"rec-{rc.lower()}"

    left, mid_col, right = st.columns([1.3, 2, 2])

    with left:
        st.markdown(f"""
        <div style="text-align:center;padding:24px 0;">
            <div class="mlbl" style="font-size:14px;margin-bottom:14px;">RECOMMENDATION</div>
            <div class="rec-badge {rcls}">{rc}</div>
            <div style="margin-top:18px">
                <div class="mlbl">Confidence Score</div>
                <div style="font-size:32px;font-weight:700;color:{rcol}">{rec['confidence']}%</div>
                <div class="conf-bg">
                    <div class="conf-fg" style="width:{rec['confidence']}%;background:{rcol}"></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with mid_col:
        st.markdown('<div class="section-hdr">Price Targets</div>', unsafe_allow_html=True)
        p = stats["current_price"]
        for label, val, chg_pct in [
            ("Current Price",   p,               0),
            ("7-Day Forecast",  rec["pred_7d"],  rec["change_7d_pct"]),
            ("30-Day Forecast", rec["pred_30d"], rec["change_30d_pct"]),
        ]:
            c = "#3fb950" if chg_pct > 0 else "#f85149" if chg_pct < 0 else "#8b949e"
            s = "▲" if chg_pct > 0 else "▼" if chg_pct < 0 else "—"
            badge = (f'<span style="color:{c};font-weight:600">{s} {abs(chg_pct):.1f}%</span>'
                     if chg_pct != 0 else "")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'border-bottom:1px solid #30363d;padding:10px 0;">'
                f'<span style="color:#8b949e;font-size:13px">{label}</span>'
                f'<span style="font-weight:600;color:#e6edf3">₹{val:,.2f} {badge}</span>'
                f'</div>', unsafe_allow_html=True)

        # Risk metrics mini-table
        st.markdown("<br>", unsafe_allow_html=True)
        for lbl, val in [
            ("Volatility",     f"{risk['volatility']:.1f}%"),
            ("Sharpe Ratio",   f"{risk['sharpe_ratio']:.4f}"),
            ("Max Drawdown",   f"-{risk['max_drawdown_pct']:.2f}%"),
            ("Exp. Return",    f"{risk['expected_return_pct']:.2f}% p.a."),
            ("Risk Level",     risk["risk_level"]),
        ]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:7px 0;border-bottom:1px solid #21262d;">'
                f'<span style="color:#8b949e;font-size:13px">{lbl}</span>'
                f'<span style="color:#e6edf3;font-size:13px">{val}</span></div>',
                unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-hdr">Analysis Reasoning</div>', unsafe_allow_html=True)
        for r_str in rec["reasoning"]:
            st.markdown(f'<div class="reason-item">{r_str}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(plotly_risk_radar(risk, ticker), use_container_width=True)

    # Factor votes breakdown
    with st.expander("🔍 Factor votes breakdown"):
        fv = rec.get("factor_votes", {})
        fv_df = pd.DataFrame({
            "Factor": list(fv.keys()),
            "Vote":   [round(v, 3) for v in fv.values()],
            "Direction": ["🟢 Bullish" if v > 0 else "🔴 Bearish" if v < 0 else "⬜ Neutral"
                          for v in fv.values()],
        })
        st.dataframe(fv_df, use_container_width=True, hide_index=True)

    st.divider()
    # Export full report
    if st.button("📄 Export Full Report CSV"):
        path = export_csv(ticker, output_dir="output")
        st.success(f"Saved to {path}")

    st.caption("⚠️ This is an educational tool. Not financial advice.")


# ╔═══════════════════════════════════════╗
# ║  TAB 7 — Portfolio                   ║
# ╚═══════════════════════════════════════╝
with tabs[6]:
    portfolio = get_portfolio()
    st.markdown('<div class="section-hdr">📂 Watchlist</div>', unsafe_allow_html=True)

    if not portfolio:
        st.info("No stocks in portfolio. Click '💾 Add to Portfolio' in the sidebar.")
    else:
        # Mini-analysis for each ticker
        port_rows = []
        for t in portfolio:
            try:
                with st.spinner(f"Loading {t}…"):
                    pf_df, pf_src   = fetch_stock_data(t)
                    pf_stats        = get_price_stats(pf_df)
                    pf_closes       = pf_df["Close"].values.astype(float)
                    pf_risk         = compute_all_risk(pf_closes)
                    pf_signals      = detect_signals(pf_df, strategy="combined")
                    pf_sig          = latest_signal(pf_signals, pf_closes)
                    pf_preds        = run_all_predictions(pf_closes, horizon=7, lookback=20)
                    pf_rec          = generate_recommendation(
                        pf_stats["current_price"], pf_preds, pf_signals, pf_sig, pf_risk)
                port_rows.append({
                    "Ticker":  t,
                    "Price":   f"₹{pf_stats['current_price']:,.2f}",
                    "Change":  f"{pf_stats['change_pct']:+.2f}%",
                    "Volatility": f"{pf_risk['volatility']:.1f}%",
                    "Sharpe":  f"{pf_risk['sharpe_ratio']:.2f}",
                    "Signal":  pf_sig,
                    "Rec":     pf_rec["recommendation"],
                    "Confidence": f"{pf_rec['confidence']}%",
                })
            except Exception:
                port_rows.append({"Ticker": t, "Price": "Error", "Change": "—",
                                  "Volatility": "—", "Sharpe": "—",
                                  "Signal": "—", "Rec": "—", "Confidence": "—"})

        pf_df_show = pd.DataFrame(port_rows)
        st.dataframe(pf_df_show, use_container_width=True, hide_index=True)

        remove_t = st.selectbox("Remove from portfolio", ["—"] + portfolio)
        if st.button("🗑️ Remove") and remove_t != "—":
            remove_from_portfolio(remove_t)
            st.success(f"{remove_t} removed.")
            st.rerun()


# ╔═══════════════════════════════════════╗
# ║  TAB 8 — History                     ║
# ╚═══════════════════════════════════════╝
with tabs[7]:
    h1, h2 = st.columns(2)

    with h1:
        st.markdown('<div class="section-hdr">Recent Searches</div>', unsafe_allow_html=True)
        history = get_history(limit=30)
        if not history.empty:
            st.dataframe(history, use_container_width=True, hide_index=True)

    with h2:
        st.markdown('<div class="section-hdr">Past Analyses</div>', unsafe_allow_html=True)
        past = get_analyses(limit=30)
        if not past.empty:
            st.dataframe(past, use_container_width=True, hide_index=True)
        else:
            st.info("No analyses stored yet.")

    if st.button("⬇️ Export All Analyses for this ticker"):
        path = export_csv(ticker, output_dir="output")
        with open(path, "rb") as f:
            st.download_button("📥 Download CSV", f.read(),
                               file_name=os.path.basename(path),
                               mime="text/csv")