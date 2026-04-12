"""
main.py  —  TRENDOPT
======================
Command-line entry point.

Usage:
    python main.py
    python main.py --ticker RELIANCE.NS --horizon 30
    python main.py --ticker AAPL --horizon 7 --brute-days 20
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import numpy as np

from backend.data_fetcher        import resolve_ticker, fetch_stock_data, get_price_stats
from backend.brute               import brute_force
from backend.greedy              import greedy
from backend.dp                  import dp_cooldown
from backend.signal_detector     import detect_signals, latest_signal, active_alerts
from backend.prediction_model    import run_all_predictions, consensus_forecast
from backend.backtesting         import backtest
from backend.risk_analysis       import compute_all_risk
from backend.recommendation_engine import generate_recommendation
from database.db_manager         import log_search, save_analysis, save_predictions
from visualization.plot_graphs   import save_dashboard_png


def run(ticker_raw: str, horizon: int = 30, brute_days: int = 22, strategy: str = "combined"):
    ticker = resolve_ticker(ticker_raw)
    LINE   = "=" * 66

    print(f"\n{LINE}")
    print(f"  TRENDOPT — Intelligent Stock Analyzer & Predictor")
    print(f"  Ticker: {ticker}   Horizon: {horizon}d   Strategy: {strategy}")
    print(LINE)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print("\n[1/7] Fetching historical data …")
    df, src = fetch_stock_data(ticker)
    is_live = src.is_live
    stats       = get_price_stats(df)
    closes      = df["Close"].values.astype(float)
    log_search(ticker, src.is_live)

    mode = "LIVE" if is_live else "SYNTHETIC (demo)"
    chg  = stats["change_pct"]
    sym  = "▲" if chg >= 0 else "▼"
    print(f"  Source       : {mode}")
    print(f"  Period       : {stats['start_date']} → {stats['end_date']}")
    print(f"  Days         : {stats['trading_days']}")
    print(f"  Price        : ₹{stats['current_price']:,.2f}  ({sym}{abs(chg):.2f}%)")
    print(f"  52W High/Low : ₹{stats['high_52w']:,.2f} / ₹{stats['low_52w']:,.2f}")

    # ── 2. Algorithms ─────────────────────────────────────────────────────
    print(f"\n[2/7] Running DAA algorithms …")
    algo: dict = {}
    for name, fn, kwargs in [
        ("brute",  brute_force,  {"max_days": brute_days}),
        ("greedy", greedy,       {}),
        ("dp",     dp_cooldown,  {}),
    ]:
        t0   = time.perf_counter()
        res  = fn(closes, **kwargs)
        dt   = time.perf_counter() - t0
        algo[f"{name}_profit"] = res["profit"]
        algo[f"{name}_time"]   = round(dt, 6)
        algo[f"{name}_complexity"] = res.get("complexity", "—")

    print(f"\n  {'Algorithm':<22} {'Profit':>12}  {'Time':>12}  {'Complexity'}")
    print(f"  {'─'*64}")
    rows = [
        ("Brute Force",   "brute",  "O(n²)"),
        ("Greedy",        "greedy", "O(n)"),
        ("Dynamic Prog.", "dp",     "O(n)"),
    ]
    for lbl, key, cplx in rows:
        print(f"  {lbl:<22} ₹{algo[f'{key}_profit']:>10,.2f}  "
              f"{algo[f'{key}_time']:>10.6f}s  {cplx}")

    # ── 3. Signals ────────────────────────────────────────────────────────
    print(f"\n[3/7] Detecting buy/sell signals (strategy={strategy}) …")
    signals  = detect_signals(df, strategy=strategy)
    sig_str  = latest_signal(signals, closes)
    alerts   = active_alerts(signals, closes, ticker)
    print(f"  BUY  signals : {len(signals['buy_indices'])}")
    print(f"  SELL signals : {len(signals['sell_indices'])}")
    print(f"  Latest       : {sig_str}")
    for a in alerts[:3]:
        print(f"  Alert: {a}")

    # ── 4. ML Predictions ─────────────────────────────────────────────────
    print(f"\n[4/7] Training ML models (horizon={horizon}d) …")
    lookback    = min(30, len(closes) // 4)
    pred_results = run_all_predictions(closes, horizon=horizon, lookback=lookback)
    consensus   = consensus_forecast(pred_results, horizon)

    for r in pred_results:
        p = r.get("predictions", [])
        if p:
            d7  = p[6]  if len(p) > 6  else "—"
            d30 = p[29] if len(p) > 29 else "—"
            print(f"  [{r['model_name']}]  7d=₹{d7:,}  30d=₹{d30:,}  "
                  f"MAE={r['mae']}  R²={r['r2']}")

    save_predictions(ticker, pred_results, horizon)

    # ── 5. Backtesting ────────────────────────────────────────────────────
    print(f"\n[5/7] Backtesting strategy …")
    bt = backtest(df, signals)
    print(f"  Total return : {bt['total_return_pct']:.2f}%")
    print(f"  Trades       : {bt['num_trades']}  (win rate {bt['win_rate']:.1f}%)")
    print(f"  Max drawdown : {bt['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe ratio : {bt['sharpe_ratio']:.4f}")

    # ── 6. Risk analysis ──────────────────────────────────────────────────
    print(f"\n[6/7] Computing risk metrics …")
    risk = compute_all_risk(closes)
    print(f"  Volatility   : {risk['volatility']:.2f}%  ({risk['risk_level']})")
    print(f"  Sharpe ratio : {risk['sharpe_ratio']:.4f}")
    print(f"  Max drawdown : {risk['max_drawdown_pct']:.2f}%")
    print(f"  VaR 95%      : {risk['var_95_pct']:.2f}%")
    print(f"  Exp. return  : {risk['expected_return_pct']:.2f}% p.a.")

    # ── 7. Recommendation ─────────────────────────────────────────────────
    print(f"\n[7/7] Generating recommendation …")
    rec = generate_recommendation(
        stats["current_price"], pred_results, signals, sig_str, risk
    )
    save_analysis(ticker, stats, rec, risk, algo)

    rec_icons = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
    print(f"""
  ╔══════════════════════════════════════════════╗
  ║  {rec_icons[rec['recommendation']]}  {rec['recommendation']:<5}  │  Confidence: {rec['confidence']}%{' ' * (14 - len(str(rec['confidence'])))}║
  ║  7-day  forecast : ₹{rec['pred_7d']:>10,.2f}               ║
  ║  30-day forecast : ₹{rec['pred_30d']:>10,.2f}               ║
  ╚══════════════════════════════════════════════╝""")
    print("\n  Reasoning:")
    for r_str in rec["reasoning"]:
        print(f"    {r_str}")

    # ── Charts ────────────────────────────────────────────────────────────
    print("\n  Generating dashboard PNG …")
    path = save_dashboard_png(df, signals, pred_results, bt, algo, rec, ticker)
    print(f"  Saved → {path}")

    print(f"\n{LINE}")
    print("  TRENDOPT run complete! Launch dashboard: streamlit run frontend/dashboard.py")
    print(f"{LINE}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TRENDOPT CLI")
    ap.add_argument("--ticker",     default="RELIANCE.NS")
    ap.add_argument("--horizon",    type=int, default=30)
    ap.add_argument("--brute-days", type=int, default=22)
    ap.add_argument("--strategy",   default="combined",
                    choices=["combined","local_extrema","rsi","ma_crossover","bb","macd"])
    args = ap.parse_args()
    try:
        run(args.ticker, args.horizon, args.brute_days, args.strategy)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)