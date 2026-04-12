"""
backend/backtesting.py  —  TRENDOPT
======================================
Strategy backtesting engine.

Simulates trading on historical data using the signals from
signal_detector.py and tracks the resulting equity curve.

Metrics computed
----------------
total_profit, total_return_pct, win_rate, loss_rate,
num_trades, max_drawdown, sharpe_ratio, avg_trade_profit,
best_trade, worst_trade, strategy_accuracy

Public API
----------
backtest(df, signals, initial_cash=100_000) → dict
"""

import numpy as np
import pandas as pd
from typing import List


def backtest(
    df:           pd.DataFrame,
    signals:      dict,
    initial_cash: float = 100_000.0,
) -> dict:
    """
    Simulate a long-only strategy on the given OHLCV data.

    Rules:
        - BUY at the closing price of each BUY signal day (buy one full share).
        - SELL at the closing price of each SELL signal day if holding.
        - At end of period, sell any open position at the last close.

    Args:
        df           : OHLCV DataFrame.
        signals      : Output of signal_detector.detect_signals().
        initial_cash : Starting capital (default ₹100 000).

    Returns:
        Dict with all performance metrics and the equity curve.
    """
    closes      = df["Close"].values.astype(float)
    n           = len(closes)
    bi_set      = set(int(x) for x in signals["buy_indices"])
    si_set      = set(int(x) for x in signals["sell_indices"])

    cash        = initial_cash
    holding     = False
    buy_price   = 0.0
    equity      = []                  # portfolio value per day
    trades: List[dict] = []

    for i in range(n):
        price = closes[i]

        if i in bi_set and not holding and cash >= price:
            shares       = int(cash // price)
            cost         = shares * price
            cash        -= cost
            holding      = True
            buy_price    = price
            buy_day      = i
            buy_shares   = shares

        elif i in si_set and holding:
            revenue  = buy_shares * price
            profit   = revenue - buy_shares * buy_price
            cash    += revenue
            trades.append({
                "buy_day":    int(buy_day),
                "sell_day":   int(i),
                "buy_price":  round(buy_price, 2),
                "sell_price": round(price, 2),
                "profit":     round(profit, 2),
                "return_pct": round(profit / (buy_shares * buy_price) * 100, 2),
            })
            holding = False

        port_val = cash + (buy_shares * price if holding else 0.0)
        equity.append(port_val)

    # Close any remaining position at final price
    if holding:
        revenue = buy_shares * closes[-1]
        profit  = revenue - buy_shares * buy_price
        cash   += revenue
        trades.append({
            "buy_day":    int(buy_day),
            "sell_day":   int(n - 1),
            "buy_price":  round(buy_price, 2),
            "sell_price": round(closes[-1], 2),
            "profit":     round(profit, 2),
            "return_pct": round(profit / (buy_shares * buy_price) * 100, 2),
        })
        equity[-1] = cash

    final_value  = equity[-1] if equity else initial_cash
    total_profit = final_value - initial_cash
    total_return = total_profit / initial_cash * 100

    # Per-trade stats
    profits       = [t["profit"] for t in trades]
    wins          = [p for p in profits if p > 0]
    losses        = [p for p in profits if p <= 0]
    win_rate      = len(wins)  / len(profits) * 100 if profits else 0.0
    loss_rate     = len(losses)/ len(profits) * 100 if profits else 0.0

    # Maximum drawdown
    eq_arr = np.array(equity if equity else [initial_cash])
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peak) / peak
    max_dd = float(np.min(dd)) * 100        # negative number

    # Daily returns & Sharpe
    eq_s       = pd.Series(eq_arr)
    daily_ret  = eq_s.pct_change().dropna()
    sharpe     = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if daily_ret.std() > 0 else 0.0)

    # Accuracy (fraction of profitable trades)
    accuracy = win_rate

    return {
        "initial_cash":    round(initial_cash, 2),
        "final_value":     round(final_value, 2),
        "total_profit":    round(total_profit, 2),
        "total_return_pct":round(total_return, 2),
        "num_trades":      len(trades),
        "win_rate":        round(win_rate, 2),
        "loss_rate":       round(loss_rate, 2),
        "max_drawdown_pct":round(max_dd, 2),
        "sharpe_ratio":    round(float(sharpe), 4),
        "avg_trade_profit":round(float(np.mean(profits)) if profits else 0.0, 2),
        "best_trade":      round(float(max(profits)) if profits else 0.0, 2),
        "worst_trade":     round(float(min(profits)) if profits else 0.0, 2),
        "strategy_accuracy": round(accuracy, 2),
        "equity_curve":    equity,
        "trades":          trades,
    }
