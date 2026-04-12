"""
backend/greedy.py  —  TRENDOPT
=================================
Greedy unlimited-transaction strategy.

Strategy  : Accumulate every positive day-over-day price delta.
Complexity: O(n)
"""

import numpy as np
from typing import List


def greedy(prices: np.ndarray) -> dict:
    """
    Maximise total profit with unlimited buy/sell transactions.

    Args:
        prices: 1-D closing price array.

    Returns:
        Dict with keys: profit, num_trades, transactions, complexity.
    """
    prices = np.asarray(prices, dtype=float)
    total  = float(np.sum(np.maximum(np.diff(prices), 0)))

    # Reconstruct individual trades for display
    trades: List[dict] = []
    i = 1
    while i < len(prices):
        while i < len(prices) and prices[i] <= prices[i - 1]:
            i += 1
        buy = i - 1
        while i < len(prices) and prices[i] >= prices[i - 1]:
            i += 1
        sell = i - 1
        if sell > buy:
            trades.append({
                "buy_day":   int(buy),
                "sell_day":  int(sell),
                "profit":    round(float(prices[sell] - prices[buy]), 2),
            })

    return {
        "profit":       round(total, 2),
        "num_trades":   len(trades),
        "transactions": trades,
        "complexity":   "O(n)",
    }
