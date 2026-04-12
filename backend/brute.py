"""
backend/brute.py  —  TRENDOPT
================================
Brute-Force single-transaction profit maximisation.

Strategy : Test every (buy_day i, sell_day j) pair where j > i.
Complexity: O(n²)   capped at ``max_days`` for interactive safety.
"""

import numpy as np


def brute_force(prices: np.ndarray, max_days: int = 22) -> dict:
    """
    Find the maximum single-transaction profit by exhaustive search.

    Args:
        prices   : 1-D array of closing prices.
        max_days : Maximum days to evaluate (default 22).

    Returns:
        Dict with keys: profit, buy_day, sell_day, buy_price, sell_price.
    """
    prices = np.asarray(prices, dtype=float)[:max_days]
    n      = len(prices)
    best   = {"profit": 0.0, "buy_day": 0, "sell_day": 0}

    for i in range(n - 1):
        for j in range(i + 1, n):
            p = prices[j] - prices[i]
            if p > best["profit"]:
                best = {"profit": p, "buy_day": i, "sell_day": j}

    return {
        "profit":     round(float(best["profit"]), 2),
        "buy_day":    int(best["buy_day"]),
        "sell_day":   int(best["sell_day"]),
        "buy_price":  round(float(prices[best["buy_day"]]),  2),
        "sell_price": round(float(prices[best["sell_day"]]), 2),
        "complexity": "O(n²)",
    }
