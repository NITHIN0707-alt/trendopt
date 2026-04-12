"""
greedy.py — TRENDOPT
Greedy stock profit optimisation (unlimited transactions).

Strategy : Accumulate every positive day-over-day price movement.
           Equivalent to buying at every local minimum and selling at every
           local maximum in a single pass.

Time Complexity  : O(n)
Space Complexity : O(1)
"""

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def _collect_transactions(prices: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Identify individual buy/sell transactions captured by the greedy strategy.

    A transaction starts whenever prices[i] > prices[i-1] and the "in-trade"
    flag is off, and ends when prices[i] < prices[i-1].

    Args:
        prices: 1-D array of closing prices.

    Returns:
        List of (buy_idx, sell_idx, profit) tuples.
    """
    transactions = []
    n = len(prices)
    i = 1
    while i < n:
        # Find the start of an upward run
        while i < n and prices[i] <= prices[i - 1]:
            i += 1
        buy_idx = i - 1

        # Find the end of the upward run
        while i < n and prices[i] >= prices[i - 1]:
            i += 1
        sell_idx = i - 1

        if sell_idx > buy_idx:
            profit = prices[sell_idx] - prices[buy_idx]
            transactions.append((buy_idx, sell_idx, round(float(profit), 2)))

    return transactions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def greedy_max_profit(prices: np.ndarray) -> dict:
    """
    Compute the maximum total profit using the greedy approach.

    Args:
        prices: Array of closing prices.

    Returns:
        Dict with keys:
            profit       — Total accumulated profit (float)
            transactions — List of (buy_day, sell_day, profit) tuples
            num_trades   — Number of individual trades executed
    """
    prices = np.asarray(prices, dtype=float)

    if len(prices) < 2:
        return {"profit": 0.0, "transactions": [], "num_trades": 0}

    # Single-pass profit accumulation
    total_profit = float(np.sum(np.maximum(np.diff(prices), 0)))

    transactions = _collect_transactions(prices)

    return {
        "profit": round(total_profit, 2),
        "transactions": transactions,
        "num_trades": len(transactions),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = np.array([100, 90, 110, 85, 130, 120, 150, 95, 160], dtype=float)
    result = greedy_max_profit(sample)
    print("[Greedy] Result:", result)
