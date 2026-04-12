"""
brute.py — TRENDOPT
Brute-force stock profit optimisation.

Strategy : Try every possible (buy_day, sell_day) pair where sell_day > buy_day.
           A recursive divide-and-conquer helper is used to illustrate the
           exponential exploration, but we cap the input at 20 days so the demo
           finishes in reasonable time.

Time Complexity  : O(2^n)  — exponential (illustrative)
Space Complexity : O(n)    — recursion stack
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Core recursive implementation
# ---------------------------------------------------------------------------

def _brute_recursive(prices: list, start: int, end: int) -> Tuple[float, int, int]:
    """
    Recursively find the maximum profit by exploring all buy/sell subsets.

    Args:
        prices : List of daily closing prices.
        start  : Starting index of the subarray to consider.
        end    : Ending index (inclusive) of the subarray to consider.

    Returns:
        Tuple of (max_profit, best_buy_day, best_sell_day).
    """
    # Base case: only one price available — no transaction possible
    if start >= end:
        return 0.0, start, start

    best_profit = 0.0
    best_buy = start
    best_sell = start

    # Enumerate every (i, j) pair: buy on day i, sell on day j
    for i in range(start, end):
        for j in range(i + 1, end + 1):
            profit = prices[j] - prices[i]
            if profit > best_profit:
                best_profit = profit
                best_buy = i
                best_sell = j

    return best_profit, best_buy, best_sell


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def brute_force_max_profit(prices: np.ndarray, max_days: int = 20) -> dict:
    """
    Compute the maximum single-transaction profit using brute force.

    Because brute force is O(2^n) we limit the input to ``max_days`` days
    so the function returns in a reasonable time during demonstration.

    Args:
        prices  : Array of closing prices.
        max_days: Maximum number of days to consider (default 20).

    Returns:
        Dict with keys:
            profit      — Maximum achievable profit (float)
            buy_day     — Index at which to buy
            sell_day    — Index at which to sell
            buy_price   — Price on buy day
            sell_price  — Price on sell day
    """
    prices = np.asarray(prices, dtype=float)

    # Cap to avoid freezing on large arrays
    subset = prices[:max_days].tolist()
    n = len(subset)

    if n < 2:
        return {"profit": 0.0, "buy_day": 0, "sell_day": 0,
                "buy_price": prices[0], "sell_price": prices[0]}

    profit, buy_idx, sell_idx = _brute_recursive(subset, 0, n - 1)

    return {
        "profit": round(profit, 2),
        "buy_day": buy_idx,
        "sell_day": sell_idx,
        "buy_price": round(subset[buy_idx], 2),
        "sell_price": round(subset[sell_idx], 2),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = [100, 90, 110, 85, 130, 120, 150, 95, 160]
    result = brute_force_max_profit(np.array(sample))
    print("[BruteForce] Result:", result)
