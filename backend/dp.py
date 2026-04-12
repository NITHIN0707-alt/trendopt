"""
backend/dp.py  —  TRENDOPT
=============================
Dynamic-Programming profit optimisation with 1-day cooldown.

States per day i
----------------
    hold[i]  = max profit while HOLDING a stock on day i
    sold[i]  = max profit just after SELLING on day i
    rest[i]  = max profit while in COOLDOWN / idle on day i

Recurrences
-----------
    hold[i] = max(hold[i-1],  rest[i-1] - price[i])
    sold[i] = hold[i-1] + price[i]
    rest[i] = max(rest[i-1], sold[i-1])

Complexity: O(n) time, O(1) space
"""

import numpy as np


def dp_cooldown(prices: np.ndarray) -> dict:
    """
    Maximise profit with unlimited transactions and a 1-day post-sell cooldown.

    Args:
        prices: 1-D closing price array.

    Returns:
        Dict with keys: profit, states_evaluated, complexity.
    """
    prices = np.asarray(prices, dtype=float)
    n      = len(prices)

    if n < 2:
        return {"profit": 0.0, "states_evaluated": 0, "complexity": "O(n)"}

    hold, sold, rest = -prices[0], 0.0, 0.0

    for i in range(1, n):
        hold, sold, rest = (
            max(hold, rest - prices[i]),
            hold + prices[i],
            max(rest, sold),
        )

    profit = max(sold, rest)
    return {
        "profit":            round(float(max(profit, 0.0)), 2),
        "states_evaluated":  n * 3,
        "complexity":        "O(n)",
    }
