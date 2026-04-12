"""
dp.py — TRENDOPT
Dynamic-programming stock profit optimisation with cooldown.

Strategy : After each SELL the trader must rest for one day (cooldown).
           Three states are tracked per day:
               buy[i]  = max profit when holding a stock on day i
               sell[i] = max profit when just sold on day i
               rest[i] = max profit when in cooldown (or doing nothing) on day i

Recurrences:
    buy[i]  = max(buy[i-1],  rest[i-1] - price[i])
    sell[i] = buy[i-1]  + price[i]
    rest[i] = max(rest[i-1], sell[i-1])

Time Complexity  : O(n)
Space Complexity : O(1)  (rolling variables — no arrays stored)
"""

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# State reconstruction helper
# ---------------------------------------------------------------------------

def _reconstruct_trades(prices: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Reconstruct the actual buy/sell actions from the DP solution.

    This is a heuristic reconstruction; it traces back through the state
    arrays to recover a valid action sequence.

    Args:
        prices: 1-D array of closing prices.

    Returns:
        List of (buy_day, sell_day, profit) tuples.
    """
    n = len(prices)
    if n < 2:
        return []

    # Build state arrays for reconstruction
    buy = np.full(n, -np.inf)
    sell = np.full(n, 0.0)
    rest = np.full(n, 0.0)

    buy[0] = -prices[0]

    for i in range(1, n):
        buy[i] = max(buy[i - 1], rest[i - 1] - prices[i])
        sell[i] = buy[i - 1] + prices[i]
        rest[i] = max(rest[i - 1], sell[i - 1])

    # Greedy reconstruction of trades
    trades = []
    i = n - 1
    while i > 0:
        if sell[i] > rest[i] and sell[i] > 0:
            sell_day = i
            # Find the buy day (last day where buy[j] + price[i] == sell[i])
            j = i - 1
            while j >= 0:
                if abs(buy[j] + prices[i] - sell[i]) < 1e-6:
                    buy_day = j
                    profit = prices[sell_day] - prices[buy_day]
                    if profit > 0:
                        trades.append((buy_day, sell_day, round(float(profit), 2)))
                    i = j - 2  # skip cooldown day
                    break
                j -= 1
            else:
                i -= 1
        else:
            i -= 1

    trades.sort(key=lambda t: t[0])
    return trades


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dp_max_profit_with_cooldown(prices: np.ndarray) -> dict:
    """
    Compute the maximum total profit using DP with a one-day cooldown.

    Args:
        prices: Array of closing prices.

    Returns:
        Dict with keys:
            profit       — Maximum achievable profit (float)
            trades       — List of (buy_day, sell_day, profit) tuples
            num_trades   — Number of trades executed
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)

    if n < 2:
        return {"profit": 0.0, "trades": [], "num_trades": 0}

    # ---- Rolling DP --------------------------------------------------------
    buy_prev  = -prices[0]   # max profit while holding after day 0
    sell_prev = 0.0          # max profit just after selling on day 0
    rest_prev = 0.0          # max profit while resting on day 0

    for i in range(1, n):
        buy_curr  = max(buy_prev,  rest_prev - prices[i])
        sell_curr = buy_prev + prices[i]
        rest_curr = max(rest_prev, sell_prev)

        buy_prev, sell_prev, rest_prev = buy_curr, sell_curr, rest_curr

    total_profit = max(sell_prev, rest_prev)

    # Reconstruct trades for display
    trades = _reconstruct_trades(prices)

    return {
        "profit": round(float(max(total_profit, 0.0)), 2),
        "trades": trades,
        "num_trades": len(trades),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = np.array([1, 2, 3, 0, 2], dtype=float)
    result = dp_max_profit_with_cooldown(sample)
    print("[DP] Result:", result)          # Expected profit: 3  (buy@1,sell@3 rest@0, buy@0,sell@2 → actually 3)
