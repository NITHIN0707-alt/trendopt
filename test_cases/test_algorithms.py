"""
Unit tests for the canonical TRENDOPT algorithm modules.

Run with:
    python -m pytest test_cases/ -v
    # or without pytest:
    python test_cases/test_algorithms.py
"""

import sys
import os
import numpy as np
import pandas as pd

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.brute import brute_force
from backend.greedy import greedy
from backend.dp import dp_cooldown
from backend.signal_detector import detect_signals


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_close(got, expected, tol=0.01, label=""):
    ok = abs(got - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: got={got:.4f}, expected={expected:.4f}")
    assert ok, f"{label}: got={got:.4f}, expected={expected:.4f}"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_brute_force():
    print("\n-- Brute Force --------------------------------------")
    p1 = np.array([7, 1, 5, 3, 6, 4], dtype=float)
    r  = brute_force(p1, max_days=len(p1))
    _assert_close(r["profit"], 5.0, label="buy@1 sell@6")

    p2 = np.array([7, 6, 4, 3, 1], dtype=float)
    r2 = brute_force(p2, max_days=len(p2))
    _assert_close(r2["profit"], 0.0, label="no profit (descending)")

    p3 = np.array([100, 90, 110, 85, 130], dtype=float)
    r3 = brute_force(p3, max_days=len(p3))
    _assert_close(r3["profit"], 45.0, label="buy@85 sell@130")


def test_greedy():
    print("\n-- Greedy -------------------------------------------")
    p1 = np.array([7, 1, 5, 3, 6, 4], dtype=float)
    r1 = greedy(p1)
    _assert_close(r1["profit"], 7.0, label="capture all ups: (5-1)+(6-3)=7")

    p2 = np.array([1, 2, 3, 4, 5], dtype=float)
    r2 = greedy(p2)
    _assert_close(r2["profit"], 4.0, label="monotone rise profit=4")

    p3 = np.array([5, 4, 3, 2, 1], dtype=float)
    r3 = greedy(p3)
    _assert_close(r3["profit"], 0.0, label="descending prices -> 0")


def test_dp():
    print("\n-- Dynamic Programming (with cooldown) --------------")
    p1 = np.array([1, 2, 3, 0, 2], dtype=float)
    r1 = dp_cooldown(p1)
    _assert_close(r1["profit"], 3.0, label="classic cooldown example -> 3")

    p2 = np.array([1], dtype=float)
    r2 = dp_cooldown(p2)
    _assert_close(r2["profit"], 0.0, label="single price -> 0")

    p3 = np.array([6, 1, 3, 2, 4, 7], dtype=float)
    r3 = dp_cooldown(p3)
    assert r3["profit"] >= 0, "DP profit must be non-negative"
    print(f"  [INFO] p3 DP profit={r3['profit']:.2f}  (cooldown may limit)")


def test_signals():
    print("\n-- Signal Detection --------------------------------")
    prices = np.array([10, 8, 12, 11, 15, 13, 16, 14, 18], dtype=float)
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 0.5,
            "Low": prices - 0.5,
            "Close": prices,
            "Volume": np.full(len(prices), 1000.0),
        }
    )
    signals = detect_signals(df, strategy="local_extrema")
    buys = signals["buy_indices"]
    sells = signals["sell_indices"]
    print(f"  Buy  indices : {buys.tolist()}")
    print(f"  Sell indices : {sells.tolist()}")
    assert len(buys) >= 1, "Expected at least one BUY signal"
    assert len(sells) >= 1, "Expected at least one SELL signal"
    print("  [PASS] At least one buy and one sell signal detected")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_brute_force()
    test_greedy()
    test_dp()
    test_signals()
    print("\nAll tests completed.\n")
