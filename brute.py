"""Compatibility wrapper for the canonical brute-force implementation."""

from backend.brute import brute_force

brute_force_max_profit = brute_force

__all__ = ["brute_force", "brute_force_max_profit"]
