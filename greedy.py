"""Compatibility wrapper for the canonical greedy implementation."""

from backend.greedy import greedy

greedy_max_profit = greedy

__all__ = ["greedy", "greedy_max_profit"]
