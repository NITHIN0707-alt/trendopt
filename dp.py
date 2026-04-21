"""Compatibility wrapper for the canonical DP implementation."""

from backend.dp import dp_cooldown

dp_max_profit_with_cooldown = dp_cooldown

__all__ = ["dp_cooldown", "dp_max_profit_with_cooldown"]
