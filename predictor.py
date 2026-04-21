"""Compatibility wrapper around the canonical prediction module."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from backend.prediction_model import consensus_forecast, run_all_predictions


def _pick_result(results: List[Dict], model_name: str) -> Dict:
    for result in results:
        if result["model_name"] == model_name:
            return result
    return {
        "model_name": model_name,
        "predictions": [],
        "mae": None,
        "rmse": None,
        "r2": None,
        "mape": None,
    }


def predict_linear_regression(
    prices: np.ndarray,
    horizon: int = 30,
    lookback: int = 30,
) -> Dict:
    return _pick_result(
        run_all_predictions(prices, horizon=horizon, lookback=lookback),
        "Linear Regression",
    )


def predict_random_forest(
    prices: np.ndarray,
    horizon: int = 30,
    lookback: int = 30,
    n_estimators: int = 100,
) -> Dict:
    del n_estimators
    return _pick_result(
        run_all_predictions(prices, horizon=horizon, lookback=lookback),
        "Random Forest",
    )


def predict_lstm(
    prices: np.ndarray,
    horizon: int = 30,
    lookback: int = 30,
    epochs: int = 20,
) -> Dict:
    del epochs
    return _pick_result(
        run_all_predictions(prices, horizon=horizon, lookback=lookback),
        "LSTM",
    )


__all__ = [
    "consensus_forecast",
    "predict_linear_regression",
    "predict_random_forest",
    "predict_lstm",
    "run_all_predictions",
]
