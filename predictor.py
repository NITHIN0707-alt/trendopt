"""
predictor.py — TRENDOPT
Machine-learning price prediction module.

Implements:
    1. Linear Regression
    2. Random Forest Regressor
    3. LSTM Neural Network (optional — activated only if TensorFlow is available)

All models predict the next ``horizon`` closing prices given a window of
``lookback`` historical prices as features.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple

# Optional deep-learning import
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_features(prices: np.ndarray, lookback: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create supervised learning (X, y) pairs using a rolling window.

    X[i] = prices[i : i + lookback]   (feature vector)
    y[i] = prices[i + lookback]        (target — next day close)

    Args:
        prices   : 1-D price array.
        lookback : Number of past days used as features.

    Returns:
        Tuple (X, y) of NumPy arrays.
    """
    X, y = [], []
    for i in range(len(prices) - lookback):
        X.append(prices[i: i + lookback])
        y.append(prices[i + lookback])
    return np.array(X), np.array(y)


def _train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2):
    """Simple chronological train/test split (no shuffling)."""
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------

def predict_linear_regression(
    prices: np.ndarray,
    horizon: int = 30,
    lookback: int = 30,
) -> Dict:
    """
    Predict future prices using Linear Regression.

    Args:
        prices   : Historical closing prices.
        horizon  : Number of future days to predict.
        lookback : Rolling window size for features.

    Returns:
        Dict with 'predictions', 'model_name', 'mae', 'rmse'.
    """
    prices = np.asarray(prices, dtype=float)
    X, y = _build_features(prices, lookback)
    X_train, X_test, y_train, y_test = _train_test_split(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Iterative future prediction
    window = prices[-lookback:].tolist()
    future = []
    for _ in range(horizon):
        feat = np.array(window[-lookback:]).reshape(1, -1)
        pred = float(model.predict(feat)[0])
        future.append(round(pred, 2))
        window.append(pred)

    return {
        "model_name": "Linear Regression",
        "predictions": future,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }


# ---------------------------------------------------------------------------
# Random Forest Regressor
# ---------------------------------------------------------------------------

def predict_random_forest(
    prices: np.ndarray,
    horizon: int = 30,
    lookback: int = 30,
    n_estimators: int = 100,
) -> Dict:
    """
    Predict future prices using a Random Forest Regressor.

    Args:
        prices        : Historical closing prices.
        horizon       : Number of future days to predict.
        lookback      : Rolling window size for features.
        n_estimators  : Number of trees in the forest.

    Returns:
        Dict with 'predictions', 'model_name', 'mae', 'rmse'.
    """
    prices = np.asarray(prices, dtype=float)
    X, y = _build_features(prices, lookback)
    X_train, X_test, y_train, y_test = _train_test_split(X, y)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_test = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Iterative future prediction
    window = prices[-lookback:].tolist()
    future = []
    for _ in range(horizon):
        feat = np.array(window[-lookback:]).reshape(1, -1)
        pred = float(model.predict(feat)[0])
        future.append(round(pred, 2))
        window.append(pred)

    return {
        "model_name": "Random Forest",
        "predictions": future,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }


# ---------------------------------------------------------------------------
# LSTM Neural Network (optional)
# ---------------------------------------------------------------------------

def predict_lstm(
    prices: np.ndarray,
    horizon: int = 30,
    lookback: int = 30,
    epochs: int = 20,
) -> Dict:
    """
    Predict future prices using an LSTM neural network.

    Requires TensorFlow / Keras. Returns a placeholder if unavailable.

    Args:
        prices   : Historical closing prices.
        horizon  : Number of future days to predict.
        lookback : Rolling window size.
        epochs   : Training epochs.

    Returns:
        Dict with 'predictions', 'model_name', 'mae', 'rmse'.
    """
    if not LSTM_AVAILABLE:
        return {
            "model_name": "LSTM (unavailable)",
            "predictions": [],
            "mae": None,
            "rmse": None,
            "note": "TensorFlow not installed. Run: pip install tensorflow",
        }

    prices = np.asarray(prices, dtype=float)

    # Scale to [0, 1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    X, y = _build_features(scaled, lookback)
    X_train, X_test, y_train, y_test = _train_test_split(X, y)

    # Reshape for LSTM: (samples, timesteps, features)
    X_train_3d = X_train.reshape(X_train.shape[0], lookback, 1)
    X_test_3d  = X_test.reshape(X_test.shape[0], lookback, 1)

    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train_3d, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
    )

    # Evaluate
    y_pred_scaled = model.predict(X_test_3d, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Iterative future prediction
    window = scaled[-lookback:].tolist()
    future_scaled = []
    for _ in range(horizon):
        feat = np.array(window[-lookback:]).reshape(1, lookback, 1)
        pred = float(model.predict(feat, verbose=0)[0][0])
        future_scaled.append(pred)
        window.append(pred)

    future = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten().tolist()
    future = [round(v, 2) for v in future]

    return {
        "model_name": "LSTM",
        "predictions": future,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_all_predictions(prices: np.ndarray, horizon: int = 30, lookback: int = 30) -> List[Dict]:
    """
    Run Linear Regression, Random Forest and LSTM on the same price series.

    Args:
        prices   : Historical closing prices.
        horizon  : Future prediction horizon in days.
        lookback : Rolling-window feature length.

    Returns:
        List of result dicts, one per model.
    """
    results = []
    print("[Predictor] Training Linear Regression …")
    results.append(predict_linear_regression(prices, horizon, lookback))

    print("[Predictor] Training Random Forest …")
    results.append(predict_random_forest(prices, horizon, lookback))

    if LSTM_AVAILABLE:
        print("[Predictor] Training LSTM …")
        results.append(predict_lstm(prices, horizon, lookback))
    else:
        print("[Predictor] LSTM skipped (TensorFlow not installed).")

    return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dummy_prices = np.cumsum(rng.normal(0, 1, 300)) + 1000
    results = run_all_predictions(dummy_prices, horizon=7, lookback=20)
    for r in results:
        print(f"\n{r['model_name']}")
        print(f"  Predictions : {r['predictions'][:7]}")
        print(f"  MAE={r['mae']}  RMSE={r['rmse']}")
