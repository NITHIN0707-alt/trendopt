"""
backend/prediction_model.py  —  TRENDOPT
==========================================
Machine-learning price prediction.

Models
------
1. Linear Regression     — fast, interpretable baseline
2. Random Forest         — non-linear ensemble
3. Gradient Boosting     — strong tree-based learner
4. LSTM                  — sequence model (requires TensorFlow)

All models use a rolling-window supervised frame:
    X[i] = [close[i-lookback], ..., close[i-1]]
    y[i] = close[i]

Public API
----------
run_all_predictions(prices, horizon, lookback) → list[dict]
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    _LSTM = True
except ImportError:
    _LSTM = False


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_xy(prices: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(prices)):
        X.append(prices[i - lookback: i])
        y.append(prices[i])
    return np.array(X), np.array(y)


def _split(X, y, ratio: float = 0.20):
    k = int(len(X) * (1 - ratio))
    return X[:k], X[k:], y[:k], y[k:]


def _metrics(yt, yp) -> dict:
    return {
        "mae":  round(float(mean_absolute_error(yt, yp)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(yt, yp))), 2),
        "r2":   round(float(r2_score(yt, yp)), 4),
        "mape": round(float(np.mean(np.abs((yt - yp) / (yt + 1e-9))) * 100), 2),
    }


def _roll_predict(model, seed: list, horizon: int, lookback: int) -> List[float]:
    """Iteratively predict ``horizon`` future prices."""
    window = list(seed[-lookback:])
    out    = []
    for _ in range(horizon):
        feat = np.array(window[-lookback:]).reshape(1, -1)
        p    = float(model.predict(feat)[0])
        out.append(round(p, 2))
        window.append(p)
    return out


# ── Individual models ──────────────────────────────────────────────────────

def _linear_regression(prices, horizon, lookback) -> dict:
    X, y                    = _build_xy(prices, lookback)
    Xtr, Xte, ytr, yte     = _split(X, y)
    m                       = LinearRegression().fit(Xtr, ytr)
    met                     = _metrics(yte, m.predict(Xte))
    return {"model_name": "Linear Regression", "predictions": _roll_predict(m, prices.tolist(), horizon, lookback), **met}


def _random_forest(prices, horizon, lookback) -> dict:
    X, y                    = _build_xy(prices, lookback)
    Xtr, Xte, ytr, yte     = _split(X, y)
    m                       = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    met                     = _metrics(yte, m.predict(Xte))
    return {"model_name": "Random Forest", "predictions": _roll_predict(m, prices.tolist(), horizon, lookback), **met}


def _gradient_boosting(prices, horizon, lookback) -> dict:
    X, y                    = _build_xy(prices, lookback)
    Xtr, Xte, ytr, yte     = _split(X, y)
    m                       = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)
    met                     = _metrics(yte, m.predict(Xte))
    return {"model_name": "Gradient Boosting", "predictions": _roll_predict(m, prices.tolist(), horizon, lookback), **met}


def _lstm(prices, horizon, lookback, epochs=20) -> dict:
    if not _LSTM:
        return {"model_name": "LSTM (unavailable)", "predictions": [],
                "mae": None, "rmse": None, "r2": None, "mape": None,
                "note": "pip install tensorflow"}
    sc    = MinMaxScaler()
    sc_p  = sc.fit_transform(prices.reshape(-1, 1)).flatten()
    X, y  = _build_xy(sc_p, lookback)
    Xtr, Xte, ytr, yte = _split(X, y)
    Xtr3, Xte3 = Xtr.reshape(-1, lookback, 1), Xte.reshape(-1, lookback, 1)
    m = Sequential([LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
                    Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    m.fit(Xtr3, ytr, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)
    yp   = sc.inverse_transform(m.predict(Xte3, verbose=0)).flatten()
    yt   = sc.inverse_transform(yte.reshape(-1, 1)).flatten()
    met  = _metrics(yt, yp)
    win  = sc_p[-lookback:].tolist()
    pred_sc = []
    for _ in range(horizon):
        feat = np.array(win[-lookback:]).reshape(1, lookback, 1)
        p    = float(m.predict(feat, verbose=0)[0][0])
        pred_sc.append(p); win.append(p)
    pred = [round(v, 2) for v in sc.inverse_transform(np.array(pred_sc).reshape(-1, 1)).flatten()]
    return {"model_name": "LSTM", "predictions": pred, **met}


# ── Public API ─────────────────────────────────────────────────────────────

def run_all_predictions(
    prices:   np.ndarray,
    horizon:  int = 30,
    lookback: int = 30,
) -> List[Dict]:
    """
    Train all available models and return a list of result dicts.

    Each dict contains:
        model_name, predictions (list of length horizon),
        mae, rmse, r2, mape
    """
    p = np.asarray(prices, dtype=float)
    results = [
        _linear_regression(p, horizon, lookback),
        _random_forest(p, horizon, lookback),
        _gradient_boosting(p, horizon, lookback),
    ]
    if _LSTM:
        results.append(_lstm(p, horizon, lookback))
    return results


def consensus_forecast(pred_results: List[Dict], horizon: int) -> np.ndarray:
    """
    Average predictions across all models that have valid forecasts.

    Returns a 1-D array of length ``horizon``.
    """
    valid = [r["predictions"] for r in pred_results
             if r.get("predictions") and len(r["predictions"]) >= horizon]
    if not valid:
        return np.array([])
    return np.mean(np.array(valid)[:, :horizon], axis=0)
