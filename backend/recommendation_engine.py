"""
backend/recommendation_engine.py  —  TRENDOPT
================================================
Weighted multi-factor BUY / SELL / HOLD recommendation engine.

Factor weights
--------------
    ML forecast vs current price  35 %
    RSI status                    20 %
    Moving-average trend          25 %
    Signal detector vote          15 %
    Volatility / risk adjustment   5 %

Score ∈ [-1, +1]:
    > +0.15  → BUY
    < -0.15  → SELL
    otherwise → HOLD

Confidence = min(100, |score|×100 + 40) + Sharpe bonus

Public API
----------
generate_recommendation(current, pred_results, signals,
                         latest_sig, risk) → dict
"""

import numpy as np
from typing import Dict, List, Optional


# ── Factor vote functions ─────────────────────────────────────────────────

def _v_ml(current: float, pred_7d: float, pred_30d: float) -> float:
    c7  = (pred_7d  - current) / current * 100
    c30 = (pred_30d - current) / current * 100
    score = 0.0
    if   c30 >= 7:   score += 1.0
    elif c30 >= 3:   score += 0.5
    elif c30 <= -7:  score -= 1.0
    elif c30 <= -3:  score -= 0.5
    if   c7  >= 2:   score += 0.3
    elif c7  <= -2:  score -= 0.3
    return float(np.clip(score, -1.0, 1.0))


def _v_rsi(rsi: float) -> float:
    if np.isnan(rsi): return 0.0
    if   rsi < 25:   return  1.0
    elif rsi < 35:   return  0.6
    elif rsi < 45:   return  0.2
    elif rsi > 75:   return -1.0
    elif rsi > 65:   return -0.6
    elif rsi > 55:   return -0.2
    return 0.0


def _v_ma(price: float, ma50: float, ma200: float) -> float:
    if np.isnan(ma50) or np.isnan(ma200): return 0.0
    s = 0.0
    s += 0.4 if price > ma50   else -0.4
    s += 0.4 if price > ma200  else -0.4
    s += 0.3 if ma50  > ma200  else -0.3   # golden / death cross
    return float(np.clip(s, -1.0, 1.0))


def _v_signal(sig: str) -> float:
    return {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}.get(sig, 0.0)


def _v_vol(vol: float) -> float:
    if vol > 0.50: return -0.3
    if vol > 0.35: return -0.1
    return 0.0


# ── Public function ───────────────────────────────────────────────────────

def generate_recommendation(
    current:      float,
    pred_results: List[Dict],
    signals:      Dict,
    latest_sig:   str,
    risk:         Dict,
) -> Dict:
    """
    Produce a BUY / SELL / HOLD recommendation with confidence and reasoning.

    Args:
        current      : Latest closing price.
        pred_results : List of ML model dicts from prediction_model.
        signals      : Output of signal_detector.detect_signals().
        latest_sig   : Most-recent signal string.
        risk         : Output of risk_analysis.compute_all_risk().

    Returns:
        Dict with recommendation, confidence, score, reasoning,
        pred_7d, pred_30d, change_7d_pct, change_30d_pct.
    """
    # ── Consensus forecast ────────────────────────────────────────────────
    valid = [r["predictions"] for r in pred_results
             if r.get("predictions") and len(r["predictions"]) >= 30]
    if valid:
        arr     = np.array(valid)
        pred_7d  = float(np.mean(arr[:, 6]))
        pred_30d = float(np.mean(arr[:, 29]))
    else:
        pred_7d = pred_30d = current

    c7  = (pred_7d  - current) / current * 100
    c30 = (pred_30d - current) / current * 100

    # ── Technical reads ───────────────────────────────────────────────────
    rsi_now = float(signals["rsi"][-1]) if not np.isnan(signals["rsi"][-1]) else 50.0
    ma50    = float(signals["sma50"][-1])  if not np.isnan(signals["sma50"][-1])  else current
    ma200   = float(signals["sma200"][-1]) if not np.isnan(signals["sma200"][-1]) else current
    vol     = risk.get("volatility", 25.0) / 100.0

    # ── Weighted score ────────────────────────────────────────────────────
    W = {"ml": 0.35, "rsi": 0.20, "ma": 0.25, "sig": 0.15, "vol": 0.05}
    V = {
        "ml":  _v_ml(current, pred_7d, pred_30d),
        "rsi": _v_rsi(rsi_now),
        "ma":  _v_ma(current, ma50, ma200),
        "sig": _v_signal(latest_sig),
        "vol": _v_vol(vol),
    }
    score = float(np.clip(sum(V[k] * W[k] for k in W), -1.0, 1.0))

    # ── Decision ─────────────────────────────────────────────────────────
    rec = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "HOLD"

    # ── Confidence ───────────────────────────────────────────────────────
    conf = min(100, int(abs(score) * 100 + 40))
    sh   = risk.get("sharpe_ratio", 0.0)
    if   sh > 1.5: conf = min(100, conf + 10)
    elif sh < 0:   conf = max(0,   conf - 8)

    # ── Reasoning ────────────────────────────────────────────────────────
    reasons = []
    if c30 >= 5:
        reasons.append(f"📈 ML consensus forecasts +{c30:.1f}% gain over 30 days → bullish")
    elif c30 <= -5:
        reasons.append(f"📉 ML consensus forecasts {c30:.1f}% drop over 30 days → bearish")
    else:
        reasons.append(f"➡️ ML forecast shows modest {c30:+.1f}% change over 30 days → neutral")

    if   rsi_now < 30:
        reasons.append(f"🟢 RSI={rsi_now:.0f} — deeply oversold, high reversal probability")
    elif rsi_now > 70:
        reasons.append(f"🔴 RSI={rsi_now:.0f} — overbought zone, pullback risk")
    else:
        reasons.append(f"⬜ RSI={rsi_now:.0f} — neutral momentum zone")

    if current > ma200:
        reasons.append(f"✅ Price is above 200-day MA (₹{ma200:,.2f}) — long-term uptrend")
    else:
        reasons.append(f"⚠️ Price is below 200-day MA (₹{ma200:,.2f}) — long-term downtrend")

    if ma50 > ma200 and not (np.isnan(ma50) or np.isnan(ma200)):
        reasons.append("✨ Golden Cross active — MA50 > MA200 (historically bullish)")
    elif ma50 < ma200 and not (np.isnan(ma50) or np.isnan(ma200)):
        reasons.append("💀 Death Cross active — MA50 < MA200 (historically bearish)")

    vol_lbl = "high" if vol > 0.35 else "moderate" if vol > 0.20 else "low"
    reasons.append(f"📊 Volatility={vol*100:.1f}% ({vol_lbl}) | Sharpe={sh:.2f}")

    return {
        "recommendation": rec,
        "confidence":     conf,
        "score":          round(score, 4),
        "reasoning":      reasons,
        "factor_votes":   {k: round(V[k], 3) for k in V},
        "pred_7d":        round(pred_7d, 2),
        "pred_30d":       round(pred_30d, 2),
        "change_7d_pct":  round(c7, 2),
        "change_30d_pct": round(c30, 2),
        "rsi_now":        round(rsi_now, 1),
        "ma50":           round(ma50, 2),
        "ma200":          round(ma200, 2),
    }
