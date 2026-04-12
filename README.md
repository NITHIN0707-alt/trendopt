# TRENDOPT — Intelligent Stock Buy-Sell Strategy Analyzer & Price Prediction System

> A university-grade **Design and Analysis of Algorithms (DAA)** + **Machine Learning** project that analyses historical stock prices, compares three algorithmic trading strategies, detects buy/sell signals through technical indicators, predicts future prices using ML models, and delivers a **BUY / SELL / HOLD** recommendation — all through a production-quality Streamlit dashboard with SQLite persistence.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [DAA Algorithms Explained](#daa-algorithms-explained)
6. [Time & Space Complexity Comparison](#time--space-complexity-comparison)
7. [Technical Indicators](#technical-indicators)
8. [Machine Learning Models](#machine-learning-models)
9. [Backtesting Module](#backtesting-module)
10. [Risk Analysis](#risk-analysis)
11. [Recommendation Engine](#recommendation-engine)
12. [Database & Storage](#database--storage)
13. [Extra Features](#extra-features-beyond-spec)
14. [Example Output](#example-output)
15. [Future Improvements](#future-improvements)
16. [Limitations](#limitations)

---

## Project Overview

TRENDOPT is a full-stack financial analysis system that combines:

| Domain | Technologies Used |
|--------|------------------|
| **DAA** | Brute Force O(n²), Greedy O(n), Dynamic Programming O(n) |
| **Machine Learning** | Linear Regression, Random Forest, Gradient Boosting, optional LSTM |
| **Financial Analysis** | RSI, MACD, Bollinger Bands, MA50/MA200, ATR, Sharpe, VaR |
| **Backtesting** | Equity curve, win-rate, max drawdown, Sharpe ratio |
| **Visualisation** | Plotly interactive charts, matplotlib export, Streamlit dashboard |
| **Database** | SQLite — search history, analysis results, predictions, portfolio |
| **Web UI** | Streamlit with 8 interactive tabs |

---

## Problem Statement

Stock markets generate enormous amounts of time-series data every day. Manually identifying optimal buy/sell points, comparing trading strategy efficiency, and predicting future prices is both time-consuming and error-prone.

**TRENDOPT solves this by:**
1. Automatically downloading 1 year of historical OHLCV data for any stock
2. Applying three DAA-based profit-maximisation algorithms and measuring their performance
3. Detecting buy/sell entry points using 5 signal strategies (local extrema, RSI, MA crossover, Bollinger Bands, MACD)
4. Training 3–4 ML models to forecast 7-day and 30-day prices
5. Generating a weighted **BUY / SELL / HOLD** recommendation with a confidence score
6. Backtesting the strategy and computing real risk metrics

---

## Project Structure

```
TRENDOPT/
│
├── backend/
│   ├── data_fetcher.py          ← yfinance download + GBM synthetic fallback
│   ├── brute.py                 ← Brute-Force O(n²) profit optimisation
│   ├── greedy.py                ← Greedy O(n) unlimited-transaction strategy
│   ├── dp.py                    ← Dynamic Programming O(n) with 1-day cooldown
│   ├── indicator_module.py      ← SMA, EMA, RSI, MACD, Bollinger Bands, ATR
│   ├── signal_detector.py       ← Multi-strategy buy/sell signal detection
│   ├── prediction_model.py      ← Linear Regression, RF, GB, optional LSTM
│   ├── backtesting.py           ← Strategy simulation + equity curve
│   ├── risk_analysis.py         ← Volatility, Sharpe, Sortino, VaR, Drawdown
│   └── recommendation_engine.py ← Weighted multi-factor BUY/SELL/HOLD engine
│
├── visualization/
│   └── plot_graphs.py           ← All Plotly interactive + matplotlib static charts
│
├── frontend/
│   └── dashboard.py             ← Streamlit web dashboard (8 tabs)
│
├── database/
│   └── db_manager.py            ← SQLite: search history, analyses, predictions
│
├── output/                      ← Generated PNG dashboards
│
├── main.py                      ← CLI entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Optional: LSTM support
pip install tensorflow
```

### 2. Launch Streamlit dashboard (recommended)

```bash
streamlit run frontend/dashboard.py
```

Open your browser at **http://localhost:8501**

### 3. CLI mode

```bash
# Default — RELIANCE.NS, 30-day horizon
python main.py

# Custom ticker and horizon
python main.py --ticker TCS.NS --horizon 30
python main.py --ticker AAPL --horizon 7
python main.py --ticker MSFT --horizon 30 --strategy rsi

# All flags
python main.py --help
```

### Supported tickers (friendly names)

| Name | Symbol | Name | Symbol |
|------|--------|------|--------|
| reliance | RELIANCE.NS | tcs | TCS.NS |
| infosys | INFY.NS | wipro | WIPRO.NS |
| sbi | SBIN.NS | hdfc | HDFCBANK.NS |
| apple | AAPL | microsoft | MSFT |
| tesla | TSLA | nvidia | NVDA |
| google | GOOGL | amazon | AMZN |

---

## DAA Algorithms Explained

### 1️⃣ Brute Force — `backend/brute.py`

**Strategy:** Test every possible (buy day i, sell day j) pair where j > i.

```python
for i in range(n):
    for j in range(i+1, n):
        profit = price[j] - price[i]
        best = max(best, profit)
```

- Guaranteed optimal for a **single** buy-sell transaction
- Capped at 22 days by default to prevent timeout on large arrays
- Used as a **correctness baseline** for verifying other algorithms
- **Time:** O(n²)  **Space:** O(1)

---

### 2️⃣ Greedy Algorithm — `backend/greedy.py`

**Strategy:** Accumulate every positive day-over-day price movement.

```python
profit += max(0, prices[i] - prices[i-1])
```

- Equivalent to buying at every local minimum and selling at every local maximum
- Optimal for **unlimited transactions** with no constraints
- Fastest algorithm — single pass through the data
- **Time:** O(n)  **Space:** O(1)

---

### 3️⃣ Dynamic Programming — `backend/dp.py`

**Strategy:** Three rolling state variables track the maximum profit under a **1-day cooldown** constraint after each sell.

```
States:
    hold[i] = max profit while HOLDING on day i
    sold[i] = max profit just after SELLING on day i
    rest[i] = max profit during COOLDOWN on day i

Recurrences:
    hold[i] = max(hold[i-1], rest[i-1] - price[i])
    sold[i] = hold[i-1] + price[i]
    rest[i] = max(rest[i-1], sold[i-1])
```

- Models **realistic trading constraints**
- Rolling variables — O(1) space (no arrays stored)
- **Time:** O(n)  **Space:** O(1)

---

## Time & Space Complexity Comparison

| Algorithm | Time | Space | Transactions | Constraint | Use Case |
|-----------|------|-------|-------------|------------|----------|
| Brute Force | O(n²) | O(1) | Single | None (capped) | Baseline correctness |
| Greedy | O(n) | O(1) | Unlimited | None | Max theoretical profit |
| Dynamic Prog. | O(n) | O(1) | Unlimited | 1-day cooldown | Realistic constrained trading |

**Key insight:** Greedy and DP achieve the same time complexity O(n) but model different trading realities. DP is more practical; Greedy maximises theoretical profit.

---

## Technical Indicators

All implemented in `backend/indicator_module.py`:

| Indicator | Formula / Logic | Signal Interpretation |
|-----------|----------------|----------------------|
| **SMA 50** | Mean of last 50 closes | Price > MA50 → bullish short-term |
| **SMA 200** | Mean of last 200 closes | Price > MA200 → long-term uptrend |
| **RSI (14)** | Wilder smoothed gain/loss ratio → [0,100] | < 30 oversold (BUY), > 70 overbought (SELL) |
| **MACD (12,26,9)** | EMA12 − EMA26, signal = EMA9 of MACD | Line crosses above signal → BUY |
| **Bollinger Bands** | MA20 ± 2σ | Price < lower → BUY, > upper → SELL |
| **ATR (14)** | Average True Range | Measures absolute volatility |

### Signal Detection Strategies (`backend/signal_detector.py`)

| Strategy | Logic |
|----------|-------|
| `local_extrema` | Local minimum → BUY; local maximum → SELL |
| `rsi` | RSI < 30 → BUY; RSI > 70 → SELL |
| `ma_crossover` | Golden Cross (MA50>MA200) → BUY; Death Cross → SELL |
| `bb` | Price < BB lower → BUY; > BB upper → SELL |
| `macd` | MACD crosses above signal line → BUY; below → SELL |
| `combined` | Union of all five strategies (default) |

---

## Machine Learning Models

All implemented in `backend/prediction_model.py`.

### Feature Engineering

A **rolling-window supervised** approach:
```
X[i] = [close[i-lookback], ..., close[i-1]]   ← feature vector (30 days)
y[i] = close[i]                                 ← target (next day)
```

Prediction is **iterative**: each model's output is appended to the window and used as input for the next step.

### Models

| Model | Library | Strengths | Weaknesses |
|-------|---------|-----------|------------|
| **Linear Regression** | scikit-learn | Fast, interpretable, low variance | Can't model non-linearity |
| **Random Forest** | scikit-learn | Non-linear, robust to noise | May underfit long horizons |
| **Gradient Boosting** | scikit-learn | Strong learner, high accuracy | Slower training |
| **LSTM** (optional) | TensorFlow | Captures temporal dependencies | Requires TF, slow to train |

### Metrics computed on held-out test set

| Metric | Meaning |
|--------|---------|
| MAE | Mean Absolute Error — average ₹ deviation |
| RMSE | Root Mean Squared Error — penalises large errors |
| R² | Coefficient of determination — % variance explained |
| MAPE | Mean Absolute Percentage Error |

---

## Backtesting Module

`backend/backtesting.py` simulates the combined signal strategy on historical data:

- **BUY** at closing price on each BUY signal day (invests full available cash)
- **SELL** at closing price on each SELL signal day
- Open positions are closed at the last available price
- Compared against a **buy-and-hold benchmark**

### Metrics computed

| Metric | Description |
|--------|-------------|
| Total return % | Portfolio growth over the year |
| Win rate | Fraction of profitable trades |
| Loss rate | Fraction of losing trades |
| Max drawdown | Largest peak-to-trough loss |
| Sharpe ratio | Risk-adjusted return (annualised) |
| Best / worst trade | Single-trade extremes |
| Strategy accuracy | Same as win rate |

---

## Risk Analysis

`backend/risk_analysis.py` computes investment risk metrics on the closing price series:

| Metric | Formula |
|--------|---------|
| Annualised Volatility | `std(log returns) × √252` |
| Sharpe Ratio | `mean(excess returns) / std × √252` |
| Sortino Ratio | Sharpe using only downside deviation |
| Max Drawdown | `min((price - peak) / peak)` |
| VaR 95% (1-day) | `μ − 1.645σ` of log returns |
| Expected Return | `mean(log returns) × 252` |

**Risk levels:** Low < 20% vol · Moderate 20–35% · High > 35%

---

## Recommendation Engine

`backend/recommendation_engine.py` combines five weighted factors:

| Factor | Weight | Bullish Condition |
|--------|--------|-------------------|
| ML forecast vs current | 35% | 30-day forecast ≥ +7% |
| RSI status | 20% | RSI < 30 (oversold) |
| Moving-average trend | 25% | Price > MA50 > MA200 |
| Signal detector vote | 15% | Latest signal = BUY |
| Volatility adjustment | 5% | Low volatility bonus |

**Score → Decision:**
```
Score ∈ [-1, +1]
  > +0.15  → BUY
  < -0.15  → SELL
  otherwise → HOLD
```

**Confidence** = `min(100, |score| × 100 + 40)` + Sharpe bonus (±8–10%)

---

## Database & Storage

`database/db_manager.py` — SQLite tables:

| Table | Contents |
|-------|----------|
| `search_history` | Ticker, timestamp, live/synthetic flag |
| `analysis_results` | Price, recommendation, confidence, Sharpe, forecasts |
| `predictions` | Model name, horizon, JSON-serialised predictions |
| `portfolio` | User watchlist tickers |

All data viewable in the **🕘 History** and **🗂️ Portfolio** dashboard tabs.

---

## Extra Features (Beyond Spec)

| Feature | Where |
|---------|-------|
| 🕯️ Interactive candlestick chart | Tab 1 — Price & Signals |
| 📊 Volume profile (horizontal) | Tab 1 |
| 📉 Bollinger Band squeeze meter | Tab 4 — Indicators |
| 📈 MACD chart | Tab 4 |
| 🧪 Equity curve vs buy-and-hold | Tab 5 — Backtesting |
| 📋 Trade log with CSV export | Tab 5 |
| 🎯 Factor-vote breakdown | Tab 6 — Recommendation |
| 🕸️ Risk radar chart | Tab 6 |
| 🗂️ Multi-stock portfolio watchlist | Tab 7 — Portfolio |
| 🕘 SQLite search history | Tab 8 — History |
| 🔔 Real-time alerts (RSI, MA cross, BB) | Sidebar |
| ⬇️ Export forecast + trades as CSV | Tabs 2, 5 |
| Sortino ratio | Risk Analysis |
| VaR 95% | Risk Analysis |
| Offline / sandboxed GBM fallback | data_fetcher.py |

---

## Example Output

```
==================================================================
  TRENDOPT — Intelligent Stock Analyzer & Predictor
  Ticker: RELIANCE.NS   Horizon: 30d   Strategy: combined
==================================================================

[1/7] Fetching historical data …
  Source       : SYNTHETIC (demo)
  Period       : 2025-04-24 → 2026-04-10
  Days         : 252
  Price        : ₹1,499.70  (▲2.35%)
  52W High/Low : ₹2,130.36 / ₹1,158.43

[2/7] Running DAA algorithms …
  Algorithm          Profit       Time      Complexity
  Brute Force        ₹71.88      0.000088s  O(n²)
  Greedy             ₹3,209.50   0.000282s  O(n)
  Dynamic Prog.      ₹2,831.86   0.000189s  O(n)

[3/7] Detecting buy/sell signals …
  BUY  signals : 80
  SELL signals : 94
  Latest       : HOLD

[4/7] Training ML models …
  [Linear Regression]  7d=₹1,562  30d=₹1,651  MAE=28.53  R²=0.91
  [Random Forest]      7d=₹1,663  30d=₹1,785  MAE=122.36 R²=-0.27
  [Gradient Boosting]  7d=₹1,630  30d=₹1,878  MAE=107.54 R²=0.02

[5/7] Backtesting …
  Total return : 347.99%
  Win rate     : 93.4%   (61 trades)
  Max drawdown : -6.28%
  Sharpe ratio : 8.84

[6/7] Risk metrics …
  Volatility   : 29.05%  (Moderate 🟡)
  Sharpe ratio : 0.52
  VaR 95%      : 2.93%
  Exp. return  : 21.04% p.a.

[7/7] Recommendation …
  ╔══════════════════════════╗
  ║  🟡  HOLD │ Conf: 54%  ║
  ║  7d  → ₹1,618           ║
  ║  30d → ₹1,772           ║
  ╚══════════════════════════╝
```

---

## Future Improvements

- **Sentiment analysis** — integrate news headlines and Reddit sentiment
- **Options pricing** — Black-Scholes model for derivative strategies
- **Portfolio optimisation** — Markowitz mean-variance, Sharpe-maximising allocation
- **Live WebSocket feed** — real-time tick data via WebSocket
- **Transformer models** — Temporal Fusion Transformer for multi-step forecasting
- **Paper trading** — simulate live trades without real money
- **Docker** — containerised deployment with Nginx reverse proxy
- **Multi-timeframe** — 1-week and 5-minute chart analysis modes

---

## Limitations

- Price predictions degrade over longer horizons due to iterative error accumulation
- Models trained on historical patterns cannot anticipate news events or earnings surprises
- Live data requires an internet connection; offline mode falls back to GBM-simulated prices
- Brute-force algorithm is capped at 22 days; the true O(2ⁿ) recursive form would timeout
- Recommendation engine is educational — weights are heuristic, not optimised from live P&L

---

*TRENDOPT is designed for educational and academic demonstration purposes only.*
*It does not constitute financial advice. Always consult a qualified financial advisor before investing.*
