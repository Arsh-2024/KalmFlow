# KalmFlow: Finance‑Focused Kalman Forecasting

KalmFlow helps quants, ML engineers, and data‑driven investors rapidly prototype and compare Kalman‑style models (Standard, Extended, Unscented, Switching, Deep‑Hybrid) against financial data streams such as equities, ETFs, FX, and crypto. It emphasizes **reproducibility, benchmarking, and interactive exploration**—so it’s perfect for a **resume / portfolio project** that demonstrates both *quantitative depth* and *software engineering maturity*.

---

## ✨ Features at a Glance

* **Finance‑ready data ingestion**: Pull historical OHLCV from CSV, parquet, or live market APIs (e.g., yfinance adapter; plug‑in for Polygon/Alpaca/etc.).
* **Return & volatility modeling**: Log returns, realized volatility, EWMA vol, Kalman‑smoothed latent volatility states.
* **Kalman model zoo**: Linear KF, Extended KF (EKF), Unscented KF (UKF), Switching/Regime KF, and Deep‑assisted Kalman (neural drift/vol components).
* **Missing data tolerant**: Handles gaps, market holidays, partial trading sessions.
* **Multi‑asset batch filtering**: Forecast portfolios (tickers grouped into factor blocks) at scale.
* **Backtesting & evaluation**: Rolling‑window forecast evaluation (MAE, RMSE, MAPE, hit‑rate for direction, volatility error, VaR breach rate).
* **Model comparison**: KF vs ARIMA vs GARCH vs naive benchmarks in one command.
* **Interactive dashboards**: Plotly or Streamlit UI to explore latent states, forecasts, errors, and regime probabilities.
* **Config‑driven pipelines**: YAML experiment configs; fully reproducible runs.
* **CI‑friendly**: Deterministic seeds, data caching, unit tests, lint hooks.
---

## ❓ Why KalmFlow?

Classical forecasting methods (ARIMA, ETS) often struggle with noisy, irregular, or multi‑sensor financial signals. Deep models need a lot of data and can be opaque. **Kalman Filters** offer:

* Probabilistic state estimates (with uncertainty!)
* Natural incorporation of **latent volatility**, **drift**, **market factors**
* Elegant handling of **missing observations** and **sensor fusion** (e.g., price + volume + implied volatility)
* Extensibility to nonlinear (EKF) or non‑Gaussian regimes (approximations, particle hybrids)

KalmFlow packages these strengths into a developer‑friendly finance toolkit.

---


---

## 🧪 Minimal Example

Below trains a basic linear Kalman Filter on log‑returns for **AAPL** and forecasts 5 days ahead.

```python
from kalmflow.data import load_yfinance
from kalmflow.features import to_log_returns
from kalmflow.models import LinearKalman

# Load data
prices = load_yfinance(tickers=["AAPL"], start="2022-01-01", end="2025-01-01")
rets = to_log_returns(prices['Close'])

# Model
kf = LinearKalman(F=1.0, H=1.0, Q=1e-4, R=1e-3)  # toy scalar example
kf.fit(rets)  # optional EM style
forecast_mean, forecast_var = kf.forecast(steps=5)
print("5‑day forecast:", forecast_mean)
```

---



## ⚙️ Configuration

Experiment runs are **YAML‑driven**. Example:

```yaml
# configs/examples/spy_daily.yaml
run_name: spy_kf_daily
seed: 42

data:
  source: yfinance
  tickers: [SPY]
  start: 2022-01-01
  end: 2025-01-01
  frequency: 1D
  fields: [Close]

features:
  - name: log_returns
  - name: realized_volatility

model:
  type: linear_kf
  state_dim: 2              # [return, volatility]
  obs_dim: 1
  init_state_mean: [0, 0.01]
  init_state_cov: [[1e-2, 0], [0, 1e-4]]
  transition_matrix: [[1,0],[0,1]]
  transition_cov: [[1e-5,0],[0,1e-6]]
  observation_matrix: [[1],[0]]
  observation_cov: [[1e-4]]

forecast:
  horizon: 5
  eval_metrics: [mae, rmse, mape]

output:
  save_artifacts: true
  plots: true
```

Run it:

```bash
kalmflow run --config configs/examples/spy_daily.yaml --out runs/spy_kf/
```

---

## 🧾 Command‑Line Interface

List commands:

```bash
kalmflow --help
```


---

## 🐍 Python API

```python
from kalmflow import Experiment
exp = Experiment.from_yaml("configs/examples/spy_daily.yaml")
results = exp.run()
print(results.metrics)
```

Access latent states:

```python
latent_df = results.latent_states()
latent_df.tail()
```

---

## 🖥 Interactive Dashboards

Launch:

```bash
kalmflow ui --run runs/spy_kf/
```

Dashboard tabs:

* **Overview** – Data coverage, parameter summary
* **Latent States** – Smoothed vs filtered states (volatility, drift)
* **Forecasts** – Mean, intervals, realized values
* **Residuals & Errors** – Rolling error metrics
* **Regimes** – Markov switching probabilities (if enabled)

Include a GIF or screenshot:

```markdown
![KalmFlow Dashboard](docs/img/dashboard.png)
```

---

## 🧠 Models Supported

| Model                     | Use Case                        | Notes                     |
| ------------------------- | ------------------------------- | ------------------------- |
| **Linear Kalman Filter**  | Smooth noisy log returns        | Fast baseline             |
| **Extended KF (EKF)**     | Mild nonlinear obs models       | Jacobian required         |
| **Unscented KF (UKF)**    | Nonlinear volatility mapping    | Sigma‑point method        |
| **Switching KF**          | Regime changes (bull/bear)      | Hidden Markov over states |
| **Deep‑Assisted KF**      | Learn drift/vol functions w/ NN | Hybrid ML + state space   |
| **Benchmark ARIMA/GARCH** | Comparison baselines            | via statsmodels wrappers  |

---

## 📈 Evaluation & Backtesting

KalmFlow standardizes *fair* comparison across models:

```bash
kalmflow backtest --config configs/examples/multi_asset.yaml --folds rolling --horizon 5
```

Metrics logged per fold:

* MAE, RMSE, sMAPE
* Directional accuracy (% sign correct)
* Volatility error (% deviation vs realized)
* VaR breach rate (if risk mode enabled)

Artifacts: CSV metrics, JSON summary, interactive plot HTML.

---

## 💼 Use Cases

* **Volatility smoothing** for options desk pre‑trade analytics.
* **Signal denoising** before feeding ML alpha models.
* **Regime detection** (risk‑on / risk‑off switching).
* **Forecast aggregation** across multi‑asset portfolios.
* **Gap filling & imputation** for sparse alternative data (e.g., on‑chain metrics).

---

## 🧭 Benchmark Workflow

1. Select assets (SPY, QQQ, GLD, BTC‑USD).
2. Generate features (log rets, realized vol, macro factor overlay).
3. Train **KF**, **UKF**, **ARIMA**, **GARCH** using identical windows.
4. Rolling backtest (walk‑forward 30d train / 5d forecast).
5. Rank models by RMSE + volatility error.
6. Export best model forecasts to CSV & Plotly dashboard.
