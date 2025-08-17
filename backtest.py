import pandas as pd
import numpy as np
from model import forecast_kf
from data import compute_log_returns

def rolling_backtest(prices, horizon=5, R=1e-3, Q=1e-5):
    rets = compute_log_returns(prices)
    results = []

    for ticker in rets.columns:
        series = rets[ticker].dropna().values
        for i in range(0, len(series) - horizon * 2, horizon):
            train = series[i:i+horizon]
            test = series[i+horizon:i+2*horizon]
            _, forecast = forecast_kf(train, steps_ahead=horizon, R=R, Q=Q)
            error = np.mean(np.abs(np.array(forecast) - test))
            results.append({
                'ticker': ticker,
                'start_idx': i,
                'mae': error
            })

    return pd.DataFrame(results)
