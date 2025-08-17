import numpy as np
from filterpy.kalman import KalmanFilter

def init_kf(R=1e-3, Q=1e-5):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[0.]])
    kf.F = np.array([[1.]])
    kf.H = np.array([[1.]])
    kf.P *= 1.0
    kf.R *= R
    kf.Q *= Q
    return kf

def forecast_kf(series, steps_ahead=5, R=1e-3, Q=1e-5):
    kf = init_kf(R, Q)
    estimates = []
    for val in series:
        kf.predict()
        kf.update(val)
        estimates.append(kf.x[0][0])
    forecast = [estimates[-1]] * steps_ahead
    return estimates, forecast
