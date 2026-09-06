from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_data(days: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate an hourly NI-like power-market dataset.

    The series is synthetic and intended only for reproducible development and
    demonstration. It includes exogenous forecasts that imitate information a
    forecaster could know ahead of delivery.
    """
    if days < 30:
        raise ValueError("days must be at least 30")

    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range("2025-11-01", periods=n, freq="h", tz="UTC")

    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()

    daily = 85 * np.sin(2 * np.pi * (hour - 7) / 24) + 35 * np.sin(4 * np.pi * (hour - 8) / 24)
    winter = 90 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    weekday = np.where(dow < 5, 35.0, -25.0)
    demand_noise = rng.normal(0, 28, n)
    demand_actual = 790 + daily + winter + weekday + demand_noise
    demand_actual = np.clip(demand_actual, 520, 1250)

    wind_state = np.zeros(n)
    wind_state[0] = 320
    for i in range(1, n):
        seasonal = 35 * np.cos(2 * np.pi * doy[i] / 365.25)
        wind_state[i] = 0.93 * wind_state[i - 1] + 0.07 * (300 + seasonal) + rng.normal(0, 30)
    wind_actual = np.clip(wind_state, 10, 850)

    temp_actual = (
        10
        - 5.5 * np.cos(2 * np.pi * (doy - 20) / 365.25)
        + 1.5 * np.sin(2 * np.pi * (hour - 14) / 24)
        + rng.normal(0, 1.6, n)
    )

    demand_forecast = np.clip(demand_actual + rng.normal(0, 22, n), 500, 1300)
    wind_forecast = np.clip(wind_actual + rng.normal(0, 45, n), 0, 900)
    temp_forecast = temp_actual + rng.normal(0, 1.1, n)

    gas = np.zeros(n)
    gas[0] = 56
    for i in range(1, n):
        gas[i] = 0.985 * gas[i - 1] + 0.015 * 56 + rng.normal(0, 0.7)

    peak = ((hour >= 16) & (hour <= 20)).astype(float) * 15
    net_demand = demand_actual - 0.72 * wind_actual
    price = 28 + 0.065 * net_demand + 0.55 * gas + peak + rng.normal(0, 7, n)

    spike_mask = rng.random(n) < 0.012
    price += spike_mask * rng.gamma(shape=2.0, scale=45.0, size=n)
    negative_mask = (wind_actual > 650) & (demand_actual < 690) & (rng.random(n) < 0.16)
    price[negative_mask] -= rng.uniform(35, 90, negative_mask.sum())
    price = np.clip(price, -80, 450)

    interconnector = np.clip(
        30 + 0.18 * (demand_actual - 800) - 0.10 * (wind_actual - 300) + rng.normal(0, 35, n),
        -450,
        450,
    )

    return pd.DataFrame(
        {
            "timestamp": idx,
            "price_gbp_mwh": np.round(price, 3),
            "demand_actual_mw": np.round(demand_actual, 3),
            "demand_forecast_mw": np.round(demand_forecast, 3),
            "wind_actual_mw": np.round(wind_actual, 3),
            "wind_forecast_mw": np.round(wind_forecast, 3),
            "temperature_actual_c": np.round(temp_actual, 3),
            "temperature_forecast_c": np.round(temp_forecast, 3),
            "interconnector_flow_mw": np.round(interconnector, 3),
            "source": "synthetic-ni-demo",
        }
    )


def write_demo_data(path: str | Path, days: int = 300, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_demo_data(days=days, seed=seed)
    df.to_csv(path, index=False)
    return path
