import numpy as np

from ni_power_forecast.data.demo import generate_demo_data
from ni_power_forecast.features import build_features


def test_features_are_day_ahead_leakage_aware_for_actuals():
    df = generate_demo_data(days=45, seed=2)
    X1, _ = build_features(df)
    row = 500

    changed = df.copy()
    changed.loc[row, "demand_actual_mw"] += 9999
    changed.loc[row, "wind_actual_mw"] += 9999
    X2, _ = build_features(changed)

    # Current-hour actual values are not direct model inputs.
    a = X1.loc[row].to_numpy(dtype=float)
    b = X2.loc[row].to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True)


def test_expected_lag_features_exist():
    df = generate_demo_data(days=45, seed=3)
    X, _ = build_features(df)
    assert "price_lag_24" in X
    assert "price_lag_168" in X
    assert "net_demand_forecast_mw" in X
