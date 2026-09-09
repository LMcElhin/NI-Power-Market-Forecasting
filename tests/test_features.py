import numpy as np

from ni_power_forecast.data.demo import (
    generate_demo_data,
)
from ni_power_forecast.features import (
    build_features,
)


def test_features_are_day_ahead_leakage_aware_for_actuals():
    df = generate_demo_data(
        days=45,
        seed=2,
    )

    X1, _ = build_features(df)

    row = 500

    changed = df.copy()

    changed.loc[
        row,
        "demand_actual_mw",
    ] += 9999

    changed.loc[
        row,
        "wind_actual_mw",
    ] += 9999

    X2, _ = build_features(changed)

    # Current-hour actual values must not enter
    # the same row's feature vector.
    a = X1.loc[row].to_numpy(dtype=float)

    b = X2.loc[row].to_numpy(dtype=float)

    assert np.allclose(
        a,
        b,
        equal_nan=True,
    )


def test_expected_price_features_exist():
    df = generate_demo_data(
        days=45,
        seed=3,
    )

    X, _ = build_features(df)

    assert "price_lag_24" in X
    assert "price_lag_25" in X
    assert "price_lag_48" in X
    assert "price_lag_168" in X
    assert "price_lag_336" in X

    assert "price_change_24_48" in X

    assert "price_change_24_168" in X

    assert "price_roll_mean_168" in X

    assert "price_vs_weekly_mean" in X


def test_expected_system_features_exist():
    df = generate_demo_data(
        days=45,
        seed=5,
    )

    X, _ = build_features(df)

    assert "demand_actual_lag_24" in X

    assert "wind_actual_lag_24" in X

    assert "net_demand_lag_24" in X

    assert "wind_share_actual_lag_24" in X

    assert "net_demand_change_24_48" in X


def test_target_forecasts_are_used_when_available():
    df = generate_demo_data(
        days=45,
        seed=7,
    )

    X, _ = build_features(df)

    assert "net_demand_forecast_mw" in X

    assert "wind_share_forecast" in X
