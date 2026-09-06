import pandas as pd

from ni_power_forecast.merge import merge_price_and_system


def test_merge_resamples_quarter_hour_system_data():
    prices = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "price_gbp_mwh": [70.0, 72.0],
        }
    )
    system = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC"),
            "demand_actual_mw": [800] * 8,
            "demand_forecast_mw": [805] * 8,
            "wind_actual_mw": [300] * 8,
            "wind_forecast_mw": [290] * 8,
        }
    )
    out = merge_price_and_system(prices, system)
    assert len(out) == 2
    assert out.loc[0, "demand_actual_mw"] == 800
