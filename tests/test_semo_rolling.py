import pandas as pd

from ni_power_forecast.data.semo_rolling import (
    complete_hours,
    select_versions,
)


def test_complete_hours_rejects_partial_hour():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-06-11",
                periods=4,
                freq="15min",
                tz="UTC",
            ),
            "wind_forecast_mw": [10, 20, 30, 40],
        }
    )

    hourly = complete_hours(frame, "wind_forecast_mw", 15)
    assert hourly.wind_forecast_mw.iloc[0] == 25

    partial = complete_hours(
        frame.iloc[:3], "wind_forecast_mw", 15
    )
    assert partial.empty


def test_selects_latest_eligible_report_covering_target():
    hour = pd.Timestamp("2026-06-11T00:00Z")
    origin = hour - pd.Timedelta(hours=24)

    def version(target, publication, value, name):
        return pd.DataFrame(
            {
                "timestamp": [target],
                "demand_forecast_mw": [value],
                "available_at": [publication],
                "metadata_at": [publication],
                "xml_at": [publication],
                "resource_name": [name],
            }
        )

    targets = pd.DatetimeIndex([hour])

    old = version(
        hour,
        origin - pd.Timedelta(hours=1),
        10,
        "old",
    )
    late = version(
        hour,
        origin + pd.Timedelta(seconds=1),
        999,
        "late",
    )
    other = version(
        hour + pd.Timedelta(hours=1),
        origin,
        20,
        "other",
    )

    result = select_versions(
        targets,
        [old, late, other],
        "load",
        "demand_forecast_mw",
    )
    assert result.demand_forecast_mw.iloc[0] == 10

    exact = version(hour, origin, 30, "exact")
    result = select_versions(
        targets,
        [old, exact],
        "load",
        "demand_forecast_mw",
    )
    assert result.demand_forecast_mw.iloc[0] == 30

    assert select_versions(
        targets,
        [late],
        "load",
        "demand_forecast_mw",
    ).empty

    assert select_versions(
        targets,
        [old],
        "load",
        "demand_forecast_mw",
        max_age_hours=0.5,
    ).empty