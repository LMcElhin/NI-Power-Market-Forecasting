from datetime import date

import pandas as pd

from ni_power_forecast.data.semo_forecasts import (
    _select_latest_safe_report,
    auction_cutoff_utc,
    parse_load_forecast_xml,
    parse_wind_forecast_xml,
    trading_day_window_utc,
)

LOAD_XML = b"""
<OutboundData
    DatasetName="PUB_DailyLoadFcst"
    PublishTime="2026-06-10T09:50:05">
  <PUB_DailyLoadFcst ROW="1">
    <DeliveryDate>2026-06-10</DeliveryDate>
    <TradeDate>2026-06-11</TradeDate>
    <StartTime>2026-06-10T22:00:00</StartTime>
    <EndTime>2026-06-10T22:30:00</EndTime>
    <LoadForecastROI>3500</LoadForecastROI>
    <LoadForecastNI>700</LoadForecastNI>
    <AggregatedForecast>4200</AggregatedForecast>
  </PUB_DailyLoadFcst>
</OutboundData>
"""


WIND_XML = b"""
<OutboundData
    DatasetName="PUB_4DayAggRollWindUnitFcst"
    PublishTime="2026-06-10T06:15:21">
  <PUB_4DayAggRollWindUnitFcst
      ROW="1"
      TradeDate="2026-06-11"
      DeliveryDate="2026-06-10"
      StartTime="2026-06-10T22:00:00"
      EndTime="2026-06-10T22:15:00"
      LoadForecastROI="700.0"
      LoadForecastNI="125.5"
      AggregatedForecast="825.5"/>
</OutboundData>
"""


def test_summer_auction_cutoff_is_1000_utc():
    cutoff = auction_cutoff_utc(
        date(
            2026,
            6,
            11,
        )
    )

    assert cutoff == pd.Timestamp("2026-06-10T10:00:00Z")


def test_summer_trading_day_window():
    start, end = trading_day_window_utc(
        date(
            2026,
            6,
            11,
        )
    )

    assert start == pd.Timestamp("2026-06-10T22:00:00Z")

    assert end == pd.Timestamp("2026-06-11T22:00:00Z")


def test_load_parser_extracts_ni_forecast():
    frame = parse_load_forecast_xml(LOAD_XML)

    assert len(frame) == 1

    assert (
        frame.loc[
            0,
            "demand_forecast_mw",
        ]
        == 700.0
    )

    assert frame.loc[
        0,
        "timestamp",
    ] == pd.Timestamp("2026-06-10T22:00:00Z")


def test_wind_parser_extracts_ni_forecast():
    frame = parse_wind_forecast_xml(WIND_XML)

    assert len(frame) == 1

    assert (
        frame.loc[
            0,
            "wind_forecast_mw",
        ]
        == 125.5
    )


def test_report_after_gate_closure_is_rejected():
    reports = [
        {
            "ResourceName": "safe.xml",
            "_publish_time_utc": (pd.Timestamp("2026-06-10T09:50:05Z")),
        },
        {
            "ResourceName": "late.xml",
            "_publish_time_utc": (pd.Timestamp("2026-06-10T10:00:15Z")),
        },
    ]

    cutoff = pd.Timestamp("2026-06-10T10:00:00Z")

    chosen = _select_latest_safe_report(
        reports,
        cutoff,
    )

    assert chosen is not None

    assert chosen["ResourceName"] == "safe.xml"
