from __future__ import annotations

import time as time_module
import warnings
import xml.etree.ElementTree as ET
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

STATIC_REPORTS_URL = (
    "https://reports.sem-o.com/api/v1/documents/static-reports"
)
RESOURCE_URL = "https://reports.sem-o.com/documents/{resource_name}"

LOAD_DPUG_ID = "BM-010"
WIND_DPUG_ID = "BM-013"

MARKET_TIMEZONE = ZoneInfo("Europe/Dublin")


def _as_utc(
    value: str | datetime | pd.Timestamp,
) -> pd.Timestamp:
    """Convert to UTC, retaining the existing naive-as-UTC assumption."""
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def _get_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    attempts: int = 4,
) -> requests.Response:
    """Retry temporary GET failures with bounded exponential backoff."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=(15, 60),
            )
            response.raise_for_status()
            return response

        except requests.HTTPError as error:
            status = (
                error.response.status_code
                if error.response is not None
                else None
            )

            if status not in {429, 500, 502, 503, 504}:
                raise

            reason = f"HTTP {status}"

        except (
            requests.Timeout,
            requests.ConnectionError,
        ) as error:
            reason = type(error).__name__

        if attempt == attempts:
            raise RuntimeError(
                f"SEMO request failed after {attempts} attempts: "
                f"{url}; {reason}"
            )

        delay = 2 ** attempt

        print(
            f"SEMO {reason}; retry {attempt + 1}/{attempts} "
            f"in {delay}s",
            flush=True,
        )

        time_module.sleep(delay)

    raise RuntimeError("Unexpected end of retry loop")


def auction_cutoff_utc(
    trade_date: date,
) -> pd.Timestamp:
    """Return the existing daily cutoff: 11:00 Dublin time on D-1.

    This retains the original workflow's convention. It is distinct
    from the rolling delivery-time-minus-24-hours prediction origin.
    """
    auction_date = trade_date - timedelta(days=1)

    local_cutoff = datetime.combine(
        auction_date,
        time(11, 0),
        tzinfo=MARKET_TIMEZONE,
    )

    return pd.Timestamp(local_cutoff).tz_convert("UTC")


def trading_day_window_utc(
    trade_date: date,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the existing 23:00-local-to-23:00-local day window."""
    previous_date = trade_date - timedelta(days=1)

    start_local = datetime.combine(
        previous_date,
        time(23, 0),
        tzinfo=MARKET_TIMEZONE,
    )

    end_local = datetime.combine(
        trade_date,
        time(23, 0),
        tzinfo=MARKET_TIMEZONE,
    )

    return (
        pd.Timestamp(start_local).tz_convert("UTC"),
        pd.Timestamp(end_local).tz_convert("UTC"),
    )


def _fetch_metadata(
    dpug_id: str,
    publish_start: pd.Timestamp,
    publish_end: pd.Timestamp,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Fetch catalogue pages and filter publication times locally."""
    selected: list[dict[str, Any]] = []
    page = 1

    while True:
        print(
            f"{dpug_id}: requesting metadata page {page}...",
            flush=True,
        )

        params = {
            "DPuG_ID": dpug_id,
            "page": page,
            "page_size": page_size,
            "sort_by": "PublishTime",
            "order_by": "ASC",
        }

        response = _get_with_retry(
            STATIC_REPORTS_URL,
            params=params,
        )

        payload = response.json()
        items = payload.get("items", [])
        pagination = payload.get("pagination", {})

        if not items:
            print(
                f"{dpug_id}: no records on page {page}",
                flush=True,
            )
            break

        passed_end = False

        for item in items:
            publish_value = item.get("PublishTime")

            if not publish_value:
                continue

            publish_time = _as_utc(publish_value)

            if publish_start <= publish_time <= publish_end:
                record = dict(item)
                record["_publish_time_utc"] = publish_time
                selected.append(record)

            if publish_time > publish_end:
                passed_end = True

        total_pages = int(
            pagination.get("totalPages", 1)
        )

        print(
            f"{dpug_id}: page {page}/{total_pages}; "
            f"{len(selected)} reports within requested window",
            flush=True,
        )

        if passed_end or page >= total_pages:
            break

        page += 1

    return selected


def _select_latest_safe_report(
    reports: list[dict[str, Any]],
    cutoff: pd.Timestamp,
    max_age_hours: int = 24,
) -> dict[str, Any] | None:
    """Select by metadata publication time for the original daily workflow."""
    oldest_allowed = cutoff - pd.Timedelta(
        hours=max_age_hours
    )

    eligible = [
        item
        for item in reports
        if (
            oldest_allowed
            <= item["_publish_time_utc"]
            <= cutoff
        )
    ]

    if not eligible:
        return None

    return max(
        eligible,
        key=lambda item: item["_publish_time_utc"],
    )


def _fetch_resource(
    resource_name: str,
    raw_dir: str | Path | None = None,
) -> bytes:
    """Download a native SEMO XML report with caching and retries."""
    cached_path: Path | None = None

    if raw_dir is not None:
        cached_path = Path(raw_dir) / resource_name

        if cached_path.exists():
            return cached_path.read_bytes()

    response = _get_with_retry(
        RESOURCE_URL.format(resource_name=resource_name),
    )

    content = response.content

    if cached_path is not None:
        cached_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        cached_path.write_bytes(content)

    return content


def parse_load_forecast_xml(
    content: bytes | str,
) -> pd.DataFrame:
    """Parse BM-010 NI load forecasts."""
    if isinstance(content, str):
        content = content.encode("utf-8")

    root = ET.fromstring(content)
    publish_time = _as_utc(root.attrib["PublishTime"])

    records: list[dict[str, Any]] = []

    for row in root.findall("PUB_DailyLoadFcst"):
        start = row.findtext("StartTime")
        value = row.findtext("LoadForecastNI")

        if start is None or value is None:
            continue

        records.append(
            {
                "timestamp": _as_utc(start),
                "demand_forecast_mw": float(value),
                "load_forecast_published_at": publish_time,
            }
        )

    if not records:
        raise ValueError(
            "No NI load forecast rows found in BM-010 XML"
        )

    return (
        pd.DataFrame(records)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def parse_wind_forecast_xml(
    content: bytes | str,
) -> pd.DataFrame:
    """Parse BM-013 Northern Ireland wind forecasts.

    The existing report schema uses LoadForecastNI as the value
    attribute even for aggregated wind forecasts.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    root = ET.fromstring(content)
    publish_time = _as_utc(root.attrib["PublishTime"])

    records: list[dict[str, Any]] = []

    for row in root.findall("PUB_4DayAggRollWindUnitFcst"):
        start = row.attrib.get("StartTime")
        value = row.attrib.get("LoadForecastNI")

        if start is None or value is None:
            continue

        records.append(
            {
                "timestamp": _as_utc(start),
                "wind_forecast_mw": float(value),
                "wind_forecast_published_at": publish_time,
            }
        )

    if not records:
        raise ValueError(
            "No NI wind forecast rows found in BM-013 XML"
        )

    return (
        pd.DataFrame(records)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def _extract_trading_day(
    frame: pd.DataFrame,
    trade_date: date,
) -> pd.DataFrame:
    start, end = trading_day_window_utc(trade_date)

    return (
        frame.loc[
            (frame["timestamp"] >= start)
            & (frame["timestamp"] < end)
        ]
        .copy()
        .reset_index(drop=True)
    )


def _load_to_hourly(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Original daily-workflow aggregation of load forecasts."""
    return (
        frame.set_index("timestamp")[["demand_forecast_mw"]]
        .resample("h")
        .mean()
        .reset_index()
    )


def _wind_to_hourly(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Original daily-workflow aggregation of wind forecasts."""
    return (
        frame.set_index("timestamp")[["wind_forecast_mw"]]
        .resample("h")
        .mean()
        .reset_index()
    )


def fetch_point_in_time_forecasts(
    date_from: str,
    date_to: str,
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Retain the original daily-cutoff forecast workflow.

    Dates refer to the existing trading-day labels.

    This function is preserved for CLI compatibility. For rolling
    24-hour selection, use the separate semo_rolling module instead.
    """
    first_trade_date = date.fromisoformat(date_from)
    final_trade_date = date.fromisoformat(date_to)

    if final_trade_date < first_trade_date:
        raise ValueError("date_to must be >= date_from")

    first_cutoff = auction_cutoff_utc(first_trade_date)
    final_cutoff = auction_cutoff_utc(final_trade_date)

    metadata_start = first_cutoff - pd.Timedelta(hours=24)
    metadata_end = final_cutoff

    load_reports = _fetch_metadata(
        LOAD_DPUG_ID,
        metadata_start,
        metadata_end,
    )

    wind_reports = _fetch_metadata(
        WIND_DPUG_ID,
        metadata_start,
        metadata_end,
    )

    pieces: list[pd.DataFrame] = []
    current = first_trade_date

    while current <= final_trade_date:
        cutoff = auction_cutoff_utc(current)

        load_report = _select_latest_safe_report(
            load_reports,
            cutoff,
        )
        wind_report = _select_latest_safe_report(
            wind_reports,
            cutoff,
        )

        if load_report is None or wind_report is None:
            warnings.warn(
                f"No complete forecast pair for trade date {current}",
                stacklevel=2,
            )
            current += timedelta(days=1)
            continue

        load_xml = _fetch_resource(
            load_report["ResourceName"],
            raw_dir=raw_dir,
        )
        wind_xml = _fetch_resource(
            wind_report["ResourceName"],
            raw_dir=raw_dir,
        )

        load = parse_load_forecast_xml(load_xml)
        wind = parse_wind_forecast_xml(wind_xml)

        load = _extract_trading_day(load, current)
        wind = _extract_trading_day(wind, current)

        if load.empty or wind.empty:
            warnings.warn(
                f"Selected reports do not cover trade date {current}",
                stacklevel=2,
            )
            current += timedelta(days=1)
            continue

        load_pub = load_report["_publish_time_utc"]
        wind_pub = wind_report["_publish_time_utc"]

        load_hourly = _load_to_hourly(load)
        wind_hourly = _wind_to_hourly(wind)

        combined = load_hourly.merge(
            wind_hourly,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        )

        combined["trade_date"] = current.isoformat()
        combined["forecast_cutoff_utc"] = cutoff
        combined["load_forecast_published_at"] = load_pub
        combined["wind_forecast_published_at"] = wind_pub

        pieces.append(combined)
        current += timedelta(days=1)

    if not pieces:
        raise ValueError(
            "No point-in-time SEMO forecast observations produced"
        )

    out = pd.concat(pieces, ignore_index=True)

    out["net_demand_forecast_mw"] = (
        out["demand_forecast_mw"]
        - out["wind_forecast_mw"]
    )

    out["wind_share_forecast"] = (
        out["wind_forecast_mw"]
        / out["demand_forecast_mw"].clip(lower=1)
    )

    return (
        out.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )