"""Select forecasts available at each rolling 24-hour prediction origin."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from ni_power_forecast.data.semo_forecasts import (
    LOAD_DPUG_ID,
    WIND_DPUG_ID,
    _as_utc,
    _fetch_metadata,
    _fetch_resource,
    parse_load_forecast_xml,
    parse_wind_forecast_xml,
)


def complete_hours(frame, value_column, minutes):
    """Average only hours containing every expected subinterval."""
    frame = frame.copy()

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
    )

    if (
        frame.timestamp.isna().any()
        or frame.timestamp.duplicated().any()
    ):
        raise ValueError(
            "Missing or duplicate forecast timestamps"
        )

    expected = set(range(0, 60, minutes))
    records = []

    for hour, group in frame.groupby(
        frame.timestamp.dt.floor("h")
    ):
        offsets = (
            group.timestamp - hour
        ).dt.total_seconds() / 60

        values = pd.to_numeric(
            group[value_column],
            errors="coerce",
        )

        if (
            set(offsets) == expected
            and np.isfinite(values).all()
        ):
            records.append(
                {
                    "timestamp": hour,
                    value_column: float(values.mean()),
                }
            )

    return pd.DataFrame(
        records,
        columns=["timestamp", value_column],
    )


def select_versions(
    targets,
    versions,
    prefix,
    value_column,
    max_age_hours=72,
):
    """Choose the latest eligible version covering each target hour."""

    columns = [
        "timestamp",
        value_column,
        f"{prefix}_forecast_published_at",
        f"{prefix}_metadata_published_at",
        f"{prefix}_xml_published_at",
        f"{prefix}_resource_name",
    ]

    rows = []

    if versions:
        candidates = pd.concat(
            versions,
            ignore_index=True,
        )

        candidates["origin"] = (
            candidates.timestamp
            - pd.Timedelta(hours=24)
        )

        candidates = candidates.loc[
            candidates.timestamp.isin(targets)
            & (
                candidates.available_at
                <= candidates.origin
            )
            & (
                candidates.available_at
                >= candidates.origin
                - pd.Timedelta(hours=max_age_hours)
            )
        ]

        candidates = (
            candidates.sort_values(
                [
                    "timestamp",
                    "available_at",
                    "resource_name",
                ]
            )
            .drop_duplicates(
                "timestamp",
                keep="last",
            )
        )

        for row in candidates.itertuples(index=False):
            rows.append(
                {
                    "timestamp": row.timestamp,
                    value_column: getattr(
                        row,
                        value_column,
                    ),
                    f"{prefix}_forecast_published_at": (
                        row.available_at
                    ),
                    f"{prefix}_metadata_published_at": (
                        row.metadata_at
                    ),
                    f"{prefix}_xml_published_at": (
                        row.xml_at
                    ),
                    f"{prefix}_resource_name": (
                        row.resource_name
                    ),
                }
            )

    result = pd.DataFrame(
        rows,
        columns=columns,
    )

    time_columns = ["timestamp"] + [
        column
        for column in columns
        if column.endswith("_at")
    ]

    for column in time_columns:
        result[column] = pd.to_datetime(
            result[column],
            utc=True,
        )

    return result


def _write_skip_log(
    skipped_reports,
    raw_dir,
    prefix,
):
    """Write details of reports that could not be processed."""

    if not skipped_reports:
        return

    log_dir = Path(raw_dir)
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        log_dir
        / f"{prefix}_skipped_reports.csv"
    )

    pd.DataFrame(skipped_reports).to_csv(
        log_path,
        index=False,
    )

    print(
        f"{prefix}: wrote skip log to {log_path}",
        flush=True,
    )


def add_cached_reports(
    reports,
    raw_dir,
    dpug,
    start,
    end,
):
    """Add valid cached reports missing from the SEMO metadata API."""

    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        return reports

    if dpug == LOAD_DPUG_ID:
        pattern = "PUB_DailyLoadFcst_*.xml"

    elif dpug == WIND_DPUG_ID:
        pattern = "PUB_4DayAggRollWindUnitFcst_*.xml"

    else:
        return reports

    existing = {
        report.get("ResourceName")
        for report in reports
    }

    added = 0

    for path in raw_dir.glob(pattern):
        if path.name in existing:
            continue

        try:
            root = ET.fromstring(
                path.read_bytes()
            )

            publish_value = root.attrib.get(
                "PublishTime"
            )

            if publish_value is None:
                continue

            publish_time = _as_utc(
                publish_value
            )

        except (
            ET.ParseError,
            ValueError,
            KeyError,
        ):
            continue

        if not (
            start
            <= publish_time
            <= end
        ):
            continue

        reports.append(
            {
                "ResourceName": path.name,

                # For historical files absent from the current
                # metadata catalogue, use the original XML
                # publication time as the metadata time too.
                "_publish_time_utc": publish_time,

                "_source": "local_cache",
            }
        )

        existing.add(path.name)
        added += 1

    reports.sort(
        key=lambda report: (
            report["_publish_time_utc"],
            report["ResourceName"],
        )
    )

    print(
        f"{dpug}: added {added} historical "
        f"reports from local cache",
        flush=True,
    )

    return reports


def fetch_rolling_forecasts(
    targets,
    raw_dir,
    max_age_hours=72,
):
    if max_age_hours <= 0:
        raise ValueError(
            "max_age_hours must be positive"
        )

    targets = pd.DatetimeIndex(
        pd.to_datetime(
            targets,
            utc=True,
        )
    ).sort_values()

    if (
        len(targets) == 0
        or targets.hasnans
        or targets.has_duplicates
    ):
        raise ValueError(
            "Need nonempty, unique, valid target timestamps"
        )

    if not (
        targets == targets.floor("h")
    ).all():
        raise ValueError(
            "Targets must be hourly interval starts"
        )

    origins = (
        targets
        - pd.Timedelta(hours=24)
    )

    start = (
        origins.min()
        - pd.Timedelta(hours=max_age_hours)
    )

    end = origins.max()

    print(
        "Rolling forecast metadata window:",
        start,
        "to",
        end,
        flush=True,
    )

    output = pd.DataFrame(
        {
            "timestamp": targets,
            "forecast_cutoff_utc": origins,
        }
    )

    specifications = [
        (
            LOAD_DPUG_ID,
            "load",
            "demand_forecast_mw",
            30,
            parse_load_forecast_xml,
        ),
        (
            WIND_DPUG_ID,
            "wind",
            "wind_forecast_mw",
            15,
            parse_wind_forecast_xml,
        ),
    ]

    for (
        dpug,
        prefix,
        value,
        minutes,
        parse_report,
    ) in specifications:

        reports = _fetch_metadata(
            dpug,
            start,
            end,
        )

        # -----------------------------------------------------
        # Recover historical files that exist locally but are
        # no longer returned by the current SEMO metadata API.
        # -----------------------------------------------------

        reports = add_cached_reports(
            reports,
            raw_dir,
            dpug,
            start,
            end,
        )

        versions = []
        seen = set()

        skipped_reports = []

        duplicate_count = 0
        download_failures = 0
        parse_failures = 0
        publication_failures = 0
        no_target_hours = 0
        accepted_reports = 0

        print(
            f"{prefix}: "
            f"{len(reports)} metadata/cache records",
            flush=True,
        )

        for number, report in enumerate(
            reports,
            start=1,
        ):
            resource = report.get(
                "ResourceName"
            )

            metadata_value = report.get(
                "_publish_time_utc"
            )

            # -------------------------------------------------
            # Validate metadata
            # -------------------------------------------------

            if (
                resource is None
                or metadata_value is None
            ):
                skipped_reports.append(
                    {
                        "resource_name": resource,
                        "metadata_published_at": (
                            metadata_value
                        ),
                        "stage": "metadata",
                        "error": (
                            "Missing ResourceName or "
                            "_publish_time_utc"
                        ),
                    }
                )

                continue

            metadata_at = _as_utc(
                metadata_value
            )

            key = (
                resource,
                metadata_at,
            )

            if key in seen:
                duplicate_count += 1
                continue

            seen.add(key)

            # -------------------------------------------------
            # Fetch XML
            # -------------------------------------------------

            try:
                content = _fetch_resource(
                    resource,
                    raw_dir=raw_dir,
                )

            except RuntimeError as exc:
                download_failures += 1

                skipped_reports.append(
                    {
                        "resource_name": resource,
                        "metadata_published_at": (
                            metadata_at
                        ),
                        "stage": "download",
                        "error": str(exc),
                    }
                )

                print(
                    f"{prefix}: WARNING "
                    f"download failed for "
                    f"{resource}: {exc}",
                    flush=True,
                )

                continue

            # -------------------------------------------------
            # Parse XML
            # -------------------------------------------------

            try:
                parsed = parse_report(
                    content
                )

            except (
                ValueError,
                KeyError,
                TypeError,
                ET.ParseError,
            ) as exc:
                parse_failures += 1

                skipped_reports.append(
                    {
                        "resource_name": resource,
                        "metadata_published_at": (
                            metadata_at
                        ),
                        "stage": "parse",
                        "error": str(exc),
                    }
                )

                if parse_failures <= 10:
                    print(
                        f"{prefix}: WARNING "
                        f"skipping {resource}: "
                        f"{exc}",
                        flush=True,
                    )

                elif parse_failures == 11:
                    print(
                        f"{prefix}: further parse "
                        f"failures will only be "
                        f"recorded in the skip log",
                        flush=True,
                    )

                continue

            if parsed is None or parsed.empty:
                parse_failures += 1

                skipped_reports.append(
                    {
                        "resource_name": resource,
                        "metadata_published_at": (
                            metadata_at
                        ),
                        "stage": "parse",
                        "error": (
                            "Parser returned no rows"
                        ),
                    }
                )

                continue

            # -------------------------------------------------
            # Validate XML publication timestamp
            # -------------------------------------------------

            publication_column = (
                f"{prefix}_forecast_published_at"
            )

            if (
                publication_column
                not in parsed.columns
            ):
                publication_failures += 1

                skipped_reports.append(
                    {
                        "resource_name": resource,
                        "metadata_published_at": (
                            metadata_at
                        ),
                        "stage": "publication",
                        "error": (
                            "Missing publication "
                            f"column "
                            f"{publication_column}"
                        ),
                    }
                )

                continue

            publication = pd.to_datetime(
                parsed[publication_column],
                utc=True,
                errors="coerce",
            )

            if (
                publication.isna().any()
                or publication.nunique() != 1
            ):
                publication_failures += 1

                skipped_reports.append(
                    {
                        "resource_name": resource,
                        "metadata_published_at": (
                            metadata_at
                        ),
                        "stage": "publication",
                        "error": (
                            "Invalid or non-unique "
                            "XML publication time"
                        ),
                    }
                )

                continue

            xml_at = publication.iloc[0]

            # -------------------------------------------------
            # Point-in-time availability
            #
            # Use the timestamp embedded in the original XML
            # as the true forecast publication time.
            #
            # The API metadata time is retained for provenance
            # but does not control forecast eligibility.
            # -------------------------------------------------

            available_at = xml_at

            # -------------------------------------------------
            # Convert complete intervals to hourly
            # -------------------------------------------------

            try:
                hourly = complete_hours(
                    parsed,
                    value,
                    minutes,
                )

            except (
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                parse_failures += 1

                skipped_reports.append(
                    {
                        "resource_name": resource,
                        "metadata_published_at": (
                            metadata_at
                        ),
                        "stage": "hourly",
                        "error": str(exc),
                    }
                )

                continue

            hourly = hourly.loc[
                hourly.timestamp.isin(
                    targets
                )
            ].copy()

            if hourly.empty:
                no_target_hours += 1

            else:
                hourly["metadata_at"] = (
                    metadata_at
                )

                hourly["xml_at"] = (
                    xml_at
                )

                hourly["available_at"] = (
                    available_at
                )

                hourly["resource_name"] = (
                    resource
                )

                versions.append(hourly)
                accepted_reports += 1

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if (
                number % 25 == 0
                or number == len(reports)
            ):
                print(
                    f"{prefix}: read "
                    f"{number}/{len(reports)} "
                    f"reports "
                    f"(accepted="
                    f"{accepted_reports}, "
                    f"parse_skips="
                    f"{parse_failures}, "
                    f"download_skips="
                    f"{download_failures})",
                    flush=True,
                )

        # -----------------------------------------------------
        # Report summary
        # -----------------------------------------------------

        print(
            "",
            flush=True,
        )

        print(
            f"{prefix}: processing summary",
            flush=True,
        )

        print(
            f"  metadata/cache records: "
            f"{len(reports)}",
            flush=True,
        )

        print(
            f"  unique reports:         "
            f"{len(seen)}",
            flush=True,
        )

        print(
            f"  duplicates skipped:     "
            f"{duplicate_count}",
            flush=True,
        )

        print(
            f"  accepted reports:       "
            f"{accepted_reports}",
            flush=True,
        )

        print(
            f"  no target hours:        "
            f"{no_target_hours}",
            flush=True,
        )

        print(
            f"  parse/hourly failures:  "
            f"{parse_failures}",
            flush=True,
        )

        print(
            f"  publication failures:   "
            f"{publication_failures}",
            flush=True,
        )

        print(
            f"  download failures:      "
            f"{download_failures}",
            flush=True,
        )

        _write_skip_log(
            skipped_reports,
            raw_dir,
            prefix,
        )

        # -----------------------------------------------------
        # Select latest valid point-in-time forecast version
        # -----------------------------------------------------

        selected = select_versions(
            targets,
            versions,
            prefix,
            value,
            max_age_hours,
        )

        output = output.merge(
            selected,
            on="timestamp",
            how="left",
            validate="one_to_one",
        )

        eligible_count = (
            output[value]
            .notna()
            .sum()
        )

        print(
            f"{prefix}: "
            f"{eligible_count}/{len(output)} "
            f"eligible hours",
            flush=True,
        )

        print(
            "",
            flush=True,
        )

    # ---------------------------------------------------------
    # Derived forecast variables
    # ---------------------------------------------------------

    output["net_demand_forecast_mw"] = (
        output.demand_forecast_mw
        - output.wind_forecast_mw
    )

    output["wind_share_forecast"] = (
        output.wind_forecast_mw
        / output.demand_forecast_mw.clip(
            lower=1
        )
    )

    return output


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(
            "data/raw/semo_forecasts"
        ),
    )

    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=72,
    )

    args = parser.parse_args()

    targets = pd.read_csv(
        args.input
    )["timestamp"]

    frame = fetch_rolling_forecasts(
        targets,
        args.raw_dir,
        args.max_age_hours,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        args.output,
        index=False,
    )

    complete = frame[
        [
            "demand_forecast_mw",
            "wind_forecast_mw",
        ]
    ].notna().all(axis=1)

    print(
        f"Saved {len(frame)} target hours; "
        f"{complete.sum()} have both forecasts: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()