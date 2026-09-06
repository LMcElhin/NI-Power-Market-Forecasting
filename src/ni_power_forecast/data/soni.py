from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ALIASES = {
    "timestamp": [
        "datetime",
        "date time",
        "timestamp",
        "date/time",
        "date",
    ],
    "gmt_offset_hours": [
        "gmt offset",
        "utc offset",
    ],
    "demand_actual_mw": [
        "ni demand",
        "system demand",
        "demand actual",
        "actual demand",
        "demand",
    ],
    "demand_forecast_mw": [
        "ni demand forecast",
        "system demand forecast",
        "demand forecast",
        "forecast demand",
    ],
    "wind_actual_mw": [
        "ni wind generation",
        "wind generation",
        "actual wind",
        "wind actual",
    ],
    "wind_forecast_mw": [
        "ni wind forecast",
        "wind forecast",
        "forecast wind",
    ],
    "interconnector_flow_mw": [
        "moyle i/c",
        "moyle",
        "interconnection",
        "interconnector",
    ],
}


def _normalise_name(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _auto_mapping(columns) -> dict[str, str]:
    normalised = {_normalise_name(c): c for c in columns}

    mapping: dict[str, str] = {}

    for canonical, candidates in ALIASES.items():
        for candidate in candidates:
            if candidate in normalised:
                mapping[canonical] = normalised[candidate]
                break

    return mapping


def normalise_soni_excel(
    excel_path: str | Path,
    output_path: str | Path,
    sheet_name: str | int = 0,
    mapping_file: str | Path | None = None,
) -> Path:
    raw = pd.read_excel(excel_path, sheet_name=sheet_name)

    mapping = _auto_mapping(raw.columns)

    if mapping_file:
        explicit = yaml.safe_load(Path(mapping_file).read_text(encoding="utf-8")) or {}

        mapping.update({key: value for key, value in explicit.items() if value in raw.columns})

    required = {
        "timestamp",
        "demand_actual_mw",
        "wind_actual_mw",
    }

    missing = required.difference(mapping)

    if missing:
        raise ValueError(
            f"Could not detect SONI columns {sorted(missing)}. "
            f"Available columns: {list(raw.columns)}. "
            "Use config/soni_columns.example.yml as an explicit mapping."
        )

    rename = {source: target for target, source in mapping.items()}

    out = raw.rename(columns=rename)[list(mapping.keys())].copy()

    # SONI DateTime is a local wall-clock timestamp.
    # GMT Offset identifies whether that timestamp is GMT or BST.
    local_timestamp = pd.to_datetime(
        out["timestamp"],
        errors="coerce",
    )

    if "gmt_offset_hours" in out.columns:
        offset = pd.to_numeric(
            out["gmt_offset_hours"],
            errors="coerce",
        )

        utc_naive = local_timestamp - pd.to_timedelta(offset.fillna(0), unit="h")

        out["timestamp"] = utc_naive.dt.tz_localize("UTC")
        out = out.drop(columns=["gmt_offset_hours"])

    else:
        # Fallback for files which genuinely contain UTC timestamps.
        out["timestamp"] = pd.to_datetime(
            local_timestamp,
            utc=True,
            errors="coerce",
        )

    for column in out.columns:
        if column != "timestamp":
            out[column] = pd.to_numeric(
                out[column],
                errors="coerce",
            )

    out = (
        out.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(path, index=False)

    return path
