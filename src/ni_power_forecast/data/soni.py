from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ALIASES = {
    "timestamp": ["datetime", "date time", "timestamp", "date/time", "date"],
    "demand_actual_mw": ["system demand", "demand actual", "actual demand", "demand"],
    "demand_forecast_mw": ["system demand forecast", "demand forecast", "forecast demand"],
    "wind_actual_mw": ["wind generation", "actual wind", "wind actual"],
    "wind_forecast_mw": ["wind forecast", "forecast wind"],
    "interconnector_flow_mw": ["interconnection", "interconnector", "moyle"],
}


def _normalise_name(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _auto_mapping(columns) -> dict[str, str]:
    normalised = {_normalise_name(c): c for c in columns}
    mapping = {}
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
        mapping.update({k: v for k, v in explicit.items() if v in raw.columns})

    required = {"timestamp", "demand_actual_mw", "wind_actual_mw"}
    missing = required.difference(mapping)
    if missing:
        raise ValueError(
            f"Could not detect SONI columns {sorted(missing)}. Available columns: {list(raw.columns)}. "
            "Use config/soni_columns.example.yml as an explicit mapping template."
        )

    rename = {source: target for target, source in mapping.items()}
    out = raw.rename(columns=rename)[list(mapping.keys())].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path
