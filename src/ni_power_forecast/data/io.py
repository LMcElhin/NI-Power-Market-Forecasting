from __future__ import annotations

from pathlib import Path

import pandas as pd

from ni_power_forecast.schema import REQUIRED_COLUMNS, TIMESTAMP


def load_market_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported data format: {path.suffix}")

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], utc=True, errors="raise")
    df = df.sort_values(TIMESTAMP).drop_duplicates(TIMESTAMP).reset_index(drop=True)
    return df


def save_market_frame(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
