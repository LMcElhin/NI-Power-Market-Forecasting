from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from ni_power_forecast.features import build_features
from ni_power_forecast.schema import TIMESTAMP


def forecast_missing_targets(
    df: pd.DataFrame,
    model_dir: str | Path,
    output_path: str | Path,
) -> Path:
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    feature_columns = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))

    X, y = build_features(df)
    missing_target = y.isna()
    usable = missing_target & X[feature_columns].notna().all(axis=1)
    if not usable.any():
        raise ValueError(
            "No forecastable rows found. Supply future rows with price_gbp_mwh empty and "
            "target-time demand/wind forecasts populated."
        )

    pred = model.predict(X.loc[usable, feature_columns])
    out = pd.DataFrame(
        {
            "timestamp": df.loc[usable, TIMESTAMP].to_numpy(),
            "forecast_price_gbp_mwh": pred,
        }
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path
