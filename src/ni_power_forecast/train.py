from __future__ import annotations

import json
from pathlib import Path

import joblib

from ni_power_forecast.features import build_features
from ni_power_forecast.models import make_model


def train_model(
    df, model_name: str = "hist_gradient_boosting", output_dir: str | Path = "outputs/model"
):
    X, y = build_features(df)
    mask = X.notna().all(axis=1) & y.notna()
    X_train = X.loc[mask]
    y_train = y.loc[mask]
    model = make_model(model_name)
    model.fit(X_train, y_train)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "feature_columns.json").write_text(json.dumps(list(X_train.columns), indent=2))
    return out
