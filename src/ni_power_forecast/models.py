from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(
    name: str,
    random_seed: int = 42,
) -> Any:
    """Construct one of the supported forecasting models."""

    name = name.lower()

    if name in {
        "hgb",
        "hist_gradient_boosting",
        "histgradientboosting",
    }:
        return HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=350,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_seed,
        )

    if name in {
        "hgb_mae",
        "hist_gradient_boosting_mae",
    }:
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.04,
            max_iter=500,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=2.0,
            random_state=random_seed,
        )

    if name in {
        "rf",
        "random_forest",
    }:
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=random_seed,
        )

    if name == "ridge":
        return Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(alpha=8.0),
                ),
            ]
        )

    if name in {
        "torch",
        "torch_mlp",
    }:
        from ni_power_forecast.torch_model import (
            TorchMLPRegressor,
        )

        return TorchMLPRegressor(random_seed=random_seed)

    raise ValueError(f"Unknown model: {name}")


def persistence_prediction(
    X,
) -> np.ndarray:
    """Return the 24-hour seasonal persistence forecast."""

    if "price_lag_24" not in X.columns:
        raise ValueError("price_lag_24 feature is required for persistence baseline")

    return X["price_lag_24"].to_numpy(dtype=float)
