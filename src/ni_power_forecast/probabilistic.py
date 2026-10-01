from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_pinball_loss,
)
from sklearn.model_selection import TimeSeriesSplit

from ni_power_forecast.features import build_features
from ni_power_forecast.schema import TIMESTAMP

DEFAULT_QUANTILES = (
    0.10,
    0.50,
    0.90,
)


DEFAULT_HGB_PARAMS: dict[str, Any] = {
    "learning_rate": 0.10,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 10.0,
    "max_iter": 200,
}


def load_hgb_parameters(
    path: str | Path,
) -> dict[str, Any]:
    """Load tuned HGB parameters from best_hgb.json."""

    payload = json.loads(
        Path(path).read_text(
            encoding="utf-8",
        )
    )

    if "best_parameters" not in payload:
        raise ValueError(
            "Tuning JSON does not contain "
            "'best_parameters'"
        )

    return dict(
        payload["best_parameters"]
    )


def _prepare_data(
    df: pd.DataFrame,
    derived_forecasts: str,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """Build complete leakage-safe features."""

    X, y = build_features(
        df,
        derived_forecasts=derived_forecasts,
    )

    mask = (
        X.notna().all(axis=1)
        & y.notna()
    )

    return (
        X.loc[mask].reset_index(drop=True),
        y.loc[mask].reset_index(drop=True),
        df.loc[mask].reset_index(drop=True),
    )


def probabilistic_backtest(
    df: pd.DataFrame,
    *,
    params: dict[str, Any] | None = None,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    n_splits: int = 3,
    derived_forecasts: str = "both",
    random_seed: int = 42,
) -> tuple[
    pd.DataFrame,
    dict[str, float],
]:
    """Run chronological HGB quantile backtesting."""

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2"
        )

    if 0.5 not in quantiles:
        raise ValueError(
            "quantiles must include 0.5"
        )

    for quantile in quantiles:
        if not 0 < quantile < 1:
            raise ValueError(
                "quantiles must lie between 0 and 1"
            )

    model_params = dict(
        DEFAULT_HGB_PARAMS
        if params is None
        else params
    )

    X, y, aligned = _prepare_data(
        df,
        derived_forecasts,
    )

    if len(X) < 24 * 30:
        raise ValueError(
            "Need at least ~30 days of usable "
            "hourly observations"
        )

    splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    pieces: list[pd.DataFrame] = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        result = pd.DataFrame(
            {
                "timestamp": (
                    aligned.loc[
                        test_idx,
                        TIMESTAMP,
                    ].to_numpy()
                ),
                "actual": (
                    y_test.to_numpy(
                        dtype=float
                    )
                ),
                "fold": fold,
            }
        )

        for quantile in quantiles:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                **model_params,
                random_state=(
                    random_seed
                    + fold
                    + int(
                        quantile * 1000
                    )
                ),
            )

            model.fit(
                X_train,
                y_train,
            )

            prediction = model.predict(
                X_test
            )

            column = (
                f"q{int(quantile * 100):02d}"
            )

            result[column] = prediction

        pieces.append(
            result
        )

    predictions = pd.concat(
        pieces,
        ignore_index=True,
    )

    actual = predictions[
        "actual"
    ].to_numpy(
        dtype=float
    )

    metrics: dict[str, float] = {}

    for quantile in quantiles:
        column = (
            f"q{int(quantile * 100):02d}"
        )

        metrics[
            f"pinball_{column}"
        ] = float(
            mean_pinball_loss(
                actual,
                predictions[
                    column
                ].to_numpy(
                    dtype=float
                ),
                alpha=quantile,
            )
        )

    median = predictions[
        "q50"
    ].to_numpy(
        dtype=float
    )

    metrics["median_mae"] = float(
        mean_absolute_error(
            actual,
            median,
        )
    )

    if (
        "q10" in predictions.columns
        and "q90" in predictions.columns
    ):
        lower = predictions[
            "q10"
        ].to_numpy(
            dtype=float
        )

        upper = predictions[
            "q90"
        ].to_numpy(
            dtype=float
        )

        crossing = (
            lower > upper
        )

        metrics[
            "quantile_crossing_rate"
        ] = float(
            np.mean(
                crossing
            )
        )

        valid = ~crossing

        if valid.any():
            metrics[
                "interval_80_coverage"
            ] = float(
                np.mean(
                    (
                        actual[valid]
                        >= lower[valid]
                    )
                    & (
                        actual[valid]
                        <= upper[valid]
                    )
                )
            )

            metrics[
                "interval_80_mean_width"
            ] = float(
                np.mean(
                    upper[valid]
                    - lower[valid]
                )
            )

        else:
            metrics[
                "interval_80_coverage"
            ] = float("nan")

            metrics[
                "interval_80_mean_width"
            ] = float("nan")

    return (
        predictions,
        metrics,
    )


def save_probabilistic_outputs(
    predictions: pd.DataFrame,
    metrics: dict[str, float],
    output_dir: str | Path,
) -> Path:
    """Save probabilistic backtest outputs."""

    out = Path(
        output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        out
        / "quantile_predictions.csv",
        index=False,
    )

    (
        out
        / "quantile_metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    return out