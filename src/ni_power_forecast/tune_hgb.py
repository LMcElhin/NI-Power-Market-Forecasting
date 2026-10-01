from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from ni_power_forecast.features import build_features
from ni_power_forecast.models import persistence_prediction

DEFAULT_HGB_CANDIDATES: list[dict[str, Any]] = [
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.025,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "l2_regularization": 0.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "l2_regularization": 0.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 0.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 0.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 10.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 10.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 10,
        "l2_regularization": 1.0,
        "max_iter": 400,
    },
        {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 3.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 5.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 20.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 30.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 50.0,
        "max_iter": 200,
    },
]


def _prepare_xy(
    df: pd.DataFrame,
    derived_forecasts: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build features and retain complete observations."""

    X, y = build_features(
        df,
        derived_forecasts=derived_forecasts,
    )

    mask = X.notna().all(axis=1) & y.notna()

    return (
        X.loc[mask].reset_index(drop=True),
        y.loc[mask].reset_index(drop=True),
    )


def _directional_accuracy(
    actual: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
) -> float:
    """Fraction of correctly predicted moves versus 24h persistence."""

    actual_move = actual - baseline
    predicted_move = prediction - baseline

    return float(
        np.mean(
            np.sign(actual_move)
            == np.sign(predicted_move)
        )
    )


def tune_hist_gradient_boosting(
    df: pd.DataFrame,
    n_splits: int = 3,
    target_mode: str = "level",
    derived_forecasts: str = "both",
    random_seed: int = 42,
    candidates: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Tune HistGradientBoosting using chronological cross-validation.

    Candidate configurations are ranked primarily by mean fold MAE.

    No future fold is used to train an earlier fold.
    """

    target_mode = target_mode.lower()

    if target_mode not in {
        "level",
        "residual",
    }:
        raise ValueError(
            "target_mode must be 'level' or 'residual'"
        )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2"
        )

    X, y = _prepare_xy(
        df,
        derived_forecasts=derived_forecasts,
    )

    if len(X) < 24 * 30:
        raise ValueError(
            "Need at least ~30 days of usable hourly "
            "observations for tuning"
        )

    configurations = (
        candidates
        if candidates is not None
        else DEFAULT_HGB_CANDIDATES
    )

    if not configurations:
        raise ValueError(
            "At least one candidate configuration is required"
        )

    splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    split_indices = list(
        splitter.split(X)
    )

    results: list[dict[str, Any]] = []

    for candidate_number, params in enumerate(
        configurations,
        start=1,
    ):
        fold_mae: list[float] = []
        fold_rmse: list[float] = []
        fold_bias: list[float] = []
        fold_directional: list[float] = []

        print(
            f"HGB candidate "
            f"{candidate_number}/{len(configurations)}: "
            f"{params}",
            flush=True,
        )

        for fold_number, (
            train_idx,
            test_idx,
        ) in enumerate(
            split_indices,
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

            train_baseline = persistence_prediction(
                X_train
            )

            test_baseline = persistence_prediction(
                X_test
            )

            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=params[
                    "learning_rate"
                ],
                max_iter=params[
                    "max_iter"
                ],
                max_leaf_nodes=params[
                    "max_leaf_nodes"
                ],
                min_samples_leaf=params[
                    "min_samples_leaf"
                ],
                l2_regularization=params[
                    "l2_regularization"
                ],
                random_state=(
                    random_seed + fold_number
                ),
            )

            if target_mode == "residual":
                training_target = (
                    y_train.to_numpy(
                        dtype=float
                    )
                    - train_baseline
                )

                model.fit(
                    X_train,
                    training_target,
                )

                prediction = (
                    test_baseline
                    + model.predict(
                        X_test
                    )
                )

            else:
                model.fit(
                    X_train,
                    y_train,
                )

                prediction = model.predict(
                    X_test
                )

            actual = y_test.to_numpy(
                dtype=float
            )

            mae = mean_absolute_error(
                actual,
                prediction,
            )

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        actual,
                        prediction,
                    )
                )
            )

            bias = float(
                np.mean(
                    prediction - actual
                )
            )

            directional = (
                _directional_accuracy(
                    actual,
                    prediction,
                    test_baseline,
                )
            )

            fold_mae.append(
                float(mae)
            )

            fold_rmse.append(
                rmse
            )

            fold_bias.append(
                bias
            )

            fold_directional.append(
                directional
            )

        result = {
            "candidate": candidate_number,
            **params,
            "mean_mae": float(
                np.mean(fold_mae)
            ),
            "std_mae": float(
                np.std(fold_mae)
            ),
            "mean_rmse": float(
                np.mean(fold_rmse)
            ),
            "std_rmse": float(
                np.std(fold_rmse)
            ),
            "mean_bias": float(
                np.mean(fold_bias)
            ),
            "mean_directional_accuracy": float(
                np.mean(
                    fold_directional
                )
            ),
        }

        results.append(
            result
        )

        print(
            "  "
            f"MAE={result['mean_mae']:.3f}, "
            f"RMSE={result['mean_rmse']:.3f}, "
            f"DA="
            f"{100 * result['mean_directional_accuracy']:.2f}%",
            flush=True,
        )

    frame = pd.DataFrame(
        results
    )

    frame = frame.sort_values(
        [
            "mean_mae",
            "mean_rmse",
            "std_mae",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    frame.insert(
        0,
        "rank",
        np.arange(
            1,
            len(frame) + 1,
        ),
    )

    return frame


def save_hgb_tuning_outputs(
    results: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Save HGB tuning results and best parameters."""

    out = Path(
        output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        out / "hgb_tuning_results.csv",
        index=False,
    )

    best = results.iloc[
        0
    ]

    parameter_names = [
        "learning_rate",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "max_iter",
    ]

    best_params = {
        name: (
            float(best[name])
            if name
            in {
                "learning_rate",
                "l2_regularization",
            }
            else int(best[name])
        )
        for name in parameter_names
    }

    summary = {
        "best_parameters": best_params,
        "cv": {
            "mean_mae": float(
                best["mean_mae"]
            ),
            "std_mae": float(
                best["std_mae"]
            ),
            "mean_rmse": float(
                best["mean_rmse"]
            ),
            "std_rmse": float(
                best["std_rmse"]
            ),
            "mean_bias": float(
                best["mean_bias"]
            ),
            "mean_directional_accuracy": float(
                best[
                    "mean_directional_accuracy"
                ]
            ),
        },
    }

    (
        out / "best_hgb.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return out