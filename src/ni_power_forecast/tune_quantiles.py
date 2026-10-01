from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import TimeSeriesSplit

from ni_power_forecast.features import build_features

DEFAULT_QUANTILES = (
    0.10,
    0.50,
    0.90,
)


DEFAULT_QUANTILE_CANDIDATES: list[dict[str, Any]] = [
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
        "learning_rate": 0.025,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 10.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 10.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 10.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 10.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 10.0,
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
        "l2_regularization": 30.0,
        "max_iter": 200,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 30.0,
        "max_iter": 400,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "l2_regularization": 10.0,
        "max_iter": 200,
    },
]


def _quantile_name(
    quantile: float,
) -> str:
    return f"q{int(round(quantile * 100)):02d}"


def _prepare_xy(
    df: pd.DataFrame,
    derived_forecasts: str,
) -> tuple[pd.DataFrame, pd.Series]:
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
    )


def tune_quantile_models(
    df: pd.DataFrame,
    *,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    candidates: list[dict[str, Any]] | None = None,
    n_splits: int = 3,
    derived_forecasts: str = "both",
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tune each HGB quantile model using chronological CV."""

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2"
        )

    for quantile in quantiles:
        if not 0 < quantile < 1:
            raise ValueError(
                "quantiles must lie between 0 and 1"
            )

    X, y = _prepare_xy(
        df,
        derived_forecasts,
    )

    if len(X) < 24 * 30:
        raise ValueError(
            "Need at least ~30 days of usable hourly data"
        )

    configurations = (
        DEFAULT_QUANTILE_CANDIDATES
        if candidates is None
        else candidates
    )

    if not configurations:
        raise ValueError(
            "At least one candidate is required"
        )

    splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    splits = list(
        splitter.split(X)
    )

    all_results: list[
        dict[str, Any]
    ] = []

    for quantile in quantiles:
        quantile_name = _quantile_name(
            quantile
        )

        print(
            "",
            flush=True,
        )

        print(
            f"Tuning {quantile_name}...",
            flush=True,
        )

        quantile_results: list[
            dict[str, Any]
        ] = []

        for candidate_number, params in enumerate(
            configurations,
            start=1,
        ):
            losses: list[float] = []

            for fold, (
                train_idx,
                test_idx,
            ) in enumerate(
                splits,
                start=1,
            ):
                model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=quantile,
                    **params,
                    random_state=(
                        random_seed
                        + fold
                        + int(
                            quantile * 1000
                        )
                    ),
                )

                model.fit(
                    X.iloc[train_idx],
                    y.iloc[train_idx],
                )

                prediction = model.predict(
                    X.iloc[test_idx]
                )

                loss = mean_pinball_loss(
                    y.iloc[test_idx],
                    prediction,
                    alpha=quantile,
                )

                losses.append(
                    float(loss)
                )

            result = {
                "quantile": quantile,
                "quantile_name": (
                    quantile_name
                ),
                "candidate": (
                    candidate_number
                ),
                **params,
                "mean_pinball": float(
                    np.mean(losses)
                ),
                "std_pinball": float(
                    np.std(losses)
                ),
            }

            quantile_results.append(
                result
            )

            print(
                f"  candidate "
                f"{candidate_number:02d}/"
                f"{len(configurations)} "
                f"pinball="
                f"{result['mean_pinball']:.4f}",
                flush=True,
            )

        frame = pd.DataFrame(
            quantile_results
        ).sort_values(
            [
                "mean_pinball",
                "std_pinball",
            ]
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

        all_results.extend(
            frame.to_dict(
                orient="records"
            )
        )

        best = frame.iloc[0]

        print(
            f"Best {quantile_name}: "
            f"pinball="
            f"{best['mean_pinball']:.4f}",
            flush=True,
        )

    results = pd.DataFrame(
        all_results
    )

    best_parameters: dict[
        str,
        dict[str, Any],
    ] = {}

    cv_summary: dict[
        str,
        dict[str, float],
    ] = {}

    parameter_names = [
        "learning_rate",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "max_iter",
    ]

    for quantile in quantiles:
        name = _quantile_name(
            quantile
        )

        best = (
            results[
                results[
                    "quantile_name"
                ]
                == name
            ]
            .sort_values(
                "rank"
            )
            .iloc[0]
        )

        best_parameters[name] = {
            parameter: (
                float(best[parameter])
                if parameter
                in {
                    "learning_rate",
                    "l2_regularization",
                }
                else int(
                    best[parameter]
                )
            )
            for parameter
            in parameter_names
        }

        cv_summary[name] = {
            "mean_pinball": float(
                best[
                    "mean_pinball"
                ]
            ),
            "std_pinball": float(
                best[
                    "std_pinball"
                ]
            ),
        }

    payload = {
        "best_parameters_by_quantile": (
            best_parameters
        ),
        "cv": cv_summary,
    }

    return results, payload


def save_quantile_tuning_outputs(
    results: pd.DataFrame,
    payload: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    out = Path(
        output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        out
        / "quantile_tuning_results.csv",
        index=False,
    )

    (
        out
        / "best_quantile_params.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return out