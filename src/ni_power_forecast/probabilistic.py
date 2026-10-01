from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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


def _quantile_name(
    quantile: float,
) -> str:
    return f"q{int(round(quantile * 100)):02d}"


def load_quantile_parameters(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load tuned quantile parameters.

    Also accepts the older point-model best_hgb.json format
    and applies those parameters to all quantiles.
    """

    payload = json.loads(
        Path(path).read_text(
            encoding="utf-8",
        )
    )

    if (
        "best_parameters_by_quantile"
        in payload
    ):
        return {
            key: dict(value)
            for key, value
            in payload[
                "best_parameters_by_quantile"
            ].items()
        }

    if "best_parameters" in payload:
        params = dict(
            payload["best_parameters"]
        )

        return {
            "q10": dict(params),
            "q50": dict(params),
            "q90": dict(params),
        }

    raise ValueError(
        "Parameter file must contain either "
        "'best_parameters_by_quantile' "
        "or 'best_parameters'"
    )


def _resolve_parameters(
    params: (
        dict[str, dict[str, Any]]
        | dict[str, Any]
        | None
    ),
    quantiles: tuple[float, ...],
) -> dict[str, dict[str, Any]]:
    names = [
        _quantile_name(
            quantile
        )
        for quantile
        in quantiles
    ]

    if params is None:
        return {
            name: dict(
                DEFAULT_HGB_PARAMS
            )
            for name
            in names
        }

    if all(
        name in params
        for name in names
    ):
        return {
            name: dict(
                params[name]
            )
            for name
            in names
        }

    return {
        name: dict(params)
        for name
        in names
    }


def _prepare_data(
    df: pd.DataFrame,
    derived_forecasts: str,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    aligned = (
        df.copy()
        .sort_values(TIMESTAMP)
        .reset_index(drop=True)
    )

    X, y = build_features(
        aligned,
        derived_forecasts=derived_forecasts,
    )

    mask = (
        X.notna().all(axis=1)
        & y.notna()
    )

    return (
        X.loc[mask].reset_index(
            drop=True
        ),
        y.loc[mask].reset_index(
            drop=True
        ),
        aligned.loc[mask].reset_index(
            drop=True
        ),
    )


def _finite_sample_quantile(
    values: np.ndarray,
    probability: float,
) -> float:
    """Finite-sample conformal order statistic."""

    if len(values) == 0:
        raise ValueError(
            "Cannot calibrate using an empty array"
        )

    ordered = np.sort(
        np.asarray(
            values,
            dtype=float,
        )
    )

    rank = int(
        np.ceil(
            (len(ordered) + 1)
            * probability
        )
    )

    rank = min(
        max(rank, 1),
        len(ordered),
    )

    return float(
        ordered[
            rank - 1
        ]
    )


def _monotonic_rearrangement(
    predictions: np.ndarray,
) -> np.ndarray:
    """Force row-wise quantiles to be non-decreasing."""

    return np.sort(
        predictions,
        axis=1,
    )


def _coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    return float(
        np.mean(
            (actual >= lower)
            & (actual <= upper)
        )
    )


def _interval_score(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """Mean interval score for a central 1-alpha interval."""

    width = upper - lower

    below = actual < lower
    above = actual > upper

    score = width.copy()

    score[below] += (
        2.0
        / alpha
        * (
            lower[below]
            - actual[below]
        )
    )

    score[above] += (
        2.0
        / alpha
        * (
            actual[above]
            - upper[above]
        )
    )

    return float(
        np.mean(score)
    )


def probabilistic_backtest(
    df: pd.DataFrame,
    *,
    params: (
        dict[str, dict[str, Any]]
        | dict[str, Any]
        | None
    ) = None,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    n_splits: int = 3,
    derived_forecasts: str = "both",
    calibration_fraction: float = 0.20,
    min_calibration_hours: int = 24,
    min_fit_hours: int = 24 * 7,
    random_seed: int = 42,
) -> tuple[
    pd.DataFrame,
    dict[str, float],
]:
    """Chronological quantile backtest with conformal calibration."""

    if tuple(sorted(quantiles)) != quantiles:
        raise ValueError(
            "quantiles must be sorted"
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

    if not 0 < calibration_fraction < 0.5:
        raise ValueError(
            "calibration_fraction must be "
            "between 0 and 0.5"
        )

    X, y, aligned = _prepare_data(
        df,
        derived_forecasts,
    )

    if len(X) < 24 * 30:
        raise ValueError(
            "Need at least ~30 days of usable hourly data"
        )

    quantile_params = (
        _resolve_parameters(
            params,
            quantiles,
        )
    )

    splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    pieces: list[pd.DataFrame] = []

    lower_quantile = quantiles[0]
    upper_quantile = quantiles[-1]

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(X),
        start=1,
    ):
        desired_calibration = max(
            min_calibration_hours,
            int(
                round(
                    len(train_idx)
                    * calibration_fraction
                )
            ),
        )

        max_calibration = (
            len(train_idx)
            - min_fit_hours
        )

        if max_calibration < 1:
            raise ValueError(
                f"Fold {fold} has insufficient "
                "training history for calibration"
            )

        n_calibration = min(
            desired_calibration,
            max_calibration,
        )

        fit_idx = train_idx[
            :-n_calibration
        ]

        calibration_idx = train_idx[
            -n_calibration:
        ]

        X_fit = X.iloc[
            fit_idx
        ]

        y_fit = y.iloc[
            fit_idx
        ]

        X_calibration = X.iloc[
            calibration_idx
        ]

        y_calibration = (
            y.iloc[
                calibration_idx
            ]
            .to_numpy(
                dtype=float
            )
        )

        X_test = X.iloc[
            test_idx
        ]

        y_test = (
            y.iloc[
                test_idx
            ]
            .to_numpy(
                dtype=float
            )
        )

        calibration_predictions: list[
            np.ndarray
        ] = []

        test_predictions: list[
            np.ndarray
        ] = []

        for quantile in quantiles:
            name = _quantile_name(
                quantile
            )

            model = (
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=quantile,
                    **quantile_params[name],
                    random_state=(
                        random_seed
                        + fold
                        + int(
                            quantile
                            * 1000
                        )
                    ),
                )
            )

            model.fit(
                X_fit,
                y_fit,
            )

            calibration_predictions.append(
                model.predict(
                    X_calibration
                )
            )

            test_predictions.append(
                model.predict(
                    X_test
                )
            )

        calibration_matrix_raw = np.column_stack(
            calibration_predictions
        )

        test_matrix_raw = np.column_stack(
            test_predictions
        )

        calibration_matrix = (
            _monotonic_rearrangement(
                calibration_matrix_raw
            )
        )

        test_matrix = (
            _monotonic_rearrangement(
                test_matrix_raw
            )
        )

        lower_calibration = (
            calibration_matrix[
                :,
                0,
            ]
        )

        upper_calibration = (
            calibration_matrix[
                :,
                -1,
            ]
        )

        lower_scores = (
            lower_calibration
            - y_calibration
        )

        upper_scores = (
            y_calibration
            - upper_calibration
        )

        lower_correction = max(
            0.0,
            _finite_sample_quantile(
                lower_scores,
                1.0
                - lower_quantile,
            ),
        )

        upper_correction = max(
            0.0,
            _finite_sample_quantile(
                upper_scores,
                upper_quantile,
            ),
        )

        result = pd.DataFrame(
            {
                "timestamp": (
                    aligned.loc[
                        test_idx,
                        TIMESTAMP,
                    ].to_numpy()
                ),
                "actual": y_test,
                "fold": fold,
                "fit_samples": (
                    len(fit_idx)
                ),
                "calibration_samples": (
                    len(
                        calibration_idx
                    )
                ),
                "lower_correction": (
                    lower_correction
                ),
                "upper_correction": (
                    upper_correction
                ),
            }
        )

        for column_index, quantile in enumerate(
            quantiles
        ):
            name = _quantile_name(
                quantile
            )

            result[
                f"{name}_raw"
            ] = test_matrix_raw[
                :,
                column_index,
            ]

            result[name] = (
                test_matrix[
                    :,
                    column_index,
                ]
            )

        lower_name = _quantile_name(
            lower_quantile
        )

        upper_name = _quantile_name(
            upper_quantile
        )

        result[
            f"{lower_name}_calibrated"
        ] = (
            result[lower_name]
            - lower_correction
        )

        result[
            f"{upper_name}_calibrated"
        ] = (
            result[upper_name]
            + upper_correction
        )

        pieces.append(
            result
        )

        print(
            f"fold {fold}/{n_splits}: "
            f"fit={len(fit_idx)}, "
            f"calibration="
            f"{len(calibration_idx)}, "
            f"lower correction="
            f"{lower_correction:.3f}, "
            f"upper correction="
            f"{upper_correction:.3f}",
            flush=True,
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

    raw_matrix = np.column_stack(
        [
            predictions[
                f"{_quantile_name(q)}_raw"
            ].to_numpy(
                dtype=float
            )
            for q in quantiles
        ]
    )

    metrics[
        "quantile_crossing_rate_before_rearrangement"
    ] = float(
        np.mean(
            np.any(
                np.diff(
                    raw_matrix,
                    axis=1,
                )
                < 0,
                axis=1,
            )
        )
    )

    metrics[
        "quantile_crossing_rate_after_rearrangement"
    ] = 0.0

    for quantile in quantiles:
        name = _quantile_name(
            quantile
        )

        forecast = predictions[
            name
        ].to_numpy(
            dtype=float
        )

        metrics[
            f"pinball_{name}"
        ] = float(
            mean_pinball_loss(
                actual,
                forecast,
                alpha=quantile,
            )
        )

    median_name = _quantile_name(
        0.5
    )

    metrics[
        "median_mae"
    ] = float(
        mean_absolute_error(
            actual,
            predictions[
                median_name
            ],
        )
    )

    lower_name = _quantile_name(
        lower_quantile
    )

    upper_name = _quantile_name(
        upper_quantile
    )

    lower_raw = predictions[
        lower_name
    ].to_numpy(
        dtype=float
    )

    upper_raw = predictions[
        upper_name
    ].to_numpy(
        dtype=float
    )

    lower_calibrated = predictions[
        f"{lower_name}_calibrated"
    ].to_numpy(
        dtype=float
    )

    upper_calibrated = predictions[
        f"{upper_name}_calibrated"
    ].to_numpy(
        dtype=float
    )

    nominal_coverage = (
        upper_quantile
        - lower_quantile
    )

    alpha = (
        1.0
        - nominal_coverage
    )

    metrics[
        "nominal_interval_coverage"
    ] = float(
        nominal_coverage
    )

    metrics[
        "interval_80_coverage_raw"
    ] = _coverage(
        actual,
        lower_raw,
        upper_raw,
    )

    metrics[
        "interval_80_coverage_calibrated"
    ] = _coverage(
        actual,
        lower_calibrated,
        upper_calibrated,
    )

    metrics[
        "interval_80_mean_width_raw"
    ] = float(
        np.mean(
            upper_raw
            - lower_raw
        )
    )

    metrics[
        "interval_80_mean_width_calibrated"
    ] = float(
        np.mean(
            upper_calibrated
            - lower_calibrated
        )
    )

    metrics[
        "below_lower_rate_raw"
    ] = float(
        np.mean(
            actual
            < lower_raw
        )
    )

    metrics[
        "above_upper_rate_raw"
    ] = float(
        np.mean(
            actual
            > upper_raw
        )
    )

    metrics[
        "below_lower_rate_calibrated"
    ] = float(
        np.mean(
            actual
            < lower_calibrated
        )
    )

    metrics[
        "above_upper_rate_calibrated"
    ] = float(
        np.mean(
            actual
            > upper_calibrated
        )
    )

    metrics[
        "interval_80_score_raw"
    ] = _interval_score(
        actual,
        lower_raw,
        upper_raw,
        alpha,
    )

    metrics[
        "interval_80_score_calibrated"
    ] = _interval_score(
        actual,
        lower_calibrated,
        upper_calibrated,
        alpha,
    )

    metrics[
        "mean_lower_conformal_correction"
    ] = float(
        predictions[
            "lower_correction"
        ].mean()
    )

    metrics[
        "mean_upper_conformal_correction"
    ] = float(
        predictions[
            "upper_correction"
        ].mean()
    )

    return (
        predictions,
        metrics,
    )


def save_probabilistic_outputs(
    predictions: pd.DataFrame,
    metrics: dict[str, float],
    output_dir: str | Path,
) -> Path:
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

    calibration = (
        predictions[
            [
                "fold",
                "fit_samples",
                "calibration_samples",
                "lower_correction",
                "upper_correction",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "fold"
        )
    )

    calibration.to_csv(
        out
        / "calibration_by_fold.csv",
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

    tail = predictions.tail(
        24 * 7
    ).copy()

    x = pd.to_datetime(
        tail["timestamp"]
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.fill_between(
        x,
        tail[
            "q10_calibrated"
        ],
        tail[
            "q90_calibrated"
        ],
        alpha=0.25,
        label="Calibrated P10–P90",
    )

    ax.plot(
        x,
        tail["actual"],
        label="Actual",
    )

    ax.plot(
        x,
        tail["q50"],
        label="P50",
    )

    ax.set_ylabel(
        "GBP/MWh"
    )

    ax.set_title(
        "Probabilistic day-ahead price forecast"
    )

    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        out
        / "quantile_backtest.png",
        dpi=160,
    )

    plt.close(
        fig
    )

    return out