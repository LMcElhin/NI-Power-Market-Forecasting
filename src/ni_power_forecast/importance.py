from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from ni_power_forecast.features import build_features

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
    """Load best HGB parameters from a tuning JSON file."""

    payload = json.loads(
        Path(path).read_text(
            encoding="utf-8"
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


def _prepare_features(
    df: pd.DataFrame,
    derived_forecasts: str,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """Build features and retain complete rows."""

    X, y = build_features(
        df,
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
    )


def cross_validated_permutation_importance(
    df: pd.DataFrame,
    *,
    params: dict[str, Any] | None = None,
    n_splits: int = 3,
    n_repeats: int = 10,
    derived_forecasts: str = "both",
    random_seed: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Measure feature importance using chronological validation folds.

    For every chronological fold:

    1. fit the tuned HGB model using the training window;
    2. evaluate validation MAE;
    3. shuffle one validation feature at a time;
    4. measure the increase in MAE.

    A positive delta MAE means that destroying the feature's information
    makes forecasts worse, indicating useful predictive information.

    Results are exploratory because the HGB hyperparameters were selected
    using the same cross-validation period. A future untouched holdout
    should be used for final importance estimates.
    """

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2"
        )

    if n_repeats < 1:
        raise ValueError(
            "n_repeats must be positive"
        )

    model_params = dict(
        DEFAULT_HGB_PARAMS
        if params is None
        else params
    )

    X, y = _prepare_features(
        df,
        derived_forecasts=derived_forecasts,
    )

    if len(X) < 24 * 30:
        raise ValueError(
            "Need at least ~30 days of usable "
            "hourly observations"
        )

    splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    raw_records: list[
        dict[str, Any]
    ] = []

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

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            **model_params,
            random_state=(
                random_seed + fold
            ),
        )

        model.fit(
            X_train,
            y_train,
        )

        baseline_prediction = (
            model.predict(
                X_test
            )
        )

        baseline_mae = float(
            mean_absolute_error(
                y_test,
                baseline_prediction,
            )
        )

        print(
            f"fold {fold}/{n_splits}: "
            f"baseline MAE={baseline_mae:.3f}",
            flush=True,
        )

        for feature_number, feature in enumerate(
            X.columns,
            start=1,
        ):
            original = (
                X_test[feature]
                .to_numpy(
                    copy=True
                )
            )

            for repeat in range(
                n_repeats
            ):
                seed = (
                    random_seed
                    + fold * 100_000
                    + feature_number * 1_000
                    + repeat
                )

                rng = np.random.default_rng(
                    seed
                )

                X_permuted = (
                    X_test.copy()
                )

                X_permuted[feature] = (
                    rng.permutation(
                        original
                    )
                )

                permuted_prediction = (
                    model.predict(
                        X_permuted
                    )
                )

                permuted_mae = float(
                    mean_absolute_error(
                        y_test,
                        permuted_prediction,
                    )
                )

                raw_records.append(
                    {
                        "fold": fold,
                        "feature": feature,
                        "repeat": repeat + 1,
                        "baseline_mae": (
                            baseline_mae
                        ),
                        "permuted_mae": (
                            permuted_mae
                        ),
                        "delta_mae": (
                            permuted_mae
                            - baseline_mae
                        ),
                        "delta_mae_pct": (
                            100
                            * (
                                permuted_mae
                                - baseline_mae
                            )
                            / baseline_mae
                        ),
                    }
                )

    raw = pd.DataFrame(
        raw_records
    )

    # Average repeated permutations within each fold first.
    # This prevents folds with more repetitions from receiving
    # unintended extra weight.
    by_fold = (
        raw.groupby(
            [
                "feature",
                "fold",
            ],
            as_index=False,
        )
        .agg(
            fold_delta_mae=(
                "delta_mae",
                "mean",
            ),
            fold_delta_mae_pct=(
                "delta_mae_pct",
                "mean",
            ),
        )
    )

    summary = (
        by_fold.groupby(
            "feature",
            as_index=False,
        )
        .agg(
            importance_mae=(
                "fold_delta_mae",
                "mean",
            ),
            importance_mae_std=(
                "fold_delta_mae",
                "std",
            ),
            importance_pct=(
                "fold_delta_mae_pct",
                "mean",
            ),
            importance_pct_std=(
                "fold_delta_mae_pct",
                "std",
            ),
        )
    )

    summary[
        "importance_mae_std"
    ] = summary[
        "importance_mae_std"
    ].fillna(0.0)

    summary[
        "importance_pct_std"
    ] = summary[
        "importance_pct_std"
    ].fillna(0.0)

    summary = summary.sort_values(
        [
            "importance_mae",
            "importance_pct",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    summary.insert(
        0,
        "rank",
        np.arange(
            1,
            len(summary) + 1,
        ),
    )

    return summary, raw


def save_importance_outputs(
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 20,
) -> Path:
    """Save importance tables and a top-feature plot."""

    out = Path(
        output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        out
        / "permutation_importance.csv",
        index=False,
    )

    raw.to_csv(
        out
        / "permutation_importance_raw.csv",
        index=False,
    )

    top = (
        summary.head(
            top_n
        )
        .sort_values(
            "importance_mae",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.barh(
        top["feature"],
        top["importance_mae"],
        xerr=top[
            "importance_mae_std"
        ],
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Increase in validation MAE after permutation"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Cross-validated permutation importance"
    )

    fig.tight_layout()

    fig.savefig(
        out
        / "permutation_importance.png",
        dpi=160,
    )

    plt.close(
        fig
    )

    return out