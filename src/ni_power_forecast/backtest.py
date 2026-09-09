from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import (
    TimeSeriesSplit,
)

from ni_power_forecast.features import (
    build_features,
)
from ni_power_forecast.metrics import (
    regression_metrics,
)
from ni_power_forecast.models import (
    make_model,
    persistence_prediction,
)
from ni_power_forecast.schema import TIMESTAMP


def _clean_xy(
    df: pd.DataFrame,
):
    """Build features and retain only fully usable observations."""

    X, y = build_features(df)

    mask = X.notna().all(axis=1) & y.notna()

    return (
        X.loc[mask].reset_index(drop=True),
        y.loc[mask].reset_index(drop=True),
        df.loc[mask].reset_index(drop=True),
    )


def walk_forward_backtest(
    df: pd.DataFrame,
    model_name: str = ("hist_gradient_boosting"),
    n_splits: int = 5,
    random_seed: int = 42,
    target_mode: str = "level",
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, float]],
]:
    """Run expanding-window chronological backtesting.

    target_mode="level"
        Predict absolute electricity price directly.

    target_mode="residual"
        Predict the correction to the 24-hour persistence forecast:

            price[t] - price[t - 24]

        Final forecast is:

            price[t - 24] + predicted correction
    """

    X, y, aligned = _clean_xy(df)

    if len(X) < 24 * 60:
        raise ValueError("Need at least ~60 days of usable hourly observations for backtesting")

    target_mode = target_mode.lower()

    if target_mode not in {
        "level",
        "residual",
    }:
        raise ValueError("target_mode must be 'level' or 'residual'")

    splitter = TimeSeriesSplit(n_splits=n_splits)

    pieces: list[pd.DataFrame] = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(X),
        start=1,
    ):
        model = make_model(
            model_name,
            random_seed=(random_seed + fold),
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_baseline = persistence_prediction(X_train)

        test_baseline = persistence_prediction(X_test)

        if target_mode == "residual":
            # Rather than learning the entire price level,
            # learn only the deviation from yesterday's
            # corresponding hourly price.
            residual_target = y_train.to_numpy(dtype=float) - train_baseline

            model.fit(
                X_train,
                residual_target,
            )

            residual_prediction = model.predict(X_test)

            prediction = test_baseline + residual_prediction

        else:
            model.fit(
                X_train,
                y_train,
            )

            prediction = model.predict(X_test)

        fold_df = pd.DataFrame(
            {
                "timestamp": (
                    aligned.loc[
                        test_idx,
                        TIMESTAMP,
                    ].to_numpy()
                ),
                "actual": (y_test.to_numpy()),
                "prediction": prediction,
                "baseline": (test_baseline),
                "fold": fold,
            }
        )

        pieces.append(fold_df)

    predictions = pd.concat(
        pieces,
        ignore_index=True,
    )

    model_metrics = regression_metrics(
        predictions["actual"],
        predictions["prediction"],
        predictions["baseline"],
    )

    baseline_metrics = regression_metrics(
        predictions["actual"],
        predictions["baseline"],
    )

    model_metrics["mae_improvement_vs_baseline_pct"] = float(
        100 * (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"]
    )

    model_metrics["rmse_improvement_vs_baseline_pct"] = float(
        100 * (baseline_metrics["rmse"] - model_metrics["rmse"]) / baseline_metrics["rmse"]
    )

    return predictions, {
        "model": model_metrics,
        "persistence_24h": (baseline_metrics),
    }


def save_backtest_outputs(
    predictions: pd.DataFrame,
    metrics: dict[
        str,
        dict[str, float],
    ],
    output_dir: str | Path,
) -> Path:
    """Save backtest predictions, metrics, and diagnostic plot."""

    out = Path(output_dir)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        out / "predictions.csv",
        index=False,
    )

    (out / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Show the final week of the backtest for a quick visual
    # comparison between actual, ML, and persistence.
    tail = predictions.tail(24 * 7)

    fig, ax = plt.subplots(figsize=(11, 4.8))

    ax.plot(
        pd.to_datetime(tail["timestamp"]),
        tail["actual"],
        label="Actual",
    )

    ax.plot(
        pd.to_datetime(tail["timestamp"]),
        tail["prediction"],
        label="ML forecast",
    )

    ax.plot(
        pd.to_datetime(tail["timestamp"]),
        tail["baseline"],
        label="24h persistence",
        alpha=0.7,
    )

    ax.set_title("Northern Ireland power-price forecast — final backtest week")

    ax.set_ylabel("GBP/MWh")

    ax.legend()

    fig.autofmt_xdate()

    fig.tight_layout()

    fig.savefig(
        out / "backtest.png",
        dpi=160,
    )

    plt.close(fig)

    return out
