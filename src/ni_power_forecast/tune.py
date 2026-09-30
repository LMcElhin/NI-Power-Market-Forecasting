"""Tune random forests on development folds without scoring the holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from ni_power_forecast.experiment import prepare_experiment
from ni_power_forecast.metrics import regression_metrics


def tune_forests(df, holdout_days=28, n_splits=3, trees=300):
    if holdout_days < 1 or n_splits < 2 or trees < 1:
        raise ValueError(
            "Require holdout_days >= 1, n_splits >= 2, trees >= 1"
        )

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], utc=True
    )

    if frame.empty or frame["timestamp"].isna().any():
        raise ValueError("Input must have valid timestamps")

    # Preserve the calendar cutoff used by the previous experiment.
    cutoff = (
        frame["timestamp"].max()
        + pd.Timedelta(hours=1)
        - pd.Timedelta(days=holdout_days)
    )
    development_end = cutoff - pd.Timedelta(hours=24)

    development = frame.loc[
        frame["timestamp"] < development_end
    ].copy()

    # Remove holdout rows before constructing features or fitting models.
    X, y, timestamps = prepare_experiment(
        development,
        include_forecasts=False,
    )

    if len(X) < 60 * 24:
        raise ValueError(
            "Need at least 60 usable development days after warm-up"
        )

    splits = list(
        TimeSeriesSplit(n_splits=n_splits, gap=24).split(X)
    )
    rows = []
    boundaries = []

    # Record boundaries and evaluate the persistence benchmarks.
    for fold, (train, valid) in enumerate(splits, start=1):
        boundaries.append(
            {
                "fold": fold,
                "train_end": str(timestamps.iloc[train[-1]]),
                "validation_start": str(timestamps.iloc[valid[0]]),
                "validation_end": str(timestamps.iloc[valid[-1]]),
            }
        )

        for lag in (24, 168):
            rows.append(
                {
                    "candidate": f"persistence_{lag}h",
                    "fold": fold,
                    "train_mae": np.nan,
                    **regression_metrics(
                        y.iloc[valid],
                        X.iloc[valid][f"price_lag_{lag}"],
                    ),
                }
            )

    # Test eight random-forest configurations.
    for mode in ("level", "residual"):
        for leaf in (3, 10, 25, 50):
            name = f"rf_{mode}_leaf{leaf}"
            print(f"Running {name}...", flush=True)

            for fold, (train, valid) in enumerate(splits, start=1):
                model = RandomForestRegressor(
                    n_estimators=trees,
                    min_samples_leaf=leaf,
                    random_state=42,
                    n_jobs=-1,
                )

                train_base = (
                    X.iloc[train]["price_lag_24"].to_numpy()
                )
                valid_base = (
                    X.iloc[valid]["price_lag_24"].to_numpy()
                )

                target = y.iloc[train].to_numpy()

                if mode == "residual":
                    target = target - train_base

                model.fit(X.iloc[train], target)

                train_prediction = model.predict(X.iloc[train])
                prediction = model.predict(X.iloc[valid])

                if mode == "residual":
                    train_prediction = train_prediction + train_base
                    prediction = prediction + valid_base

                rows.append(
                    {
                        "candidate": name,
                        "fold": fold,
                        "train_mae": regression_metrics(
                            y.iloc[train],
                            train_prediction,
                        )["mae"],
                        **regression_metrics(
                            y.iloc[valid],
                            prediction,
                        ),
                    }
                )

    folds = pd.DataFrame(rows)

    ranking = folds.groupby("candidate").agg(
        mean_mae=("mae", "mean"),
        worst_fold_mae=("mae", "max"),
        mean_rmse=("rmse", "mean"),
        mean_bias=("bias", "mean"),
        mean_train_mae=("train_mae", "mean"),
    )

    daily = (
        folds.loc[folds.candidate == "persistence_24h"]
        .set_index("fold")["mae"]
    )

    folds["improvement_vs_daily_pct"] = 100 * (
        1 - folds.mae / folds.fold.map(daily)
    )

    ranking["folds_beating_daily"] = (
        folds.assign(win=folds.improvement_vs_daily_pct > 0)
        .groupby("candidate")["win"]
        .sum()
    )

    ranking["improvement_vs_daily_pct"] = 100 * (
        1
        - ranking.mean_mae
        / ranking.loc["persistence_24h", "mean_mae"]
    )

    ranking = ranking.sort_values("mean_mae", kind="stable")

    summary = {
        "selected_candidate": ranking.index[0],
        "selection_metric": "mean validation MAE",
        "holdout_scored": False,
        "excluded_holdout_start": str(cutoff),
        "development_last_timestamp": str(timestamps.iloc[-1]),
        "n_estimators": trees,
        "seed": 42,
        "features": list(X.columns),
        "fold_boundaries": boundaries,
        "note": (
            "Fixed model per fold; rolling 24h inputs; "
            "publication delays not audited."
        ),
    }

    return folds, ranking, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rf_tuning"),
    )
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument("--trees", type=int, default=300)

    args = parser.parse_args()
    frame = pd.read_csv(args.input)

    folds, ranking, summary = tune_forests(
        frame,
        holdout_days=args.holdout_days,
        trees=args.trees,
    )

    summary["input_sha256"] = hashlib.sha256(
        args.input.read_bytes()
    ).hexdigest()

    args.output.mkdir(parents=True, exist_ok=True)

    folds.to_csv(
        args.output / "tuning_folds.csv",
        index=False,
    )
    ranking.to_csv(args.output / "tuning_ranking.csv")

    (args.output / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nDEVELOPMENT VALIDATION ONLY — no holdout scores")
    print(ranking.round(3).to_string())
    print(f"\nSelected: {summary['selected_candidate']}")


if __name__ == "__main__":
    main()