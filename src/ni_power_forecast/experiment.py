"""Chronological model selection with a final untouched holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ni_power_forecast.features import build_features
from ni_power_forecast.metrics import regression_metrics
from ni_power_forecast.models import make_model
from ni_power_forecast.schema import TIMESTAMP


def prepare_experiment(df, include_forecasts=False):
    frame = df.copy()
    frame[TIMESTAMP] = pd.to_datetime(
        frame[TIMESTAMP], utc=True, errors="raise"
    )
    frame = frame.sort_values(TIMESTAMP).reset_index(drop=True)

    if (
        frame[TIMESTAMP].isna().any()
        or not frame[TIMESTAMP]
        .diff()
        .iloc[1:]
        .eq(pd.Timedelta(hours=1))
        .all()
    ):
        raise ValueError(
            "Input must have unique, continuous hourly timestamps; "
            "do not bridge gaps."
        )

    if not include_forecasts:
        frame = frame.drop(
            columns=[c for c in frame if "forecast" in c]
        )

    X, y = build_features(frame)
    valid = X.notna().all(axis=1) & y.notna()

    if not np.isfinite(y.loc[valid]).all():
        raise ValueError("Targets must be finite.")

    return (
        X.loc[valid].reset_index(drop=True),
        y.loc[valid].reset_index(drop=True),
        frame.loc[valid, TIMESTAMP].reset_index(drop=True),
    )


def run_experiment(
    df,
    holdout_days=28,
    n_splits=3,
    seed=42,
    include_forecasts=False,
):
    """Select on development folds; evaluate on the final holdout.

    Rolling 24h-ahead simulation with immediate label availability assumed.
    Models are fixed within each evaluation block; lagged inputs update hourly.
    This is not a single daily auction-origin simulation.
    """
    if holdout_days < 1 or n_splits < 2:
        raise ValueError(
            "holdout_days >= 1 and n_splits >= 2 are required"
        )

    X, y, ts = prepare_experiment(df, include_forecasts)

    if len(X) == 0:
        raise ValueError("No usable rows after feature warm-up")

    cutoff = (
        ts.iloc[-1]
        + pd.Timedelta(hours=1)
        - pd.Timedelta(days=holdout_days)
    )

    test = np.flatnonzero(ts >= cutoff)
    dev = np.flatnonzero(ts < cutoff - pd.Timedelta(hours=24))

    if len(dev) < 24 * 60 or len(test) < 24 * holdout_days:
        raise ValueError(
            "Need 60 usable development days plus a complete "
            "holdout and 24h gap"
        )

    splits = list(
        TimeSeriesSplit(n_splits=n_splits, gap=24).split(dev)
    )

    rows = []
    boundaries = []

    for fold, (tr, va) in enumerate(splits, 1):
        boundaries.append(
            {
                "fold": fold,
                "train_end": str(ts.iloc[tr[-1]]),
                "validation_start": str(ts.iloc[va[0]]),
                "validation_end": str(ts.iloc[va[-1]]),
            }
        )

    # Compare the existing ML model configurations.
    for name in ("ridge", "hgb", "rf"):
        model = make_model(name, seed)

        # Avoid automatic random internal validation on larger datasets.
        if name == "hgb":
            model.set_params(early_stopping=False)

        for fold, (tr, va) in enumerate(splits, 1):
            model.fit(X.iloc[tr], y.iloc[tr])

            train_mae = regression_metrics(
                y.iloc[tr],
                model.predict(X.iloc[tr]),
            )["mae"]

            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "n": len(va),
                    "train_mae": train_mae,
                    **regression_metrics(
                        y.iloc[va],
                        model.predict(X.iloc[va]),
                    ),
                }
            )

    # Evaluate both persistence baselines on identical validation rows.
    for lag in (24, 168):
        for fold, (_, va) in enumerate(splits, 1):
            rows.append(
                {
                    "model": f"persistence_{lag}h",
                    "fold": fold,
                    "n": len(va),
                    "train_mae": None,
                    **regression_metrics(
                        y.iloc[va],
                        X.iloc[va][f"price_lag_{lag}"],
                    ),
                }
            )

    folds = pd.DataFrame(rows)

    ranking = (
        folds.groupby("model")[["mae", "rmse"]]
        .mean()
        .sort_values("mae")
    )

    # Equal validation fold sizes make mean MAE equal pooled MAE.
    winner = ranking.index[0]

    if winner.startswith("persistence_"):
        lag = int(
            winner.removeprefix("persistence_").removesuffix("h")
        )
        prediction = X.iloc[test][f"price_lag_{lag}"].to_numpy()
    else:
        model = make_model(winner, seed)

        if winner == "hgb":
            model.set_params(early_stopping=False)

        model.fit(X.iloc[dev], y.iloc[dev])
        prediction = model.predict(X.iloc[test])

    predictions = pd.DataFrame(
        {
            "timestamp": ts.iloc[test].to_numpy(),
            "actual": y.iloc[test].to_numpy(),
            "selected": prediction,
            "persistence_24h": (
                X.iloc[test]["price_lag_24"].to_numpy()
            ),
            "persistence_168h": (
                X.iloc[test]["price_lag_168"].to_numpy()
            ),
        }
    )

    summary = {
        "selected_model": winner,
        "selection_metric": "mean validation MAE",
        "seed": seed,
        "include_forecasts": include_forecasts,
        "features": list(X.columns),
        "development_end": str(ts.iloc[dev[-1]]),
        "holdout_start": str(ts.iloc[test[0]]),
        "holdout_end": str(ts.iloc[test[-1]]),
        "fold_boundaries": boundaries,
        "holdout_metrics": {
            column: regression_metrics(
                predictions["actual"],
                predictions[column],
            )
            for column in (
                "selected",
                "persistence_24h",
                "persistence_168h",
            )
        },
    }

    return folds, ranking, predictions, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiment"),
    )
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument(
        "--include-forecasts",
        action="store_true",
        help=(
            "Only use after auditing publication times "
            "against forecast origins"
        ),
    )

    args = parser.parse_args()

    # Read directly so duplicates are caught rather than silently removed.
    raw = (
        pd.read_csv(args.input)
        if args.input.suffix.lower() == ".csv"
        else pd.read_parquet(args.input)
    )

    folds, ranking, predictions, summary = run_experiment(
        raw,
        args.holdout_days,
        include_forecasts=args.include_forecasts,
    )

    summary["input_sha256"] = hashlib.sha256(
        args.input.read_bytes()
    ).hexdigest()
    summary["input_path"] = str(args.input)

    args.output.mkdir(parents=True, exist_ok=True)

    folds.to_csv(
        args.output / "validation_folds.csv",
        index=False,
    )
    ranking.to_csv(args.output / "validation_ranking.csv")
    predictions.to_csv(
        args.output / "holdout_predictions.csv",
        index=False,
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(ranking.to_string())
    print(json.dumps(summary["holdout_metrics"], indent=2))
    print(
        f"Selected using validation only: "
        f"{summary['selected_model']}"
    )


if __name__ == "__main__":
    main()