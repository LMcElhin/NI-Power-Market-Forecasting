import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from ni_power_forecast.data.demo import generate_demo_data
from ni_power_forecast.experiment import (
    prepare_experiment,
    run_experiment,
)


def test_rejects_gaps_and_duplicates():
    df = generate_demo_data(days=100)

    for broken in (
        df.drop(index=100),
        pd.concat([df, df.iloc[[100]]]),
    ):
        with pytest.raises(ValueError, match="continuous hourly"):
            prepare_experiment(broken)


def test_forecasts_require_opt_in_and_alignment_is_sorted():
    df = generate_demo_data(days=100)

    X, y, ts = prepare_experiment(
        df.sample(frac=1, random_state=7)
    )

    assert not any("forecast" in column for column in X)
    assert ts.is_monotonic_increasing
    assert np.array_equal(
        y,
        df.set_index("timestamp").loc[ts, "price_gbp_mwh"],
    )

    X_forecast, _, _ = prepare_experiment(
        df,
        include_forecasts=True,
    )

    assert "wind_forecast_mw" in X_forecast


def test_holdout_targets_do_not_change_selection_or_earliest_prediction(
    monkeypatch,
):
    monkeypatch.setattr(
        "ni_power_forecast.experiment.make_model",
        lambda *args: FastModel(),
    )

    df = generate_demo_data(days=110)

    _, _, pred, summary = run_experiment(
        df,
        holdout_days=14,
    )

    changed = df.copy()
    start = pd.Timestamp(summary["holdout_start"])

    changed.loc[
        changed.timestamp >= start,
        "price_gbp_mwh",
    ] += 1000

    _, _, changed_pred, changed_summary = run_experiment(
        changed,
        holdout_days=14,
    )

    assert (
        summary["selected_model"]
        == changed_summary["selected_model"]
    )
    assert pred.selected.iloc[0] == changed_pred.selected.iloc[0]

    for boundary in summary["fold_boundaries"]:
        separation = (
            pd.Timestamp(boundary["validation_start"])
            - pd.Timestamp(boundary["train_end"])
        )
        assert separation > pd.Timedelta(hours=24)

    assert (
        start - pd.Timestamp(summary["development_end"])
        > pd.Timedelta(hours=24)
    )


class FastModel:
    """Use a cheap model to test experiment logic."""

    def __init__(self):
        self.model = Ridge()

    def set_params(self, **kwargs):
        return self

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)