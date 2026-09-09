import numpy as np
import pytest

from ni_power_forecast.backtest import (
    walk_forward_backtest,
)
from ni_power_forecast.data.demo import (
    generate_demo_data,
)


def test_walk_forward_level_backtest_runs():
    df = generate_demo_data(
        days=100,
        seed=4,
    )

    predictions, metrics = walk_forward_backtest(
        df,
        n_splits=3,
        target_mode="level",
    )

    assert len(predictions) > 0

    assert {
        "actual",
        "prediction",
        "baseline",
        "fold",
    }.issubset(predictions.columns)

    assert np.isfinite(metrics["model"]["mae"])

    assert np.isfinite(metrics["persistence_24h"]["mae"])

    assert "rmse_improvement_vs_baseline_pct" in metrics["model"]


def test_walk_forward_residual_backtest_runs():
    df = generate_demo_data(
        days=100,
        seed=8,
    )

    predictions, metrics = walk_forward_backtest(
        df,
        model_name=("hist_gradient_boosting"),
        n_splits=3,
        target_mode="residual",
    )

    assert len(predictions) > 0

    assert np.isfinite(predictions["prediction"].to_numpy()).all()

    assert np.isfinite(metrics["model"]["mae"])

    assert np.isfinite(metrics["model"]["rmse"])


def test_invalid_target_mode_raises():
    df = generate_demo_data(
        days=100,
        seed=9,
    )

    with pytest.raises(
        ValueError,
        match="target_mode",
    ):
        walk_forward_backtest(
            df,
            n_splits=3,
            target_mode="nonsense",
        )
