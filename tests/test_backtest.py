import numpy as np

from ni_power_forecast.backtest import walk_forward_backtest
from ni_power_forecast.data.demo import generate_demo_data


def test_walk_forward_backtest_runs():
    df = generate_demo_data(days=100, seed=4)
    predictions, metrics = walk_forward_backtest(df, n_splits=3)
    assert len(predictions) > 0
    assert set(["actual", "prediction", "baseline", "fold"]).issubset(predictions.columns)
    assert np.isfinite(metrics["model"]["mae"])
    assert np.isfinite(metrics["persistence_24h"]["mae"])
