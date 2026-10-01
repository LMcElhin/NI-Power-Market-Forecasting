from ni_power_forecast.data.demo import (
    generate_demo_data,
)
from ni_power_forecast.probabilistic import (
    probabilistic_backtest,
)


def test_probabilistic_backtest_returns_quantiles():
    df = generate_demo_data(
        days=45,
        seed=19,
    )

    params = {
        "learning_rate": 0.1,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "max_iter": 30,
    }

    predictions, metrics = (
        probabilistic_backtest(
            df,
            params=params,
            n_splits=2,
        )
    )

    assert {
        "q10",
        "q50",
        "q90",
    }.issubset(
        predictions.columns
    )

    assert "median_mae" in metrics

    assert (
        "interval_80_coverage"
        in metrics
    )

    assert (
        "interval_80_mean_width"
        in metrics
    )

    assert (
        "quantile_crossing_rate"
        in metrics
    )