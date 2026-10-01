from ni_power_forecast.data.demo import (
    generate_demo_data,
)
from ni_power_forecast.probabilistic import (
    probabilistic_backtest,
)


def test_probabilistic_backtest_is_monotonic_and_calibrated():
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
        "q10_raw",
        "q50_raw",
        "q90_raw",
        "q10",
        "q50",
        "q90",
        "q10_calibrated",
        "q90_calibrated",
    }.issubset(
        predictions.columns
    )

    assert (
        predictions["q10"]
        <= predictions["q50"]
    ).all()

    assert (
        predictions["q50"]
        <= predictions["q90"]
    ).all()

    assert (
        predictions[
            "q10_calibrated"
        ]
        <= predictions["q10"]
    ).all()

    assert (
        predictions[
            "q90_calibrated"
        ]
        >= predictions["q90"]
    ).all()

    assert (
        metrics[
            "quantile_crossing_rate_after_rearrangement"
        ]
        == 0.0
    )

    assert (
        metrics[
            "interval_80_mean_width_calibrated"
        ]
        >= metrics[
            "interval_80_mean_width_raw"
        ]
    )

    assert (
        metrics[
            "interval_80_coverage_calibrated"
        ]
        >= metrics[
            "interval_80_coverage_raw"
        ]
    )