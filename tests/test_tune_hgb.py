from ni_power_forecast.data.demo import generate_demo_data
from ni_power_forecast.tune_hgb import (
    tune_hist_gradient_boosting,
)


def test_hgb_tuning_returns_ranked_results():
    df = generate_demo_data(
        days=45,
        seed=11,
    )

    candidates = [
        {
            "learning_rate": 0.1,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
            "max_iter": 50,
        },
        {
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "max_iter": 50,
        },
    ]

    results = tune_hist_gradient_boosting(
        df,
        n_splits=2,
        candidates=candidates,
    )

    assert len(results) == 2

    assert list(
        results["rank"]
    ) == [
        1,
        2,
    ]

    assert (
        results["mean_mae"].iloc[0]
        <= results["mean_mae"].iloc[1]
    )

    assert {
        "mean_mae",
        "std_mae",
        "mean_rmse",
        "mean_directional_accuracy",
    }.issubset(
        results.columns
    )