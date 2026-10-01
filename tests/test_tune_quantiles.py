from ni_power_forecast.data.demo import (
    generate_demo_data,
)
from ni_power_forecast.tune_quantiles import (
    tune_quantile_models,
)


def test_quantile_tuning_returns_best_parameters():
    df = generate_demo_data(
        days=45,
        seed=23,
    )

    candidates = [
        {
            "learning_rate": 0.1,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "max_iter": 30,
        },
        {
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 10.0,
            "max_iter": 30,
        },
    ]

    results, payload = (
        tune_quantile_models(
            df,
            candidates=candidates,
            n_splits=2,
        )
    )

    assert len(results) == 6

    for name in (
        "q10",
        "q50",
        "q90",
    ):
        subset = results[
            results[
                "quantile_name"
            ]
            == name
        ]

        assert set(
            subset["rank"]
        ) == {
            1,
            2,
        }

        assert (
            name
            in payload[
                "best_parameters_by_quantile"
            ]
        )

        assert (
            name
            in payload["cv"]
        )