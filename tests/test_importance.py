from ni_power_forecast.data.demo import (
    generate_demo_data,
)
from ni_power_forecast.importance import (
    cross_validated_permutation_importance,
)


def test_permutation_importance_returns_ranked_features():
    df = generate_demo_data(
        days=45,
        seed=13,
    )

    params = {
        "learning_rate": 0.1,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "max_iter": 40,
    }

    summary, raw = (
        cross_validated_permutation_importance(
            df,
            params=params,
            n_splits=2,
            n_repeats=2,
        )
    )

    assert not summary.empty
    assert not raw.empty

    assert list(
        summary["rank"]
    ) == list(
        range(
            1,
            len(summary) + 1,
        )
    )

    assert {
        "feature",
        "importance_mae",
        "importance_mae_std",
        "importance_pct",
    }.issubset(
        summary.columns
    )

    assert {
        "fold",
        "feature",
        "repeat",
        "baseline_mae",
        "permuted_mae",
        "delta_mae",
    }.issubset(
        raw.columns
    )


def test_semo_forecasts_appear_in_importance():
    df = generate_demo_data(
        days=45,
        seed=17,
    )

    params = {
        "learning_rate": 0.1,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "max_iter": 30,
    }

    summary, _ = (
        cross_validated_permutation_importance(
            df,
            params=params,
            n_splits=2,
            n_repeats=1,
            derived_forecasts="both",
        )
    )

    features = set(
        summary["feature"]
    )

    assert "demand_forecast_mw" in features
    assert "wind_forecast_mw" in features
    assert "net_demand_forecast_mw" in features
    assert "wind_share_forecast" in features