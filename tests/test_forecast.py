from pathlib import Path

import pandas as pd

from ni_power_forecast.data.demo import generate_demo_data
from ni_power_forecast.forecast import forecast_missing_targets
from ni_power_forecast.train import train_model


def test_train_and_forecast_missing_final_day(tmp_path: Path):
    df = generate_demo_data(days=80, seed=6)
    df.loc[df.index[-24:], "price_gbp_mwh"] = float("nan")
    model_dir = tmp_path / "model"
    out = tmp_path / "forecast.csv"
    train_model(df, output_dir=model_dir)
    forecast_missing_targets(df, model_dir=model_dir, output_path=out)
    pred = pd.read_csv(out)
    assert len(pred) == 24
    assert pred["forecast_price_gbp_mwh"].notna().all()
