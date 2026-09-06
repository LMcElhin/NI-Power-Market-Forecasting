from pathlib import Path

from ni_power_forecast.data.demo import write_demo_data
from ni_power_forecast.data.io import load_market_frame


def test_round_trip_demo_csv(tmp_path: Path):
    path = tmp_path / "demo.csv"
    write_demo_data(path, days=35, seed=5)
    df = load_market_frame(path)
    assert len(df) == 35 * 24
    assert str(df["timestamp"].dtype).startswith("datetime64")
