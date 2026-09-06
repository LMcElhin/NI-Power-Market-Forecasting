import numpy as np

from ni_power_forecast.data.demo import generate_demo_data
from ni_power_forecast.schema import REQUIRED_COLUMNS


def test_demo_has_required_schema_and_size():
    df = generate_demo_data(days=40, seed=1)
    assert REQUIRED_COLUMNS.issubset(df.columns)
    assert len(df) == 40 * 24
    assert df["timestamp"].is_monotonic_increasing
    assert np.isfinite(df["price_gbp_mwh"]).all()
