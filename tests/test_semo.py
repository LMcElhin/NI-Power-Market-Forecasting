import json
from pathlib import Path

from ni_power_forecast.data.semo import inspect_document, parse_market_result_document


def _sample_payload():
    return {
        "rows": [
            [
                ["Market", "NI-DA"],
                ["Index", "Price", "GBP"],
                ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
                ["73.1", "75,4"],
                ["Index", "Volume", "MWh"],
                ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
                [100, 120],
            ]
        ]
    }


def test_semo_inspector_flattens_rows(tmp_path: Path):
    source = tmp_path / "report.json"
    source.write_text(json.dumps(_sample_payload()), encoding="utf-8")
    df = inspect_document(source)
    assert len(df) == 1
    assert "NI-DA" in df.loc[0, "values"]
    assert "GBP" in df.loc[0, "values"]


def test_semo_price_parser_selects_gbp_price_series():
    df = parse_market_result_document(_sample_payload())
    assert len(df) == 2
    assert df.loc[0, "price_gbp_mwh"] == 73.1
    assert df.loc[1, "price_gbp_mwh"] == 75.4
