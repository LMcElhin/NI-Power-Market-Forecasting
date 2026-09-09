from __future__ import annotations

import json
from pathlib import Path

import typer

from ni_power_forecast.backtest import (
    save_backtest_outputs,
    walk_forward_backtest,
)
from ni_power_forecast.data.demo import (
    write_demo_data,
)
from ni_power_forecast.data.io import (
    load_market_frame,
)
from ni_power_forecast.data.semo import (
    download_report_documents,
    fetch_day_ahead_prices,
    inspect_document,
    parse_market_result_file,
)
from ni_power_forecast.data.soni import (
    normalise_soni_excel,
)
from ni_power_forecast.forecast import (
    forecast_missing_targets,
)
from ni_power_forecast.merge import (
    merge_price_and_system,
)
from ni_power_forecast.train import (
    train_model,
)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
)


@app.command("generate-demo")
def generate_demo(
    days: int = typer.Option(
        300,
        min=30,
    ),
    output: Path = typer.Option(Path("data/demo/ni_power_demo.csv")),
    seed: int = typer.Option(42),
):
    """Generate a reproducible synthetic NI-like hourly dataset."""

    path = write_demo_data(
        output,
        days=days,
        seed=seed,
    )

    typer.echo(f"Wrote demo data to {path}")


@app.command("backtest")
def backtest(
    input: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(Path("outputs/backtest")),
    model: str = typer.Option("hist_gradient_boosting"),
    n_splits: int = typer.Option(
        5,
        min=2,
        max=10,
    ),
    target_mode: str = typer.Option(
        "level",
        help=(
            "Prediction target. "
            "'level' predicts price "
            "directly; 'residual' "
            "predicts corrections to "
            "24h persistence."
        ),
    ),
):
    """Run expanding-window backtests against 24h persistence."""

    df = load_market_frame(input)

    predictions, metrics = walk_forward_backtest(
        df,
        model_name=model,
        n_splits=n_splits,
        target_mode=target_mode,
    )

    save_backtest_outputs(
        predictions,
        metrics,
        output_dir,
    )

    typer.echo(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    typer.echo(f"Saved outputs to {output_dir}")


@app.command("merge-data")
def merge_data(
    prices: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    system: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(Path("data/processed/model_table.csv")),
):
    """Merge hourly prices with normalised SONI system data."""

    import pandas as pd

    price_df = pd.read_csv(prices)

    system_df = pd.read_csv(system)

    frame = merge_price_and_system(
        price_df,
        system_df,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output,
        index=False,
    )

    typer.echo(f"Wrote {len(frame)} merged observations to {output}")


@app.command("train")
def train(
    input: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(Path("outputs/model")),
    model: str = typer.Option("hist_gradient_boosting"),
):
    """Fit a model to all usable observations and persist it."""

    df = load_market_frame(input)

    path = train_model(
        df,
        model_name=model,
        output_dir=output_dir,
    )

    typer.echo(f"Saved trained model to {path}")


@app.command("forecast")
def forecast(
    input: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    model_dir: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
    ),
    output: Path = typer.Option(Path("outputs/forecast.csv")),
):
    """Forecast rows with missing targets using a trained model."""

    df = load_market_frame(input)

    path = forecast_missing_targets(
        df,
        model_dir=model_dir,
        output_path=output,
    )

    typer.echo(f"Saved forecasts to {path}")


@app.command("normalise-soni")
def normalise_soni(
    input: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(Path("data/processed/soni_system.csv")),
    mapping: Path | None = typer.Option(None),
):
    """Normalise a SONI System Data Excel workbook."""

    path = normalise_soni_excel(
        input,
        output,
        mapping_file=mapping,
    )

    typer.echo(f"Wrote normalised SONI data to {path}")


@app.command("fetch-semo-raw")
def fetch_semo_raw(
    date_from: str = typer.Option(
        ...,
        help="YYYY-MM-DD",
    ),
    date_to: str = typer.Option(
        ...,
        help="YYYY-MM-DD",
    ),
    output_dir: Path = typer.Option(Path("data/raw/semo")),
):
    """Download raw public SEMOpx day-ahead report documents."""

    paths = download_report_documents(
        date_from,
        date_to,
        output_dir,
    )

    typer.echo(f"Downloaded {len(paths)} reports to {output_dir}")


@app.command("fetch-semo-prices")
def fetch_semo_prices(
    date_from: str = typer.Option(
        ...,
        help="YYYY-MM-DD",
    ),
    date_to: str = typer.Option(
        ...,
        help="YYYY-MM-DD",
    ),
    output: Path = typer.Option(Path("data/processed/semo_ni_da_prices.csv")),
    market: str = typer.Option("NI-DA"),
    currency: str = typer.Option("GBP"),
):
    """Fetch and normalise public SEMOpx day-ahead prices."""

    frame = fetch_day_ahead_prices(
        date_from,
        date_to,
        market=market,
        currency=currency,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output,
        index=False,
    )

    typer.echo(f"Wrote {len(frame)} price observations to {output}")


@app.command("parse-semo-prices")
def parse_semo_prices(
    input: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(Path("data/processed/semo_ni_da_prices.csv")),
    market: str = typer.Option("NI-DA"),
    currency: str = typer.Option("GBP"),
):
    """Parse an EA-001 document into timestamp/price rows."""

    frame = parse_market_result_file(
        input,
        market=market,
        currency=currency,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output,
        index=False,
    )

    typer.echo(f"Wrote {len(frame)} price observations to {output}")


@app.command("inspect-semo")
def inspect_semo(
    input: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(Path("data/processed/semo_inspection.csv")),
):
    """Flatten a raw SEMO document for schema inspection."""

    frame = inspect_document(input)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output,
        index=False,
    )

    typer.echo(f"Wrote {len(frame)} rows to {output}")
