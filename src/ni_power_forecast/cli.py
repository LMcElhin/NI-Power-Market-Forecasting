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
from ni_power_forecast.data.semo_forecasts import (
    fetch_point_in_time_forecasts,
)
from ni_power_forecast.data.soni import (
    normalise_soni_excel,
)
from ni_power_forecast.forecast import (
    forecast_missing_targets,
)
from ni_power_forecast.importance import (
    cross_validated_permutation_importance,
    load_hgb_parameters,
    save_importance_outputs,
)
from ni_power_forecast.merge import (
    merge_price_and_system,
)
from ni_power_forecast.probabilistic import (
    load_hgb_parameters as load_quantile_hgb_parameters,
)
from ni_power_forecast.probabilistic import (
    probabilistic_backtest,
    save_probabilistic_outputs,
)
from ni_power_forecast.train import (
    train_model,
)
from ni_power_forecast.tune_hgb import (
    save_hgb_tuning_outputs,
    tune_hist_gradient_boosting,
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

@app.command("feature-importance")
def feature_importance(
    input: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        Path(
            "outputs/feature_importance"
        )
    ),
    params_file: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help=(
            "Optional best_hgb.json file. "
            "Uses tuned default parameters "
            "when omitted."
        ),
    ),
    n_splits: int = typer.Option(
        3,
        min=2,
        max=10,
    ),
    n_repeats: int = typer.Option(
        10,
        min=1,
        max=100,
    ),
    derived_forecasts: str = typer.Option(
        "both",
        help=(
            "Derived SEMO forecast features: "
            "none, net, share, or both."
        ),
    ),
):
    """Calculate chronological permutation feature importance."""

    df = load_market_frame(
        input
    )

    params = None

    if params_file is not None:
        params = load_hgb_parameters(
            params_file
        )

    summary, raw = (
        cross_validated_permutation_importance(
            df,
            params=params,
            n_splits=n_splits,
            n_repeats=n_repeats,
            derived_forecasts=(
                derived_forecasts
            ),
        )
    )

    save_importance_outputs(
        summary,
        raw,
        output_dir,
    )

    typer.echo(
        ""
    )

    typer.echo(
        "Top permutation features:"
    )

    typer.echo(
        summary.head(
            20
        ).to_string(
            index=False
        )
    )

    typer.echo(
        ""
    )

    typer.echo(
        f"Saved importance outputs to "
        f"{output_dir}"
    )
@app.command("quantile-backtest")
def quantile_backtest(
    input: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        Path(
            "outputs/quantile_backtest"
        )
    ),
    params_file: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
    ),
    n_splits: int = typer.Option(
        3,
        min=2,
        max=10,
    ),
    derived_forecasts: str = typer.Option(
        "both",
    ),
):
    """Backtest P10/P50/P90 price forecasts."""

    df = load_market_frame(
        input
    )

    params = None

    if params_file is not None:
        params = (
            load_quantile_hgb_parameters(
                params_file
            )
        )

    predictions, metrics = (
        probabilistic_backtest(
            df,
            params=params,
            n_splits=n_splits,
            derived_forecasts=(
                derived_forecasts
            ),
        )
    )

    save_probabilistic_outputs(
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

    typer.echo(
        f"Saved outputs to {output_dir}"
    )
    
@app.command("backtest")
def backtest(
    input: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        Path("outputs/backtest")
    ),
    model: str = typer.Option(
        "hist_gradient_boosting"
    ),
    n_splits: int = typer.Option(
        5,
        min=2,
        max=10,
    ),
    target_mode: str = typer.Option(
        "level",
        help=(
            "Prediction target: "
            "'level' or 'residual'."
        ),
    ),
    derived_forecasts: str = typer.Option(
        "both",
        help=(
            "Derived SEMO forecast features: "
            "none, net, share, or both."
        ),
    ),
):
    """Run expanding-window backtests against 24h persistence."""

    df = load_market_frame(
        input
    )

    predictions, metrics = (
        walk_forward_backtest(
            df,
            model_name=model,
            n_splits=n_splits,
            target_mode=target_mode,
            derived_forecasts=(
                derived_forecasts
            ),
        )
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

    typer.echo(
        f"Saved outputs to {output_dir}"
    )

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
    forecasts: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional point-in-time SEMO forecast CSV.",
    ),
    output: Path = typer.Option(
        Path("data/processed/model_table.csv")
    ),
):
    """Merge prices, SONI system data and optional SEMO forecasts."""

    import pandas as pd

    price_df = pd.read_csv(prices)
    system_df = pd.read_csv(system)

    forecast_df = None

    if forecasts is not None:
        forecast_df = pd.read_csv(forecasts)

    frame = merge_price_and_system(
        price_df,
        system_df,
        forecast_df=forecast_df,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output,
        index=False,
    )

    typer.echo(
        f"Wrote {len(frame)} merged observations to {output}"
    )

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

@app.command("tune-hgb")
def tune_hgb(
    input: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        Path("outputs/hgb_tuning")
    ),
    n_splits: int = typer.Option(
        3,
        min=2,
        max=10,
    ),
    target_mode: str = typer.Option(
        "level",
        help=(
            "Prediction target: "
            "'level' or 'residual'."
        ),
    ),
    derived_forecasts: str = typer.Option(
        "both",
        help=(
            "Derived SEMO forecast features: "
            "none, net, share, or both."
        ),
    ),
):
    """Tune HistGradientBoosting with chronological CV."""

    df = load_market_frame(
        input
    )

    results = tune_hist_gradient_boosting(
        df,
        n_splits=n_splits,
        target_mode=target_mode,
        derived_forecasts=derived_forecasts,
    )

    save_hgb_tuning_outputs(
        results,
        output_dir,
    )

    typer.echo(
        ""
    )

    typer.echo(
        "Top HGB configurations:"
    )

    typer.echo(
        results.head(
            10
        ).to_string(
            index=False
        )
    )

    typer.echo(
        ""
    )

    typer.echo(
        f"Saved tuning outputs to {output_dir}"
    )

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


@app.command("fetch-semo-forecasts")
def fetch_semo_forecasts(
    date_from: str = typer.Option(
        ...,
        help=("First SEM trading-day label, YYYY-MM-DD"),
    ),
    date_to: str = typer.Option(
        ...,
        help=("Final SEM trading-day label, YYYY-MM-DD"),
    ),
    output: Path = typer.Option(Path("data/processed/semo_forecasts.csv")),
    raw_dir: Path = typer.Option(
        Path("data/raw/semo_forecasts"),
        help=("Directory used to cache native SEMO forecast XML."),
    ),
):
    """Fetch leakage-safe NI demand and wind forecasts."""

    frame = fetch_point_in_time_forecasts(
        date_from,
        date_to,
        raw_dir=raw_dir,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output,
        index=False,
    )

    typer.echo(f"Wrote {len(frame)} forecast observations to {output}")

    typer.echo(f"Forecast range: {frame['timestamp'].min()} -> {frame['timestamp'].max()}")

    if "trade_date" in frame.columns:
        typer.echo(f"Trading-day range: {frame['trade_date'].min()} -> {frame['trade_date'].max()}")

    if "load_forecast_published_at" in frame.columns:
        typer.echo(f"Latest load forecast used: {frame['load_forecast_published_at'].max()}")

    if "wind_forecast_published_at" in frame.columns:
        typer.echo(f"Latest wind forecast used: {frame['wind_forecast_published_at'].max()}")


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
