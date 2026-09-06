# Northern Ireland Power Market Forecasting

A reproducible Python research project for **24-hour-ahead electricity-price forecasting** in the Northern Ireland / Single Electricity Market context.

The project is designed as a portfolio-quality quantitative research workflow rather than a single notebook. It includes:

- leakage-aware time-series feature engineering;
- a 24-hour persistence benchmark;
- gradient-boosted, random-forest and ridge models;
- an optional PyTorch MLP;
- expanding-window / walk-forward backtesting;
- forecast-error and directional-value diagnostics;
- synthetic Northern Ireland-like demo data for instant reproducibility;
- adapters for public SONI system-data workbooks;
- live SEMOpx day-ahead price retrieval through the public SEMO Report API;
- automated tests, linting, Docker and GitHub Actions.

> **Research disclaimer:** this repository is a forecasting demonstration, not a live trading system or investment recommendation. Synthetic benchmark results are clearly labelled. A production trading model would require strict point-in-time feature availability, exchange and settlement rules, transaction costs, liquidity, execution and risk constraints.

---

## Why Northern Ireland?

Northern Ireland participates in the all-island Single Electricity Market (SEM).

Short-term electricity prices are influenced by factors including:

- system demand;
- wind generation;
- renewable forecasts;
- interconnection;
- generation availability;
- fuel and carbon conditions;
- calendar effects;
- short-term supply/demand imbalances.

This makes the market a useful setting for time-series forecasting, model validation and quantitative research.

Public data sources used or supported by the project include:

- **SONI** — Northern Ireland system demand, wind generation, interconnection and historical system reports:  
  https://www.soni.ltd.uk/grid/system-and-renewable-data-reports

- **Smart Grid Dashboard** — real-time and historical Northern Ireland system views:  
  https://www.smartgriddashboard.com/ni/

- **SEMO / SEMOpx** — Single Electricity Market reports and day-ahead auction results:  
  https://www.semopx.com/market-data

- **SEMO Report API** — public market-report API documented by SEMO:  
  https://www.sem-o.com/sites/semo/files/documents/general-publications/SEMO-Website-Report-API.pdf

---

## Modelling question

For delivery hour `t`, predict the day-ahead electricity price using information that could plausibly have been available before the target period.

Candidate features include:

- price lags at 24 h, 48 h and 168 h;
- rolling price statistics shifted by at least 24 h;
- lagged actual NI demand;
- lagged actual NI wind generation;
- target-hour demand forecasts;
- target-hour wind forecasts;
- optional target-hour weather forecasts;
- cyclical hour, day-of-week and month features.

The implementation deliberately avoids using target-hour realised demand or wind as model inputs in the day-ahead experiment.

This distinction is important because using realised target-period information would introduce look-ahead bias.

---

## Repository structure

```text
.
|-- README.md
|-- pyproject.toml
|-- Makefile
|-- Dockerfile
|-- LICENSE
|-- config/
|   |-- model.yml
|   `-- soni_columns.example.yml
|-- data/
|   |-- demo/
|   |-- raw/
|   `-- processed/
|-- docs/
|   `-- demo_backtest.png
|-- outputs/
|-- scripts/
|   |-- run_demo.ps1
|   `-- run_demo.sh
|-- src/
|   `-- ni_power_forecast/
|       |-- __init__.py
|       |-- __main__.py
|       |-- backtest.py
|       |-- cli.py
|       |-- features.py
|       |-- forecast.py
|       |-- metrics.py
|       |-- models.py
|       |-- schema.py
|       |-- torch_model.py
|       |-- train.py
|       `-- data/
|           |-- __init__.py
|           |-- demo.py
|           |-- io.py
|           |-- semo.py
|           `-- soni.py
`-- tests/