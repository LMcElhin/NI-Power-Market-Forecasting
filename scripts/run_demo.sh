#!/usr/bin/env bash
set -euo pipefail
python -m ni_power_forecast generate-demo --days 300 --output data/demo/ni_power_demo.csv
python -m ni_power_forecast backtest --input data/demo/ni_power_demo.csv --output-dir outputs/demo
echo "Demo complete. See outputs/demo"
