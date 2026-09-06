.PHONY: install test lint demo clean

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check .

demo:
	python -m ni_power_forecast generate-demo --days 300 --output data/demo/ni_power_demo.csv
	python -m ni_power_forecast backtest --input data/demo/ni_power_demo.csv --output-dir outputs/demo

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in map(pathlib.Path, ['.pytest_cache','.ruff_cache','outputs/demo'])]"
