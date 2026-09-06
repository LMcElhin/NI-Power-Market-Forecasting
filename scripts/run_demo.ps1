$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment Python not found at $Python. Run: py -m venv .venv"
}

Write-Host "Generating synthetic Northern Ireland power-market data..."

& $Python -m ni_power_forecast generate-demo `
    --days 300 `
    --output "$ProjectRoot\data\demo\ni_power_demo.csv"

if ($LASTEXITCODE -ne 0) {
    throw "Demo data generation failed."
}

Write-Host "Running walk-forward backtest..."

& $Python -m ni_power_forecast backtest `
    --input "$ProjectRoot\data\demo\ni_power_demo.csv" `
    --output-dir "$ProjectRoot\outputs\demo"

if ($LASTEXITCODE -ne 0) {
    throw "Backtest failed."
}

Write-Host ""
Write-Host "Demo complete."
Write-Host "Outputs: $ProjectRoot\outputs\demo"