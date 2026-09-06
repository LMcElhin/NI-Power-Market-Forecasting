# Data layout

- `demo/`: generated synthetic data that may be committed.
- `raw/`: downloaded SONI/SEMO source files; ignored by Git except `.gitkeep`.
- `processed/`: normalised modelling tables; ignored by Git except `.gitkeep`.

The modelling pipeline expects UTC timestamps and an hourly canonical table. See the root README for schema details.
