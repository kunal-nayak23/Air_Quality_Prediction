# AQI Academic Project — Organized and Viva Ready

This package reorganizes the AQI project into distinct notebooks, reusable ML code, deployment assets, datasets, verified analytics, and preserved model artifacts. It does not overwrite or silently retrain the production models.

## Folder map

```text
AQI_Academic_Organized_Viva_Ready/
├── EDA/            # EDA only: structure, missingness and visual exploration
├── preprocessing/  # cleaning, leakage-safe split, imputation and scaling only
├── training/       # four numbered AQI regression algorithm notebooks
├── src/            # shared loading, date-grouped splitting, pipelines, metrics
├── scripts/        # safe verification, candidate training, benchmark reproduction
├── deployment/     # Streamlit UI and launcher
├── artifacts/      # all .pkl files: preserved production models and manifest
├── data/raw/       # one authoritative dataset: station_day.csv
├── analytics/      # actual held-out metrics and plots
└── docs/           # concise viva explanation of model differences
```

## Correct academic workflow

1. Explore the untouched source data in `EDA/EDA.ipynb`.
2. Study data preparation in `preprocessing/preprocessing.ipynb`.
3. Study or run `training/01` through `training/04` for the four numerical AQI regressors.
4. Reuse `src/aqi_pipeline.py`; it prevents each notebook from implementing a different split.
5. Run `python scripts/verify_project.py` for a quick integrity and model-inference check.
6. Run `python scripts/benchmark_regressors.py` only when you intentionally want a full recomputation. RBF SVR can be slow and memory intensive.
7. Launch the UI with `python -m streamlit run deployment/app.py`.

## Leakage controls

- Modeling source: `data/raw/station_day.csv`; city aggregates are not mixed into training.
- Unused city-level, hourly, merged, and legacy-preprocessed CSVs are intentionally excluded from this submission to avoid ambiguity and reduce archive size.
- Rows without `AQI` or `AQI_Bucket` are excluded; 160 fully duplicated labeled station observations are removed.
- Predictors are only the 11 pollutant columns. `AQI` and `AQI_Bucket` never predict each other.
- `GroupShuffleSplit(random_state=42)` keeps every date wholly in train or test.
- Median imputation, and scaling for KNN/SVR, are fitted inside sklearn pipelines using training data only.
- The holdout is date-grouped random validation, not chronological forecasting or external validation.

## Verified results (not invented)

The included `analytics/regression_metrics.json` records 61,404 training and 15,850 test observations:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Random Forest | 22.0056 | 1512.5090 | **38.8910** | 0.915745 |
| XGBoost | 22.4356 | 1542.3800 | 39.2731 | 0.914081 |
| SVR (RBF) | 24.3471 | 1781.0388 | 42.2024 | 0.900787 |
| KNN | 24.6425 | 1775.4160 | 42.1357 | 0.901100 |

The production Random Forest model and separate XGBoost classifier are preserved byte-for-byte from the supplied archive. Candidate retraining writes only to `artifacts/candidates/`.

See `docs/MODEL_DIFFERENCES.md` for the regression-versus-classification viva explanation.

