# Validation report

Validated on 18 October 2026 without modifying either production model.

## Passed

- Shared source, scripts, and Streamlit application compile as Python.
- All code cells in the separate EDA notebook, separate preprocessing notebook, and four numbered training notebooks compile.
- Each training notebook includes a held-out sample showing pollutant inputs, actual AQI, predicted AQI and absolute error after model fitting.
- The modeling dataset reproduces 61,404 training rows and 15,850 test rows.
- `data/raw/station_day.csv` is the only included and authoritative modeling dataset.
- Train/test dates are disjoint (zero date overlap).
- Report-only preprocessing audits pass: all predictors are numeric, train/test target distributions are displayed, and identical labeled observations do not cross the split.
- Stored best model is Random Forest, matching the included comparison metrics.
- Production regression and classification pipelines both load and return three smoke-test predictions.
- SHA-256 hashes of both organized model files exactly match the supplied originals.
- Notebook files are valid notebook-format JSON.

## Environment note

The supplied sklearn pipelines were serialized with scikit-learn 1.8.0. The validation computer had 1.9.1, so sklearn emitted a version warning, but model loading and inference succeeded. `requirements.txt` pins 1.8.0 for closer compatibility. XGBoost also warned that pickle is less portable across library versions; the original classifier was preserved rather than converted or retrained.

## Not fully exercised

- The full four-model benchmark was not rerun because the included benchmark already has successful outputs and RBF-SVR on roughly 61k training rows is intentionally expensive. Use `scripts/benchmark_regressors.py` to recompute into `analytics/recomputed/`.
- A local Streamlit process was launched for a startup attempt, but this sandbox blocked the localhost socket request. The application compiles and its models passed direct inference; final browser interaction, responsive layout, and download-button checks remain manual deployment tests.

