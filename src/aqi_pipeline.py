"""Single source of truth for leakage-safe AQI regression experiments."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "station_day.csv"
FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene",
]
TARGET = "AQI"
BUCKET_TARGET = "AQI_Bucket"
RANDOM_STATE = 42


@dataclass(frozen=True)
class AQISplit:
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def X_train(self) -> pd.DataFrame:
        return self.train[FEATURES]

    @property
    def X_test(self) -> pd.DataFrame:
        return self.test[FEATURES]

    @property
    def y_train(self) -> pd.Series:
        return self.train[TARGET]

    @property
    def y_test(self) -> pd.Series:
        return self.test[TARGET]


def load_modeling_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the station-day source and apply only row-level eligibility rules.

    Missing predictor values are intentionally retained. Their medians must be
    learned from the training partition by the estimator Pipeline.
    """
    data = pd.read_csv(path)
    data = data.dropna(subset=[TARGET, BUCKET_TARGET]).copy()
    data = data.drop_duplicates(subset=FEATURES + [TARGET, BUCKET_TARGET])
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    return data.sort_values(["Date", "StationId"]).reset_index(drop=True)


def make_date_grouped_split(
    data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> AQISplit:
    """Split by whole dates so a date cannot occur in both partitions."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(data, groups=data["Date"]))
    split = AQISplit(data.iloc[train_idx].copy(), data.iloc[test_idx].copy())
    overlap = set(split.train["Date"]) & set(split.test["Date"])
    if overlap:
        raise AssertionError(f"Date leakage detected: {len(overlap)} overlapping dates")
    return split


def build_regressor(name: str) -> Pipeline:
    """Return the exact benchmark configuration for one requested regressor."""
    key = name.strip().lower().replace("_", " ")
    if key in {"knn", "knn regression"}:
        estimator = KNeighborsRegressor(
            n_neighbors=7, weights="distance", n_jobs=4
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    elif key in {"svr", "svr rbf", "svr (rbf)"}:
        estimator = SVR(
            kernel="rbf", C=100, epsilon=10, gamma="scale", cache_size=1000
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    elif key in {"random forest", "random forest regression"}:
        estimator = RandomForestRegressor(
            n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    elif key in {"xgboost", "xgboost regression"}:
        estimator = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=4,
            random_state=RANDOM_STATE,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    else:
        raise ValueError(f"Unknown regressor: {name}")
    return Pipeline(steps)


def evaluate_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, predictions))
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, predictions)),
    }


def verified_regression_metrics() -> dict[str, Any]:
    path = PROJECT_ROOT / "analytics" / "regression_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_production_models() -> tuple[Any, Any]:
    """Load, but never modify, the deployed regression and classification models."""
    with (PROJECT_ROOT / "artifacts" / "best_model_reg.pkl").open("rb") as stream:
        regressor = pickle.load(stream)
    with (PROJECT_ROOT / "artifacts" / "best_model_clf.pkl").open("rb") as stream:
        classifier = pickle.load(stream)
    return regressor, classifier

