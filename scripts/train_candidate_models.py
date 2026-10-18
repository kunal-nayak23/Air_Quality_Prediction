"""Train candidate production models in a safe, non-production output folder."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aqi_pipeline import (  # noqa: E402
    BUCKET_TARGET,
    FEATURES,
    build_regressor,
    evaluate_predictions,
    load_modeling_data,
    make_date_grouped_split,
)


def main() -> None:
    output = ROOT / "artifacts" / "candidates"
    output.mkdir(parents=True, exist_ok=True)
    split = make_date_grouped_split(load_modeling_data())

    regressor = build_regressor("Random Forest")
    regressor.fit(split.X_train, split.y_train)
    reg_predictions = regressor.predict(split.X_test)

    encoder = LabelEncoder().fit(
        np.concatenate([split.train[BUCKET_TARGET], split.test[BUCKET_TARGET]])
    )
    classifier = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=50, random_state=42, n_jobs=4, eval_metric="mlogloss"
        )),
    ])
    classifier.fit(split.X_train, encoder.transform(split.train[BUCKET_TARGET]))
    class_predictions = classifier.predict(split.X_test)

    with (output / "candidate_model_reg.pkl").open("wb") as stream:
        pickle.dump(regressor, stream)
    with (output / "candidate_model_clf.pkl").open("wb") as stream:
        pickle.dump(classifier, stream)
    metrics = {
        "regression": evaluate_predictions(split.y_test, reg_predictions),
        "classification": {
            "accuracy": float(accuracy_score(
                encoder.transform(split.test[BUCKET_TARGET]), class_predictions
            )),
            "macro_f1": float(f1_score(
                encoder.transform(split.test[BUCKET_TARGET]),
                class_predictions,
                average="macro",
            )),
        },
        "class_labels": encoder.classes_.tolist(),
        "warning": "Candidate artifacts are not deployed automatically.",
    }
    (output / "candidate_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Saved candidates to {output}")


if __name__ == "__main__":
    main()

