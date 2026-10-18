"""Fast integrity and inference checks for the organized project."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aqi_pipeline import (  # noqa: E402
    FEATURES,
    load_modeling_data,
    load_production_models,
    make_date_grouped_split,
    verified_regression_metrics,
)


def main() -> None:
    split = make_date_grouped_split(load_modeling_data())
    metrics = verified_regression_metrics()
    assert len(split.train) == metrics["train_rows"] == 61404
    assert len(split.test) == metrics["test_rows"] == 15850
    assert set(split.train["Date"]).isdisjoint(split.test["Date"])
    assert metrics["best_model_by_rmse"] == "Random Forest"

    regressor, classifier = load_production_models()
    sample = split.X_test.iloc[:3][FEATURES]
    reg_predictions = regressor.predict(sample)
    class_predictions = classifier.predict(sample)
    assert len(reg_predictions) == len(class_predictions) == 3
    assert all(value == value for value in reg_predictions)

    manifest = json.loads((ROOT / "artifacts" / "MODEL_MANIFEST.json").read_text())
    for filename, expected in manifest["sha256"].items():
        digest = hashlib.sha256((ROOT / "artifacts" / filename).read_bytes()).hexdigest().upper()
        assert digest == expected

    notebook_paths = [
        ROOT / "EDA" / "EDA.ipynb",
        ROOT / "preprocessing" / "preprocessing.ipynb",
        *sorted((ROOT / "training").glob("0[1-4]_*.ipynb")),
    ]
    notebook_count = 0
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")
        notebook_count += 1
    assert notebook_count == 6
    report = {
        "status": "PASS",
        "train_rows": len(split.train),
        "test_rows": len(split.test),
        "date_overlap": 0,
        "production_regression_smoke_predictions": reg_predictions.tolist(),
        "production_classifier_smoke_predictions": class_predictions.tolist(),
        "eda_preprocessing_and_training_notebooks_syntax_checked": notebook_count,
        "production_model_hashes_checked": 2,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

