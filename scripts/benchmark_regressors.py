"""Recompute the four-model benchmark without modifying deployed models."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aqi_pipeline import (  # noqa: E402
    build_regressor,
    evaluate_predictions,
    load_modeling_data,
    make_date_grouped_split,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analytics" / "recomputed",
        help="A separate directory is used by default so verified artifacts remain intact.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = make_date_grouped_split(load_modeling_data())
    results = []
    for name in ("KNN", "SVR (RBF)", "Random Forest", "XGBoost"):
        if name == "Random Forest":
            with (ROOT / "artifacts" / "best_model_reg.pkl").open("rb") as stream:
                model = pickle.load(stream)
        else:
            model = build_regressor(name)
            model.fit(split.X_train, split.y_train)
        predictions = model.predict(split.X_test)
        results.append({"model": name, **evaluate_predictions(split.y_test, predictions)})
        print(name, results[-1], flush=True)

    payload = {
        "dataset": "station_day.csv (labeled, deduplicated)",
        "split": "GroupShuffleSplit by Date, 80/20, random_state=42",
        "train_rows": len(split.train),
        "test_rows": len(split.test),
        "results": results,
        "note": "Recomputed outputs; deployed model files were read-only.",
    }
    (args.output_dir / "regression_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(args.output_dir / "regression_comparison.csv", index=False)


if __name__ == "__main__":
    main()

