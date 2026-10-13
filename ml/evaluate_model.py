"""
Re-evaluate the SAVED model on the same held-out test split used in training.

    python ml/evaluate_model.py
    python ml/evaluate_model.py --csv data/heart_2020_cleaned.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import classification_report

from ml.artifacts import ModelNotAvailableError, load_artifact
from ml.config import DEFAULT_CSV
from ml.data import load_dataset, split_data
from ml.metrics import compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    args = p.parse_args()
    try:
        pipe, meta = load_artifact()
    except ModelNotAvailableError as exc:
        print(exc)
        sys.exit(1)
    df, _ = load_dataset(args.csv)
    _, X_test, _, y_test = split_data(df, meta["split"]["test_size"], meta["split"]["seed"])
    scores = pipe.predict_proba(X_test)[:, 1]
    print(f"Model: {meta['model_version']}   test rows: {len(X_test):,}\n")
    for name, thr in [("default 0.5", 0.5),
                      ("borderline", meta["thresholds"]["borderline"]),
                      ("elevated", meta["thresholds"]["elevated"])]:
        m = compute_metrics(y_test, scores, thr)
        print(f"--- threshold '{name}' = {thr:.4f}")
        print({k: round(v, 4) for k, v in m.items() if k != "confusion_matrix"})
        print("confusion matrix (tn, fp, fn, tp):", m["confusion_matrix"])
    print("\nClassification report at the 'elevated' threshold:")
    print(classification_report(y_test, (scores >= meta["thresholds"]["elevated"]).astype(int),
                                target_names=["No heart disease", "Heart disease"], zero_division=0))


if __name__ == "__main__":
    main()
