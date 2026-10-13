"""
Train, compare and save the heart-disease model.

Run from the project root (venv active):
    python ml/train_model.py                 # full run on data/heart_2020_cleaned.csv
    python ml/train_model.py --quick         # faster run on subsamples (for a first try)

What it does
  1. Loads + validates + cleans the CSV (ml/data.py)
  2. Splits train/test (stratified). The test set is NOT used for any choice.
  3. 5-fold cross-validation on the training data for 3 algorithms x 2
     imbalance strategies. Preprocessing is inside the Pipeline, so it is
     re-fitted inside every fold (no leakage).
  4. Selects the configuration with the best mean CV ROC-AUC.
  5. Chooses decision thresholds from the CV out-of-fold scores so that
     recall reaches the targets (e.g. 80%) - again without the test set.
  6. Fits every configuration on the full training set and evaluates ONCE on
     the untouched test set (comparison table + calibration check).
  7. Saves the selected pipeline (Joblib) + metadata JSON.
All numbers printed come from running the code on YOUR data.
"""
import argparse
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

# allow "python ml/train_model.py" from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml.artifacts import save_artifact
from ml.config import (
    CATEGORICAL_CHOICES, DEFAULT_CSV, EXCLUDED_COLUMNS, FEATURES,
    NUMERIC_FEATURES, NUMERIC_LIMITS, get_model_dir,
)
from ml.data import DatasetSchemaError, load_dataset, split_data
from ml.metrics import calibration_table, compute_metrics, threshold_for_recall
from ml.pipeline import ALGORITHMS, STRATEGIES, fit_pipeline, make_pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-rows", type=int, default=60000,
                   help="rows of the training set used for cross-validation (speed)")
    p.add_argument("--max-train-rows", type=int, default=None,
                   help="optionally cap rows for the final fits (speed)")
    p.add_argument("--recall-target", type=float, default=0.80)
    p.add_argument("--borderline-recall", type=float, default=0.95)
    p.add_argument("--quick", action="store_true", help="small subsamples, 3 folds")
    return p.parse_args()


def subsample(X, y, n, seed):
    if n is None or n >= len(X):
        return X, y
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n, random_state=seed, stratify=y)
    return X_s, y_s


def cross_validate_config(algorithm, strategy, X, y, folds, seed):
    """Own CV loop -> returns per-fold metrics and out-of-fold scores."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    fold_metrics = []
    y_arr = y.to_numpy()
    for tr, va in skf.split(X, y):
        pipe = make_pipeline(algorithm, strategy, seed)
        fit_pipeline(pipe, X.iloc[tr], y.iloc[tr], algorithm, strategy)
        s = pipe.predict_proba(X.iloc[va])[:, 1]
        oof[va] = s
        fold_metrics.append(compute_metrics(y_arr[va], s, 0.5))
    mean = lambda k: float(np.mean([m[k] for m in fold_metrics]))
    return {
        "roc_auc_mean": mean("roc_auc"),
        "roc_auc_std": float(np.std([m["roc_auc"] for m in fold_metrics])),
        "recall_mean@0.5": mean("recall"),
        "precision_mean@0.5": mean("precision"),
        "f1_mean@0.5": mean("f1"),
    }, oof


def main():
    args = parse_args()
    if args.quick:
        args.cv_folds, args.cv_rows, args.max_train_rows = 3, 20000, 60000

    print("=" * 78, "\n1) LOAD + VALIDATE DATA\n", "=" * 78, sep="")
    try:
        df, report = load_dataset(args.csv)
    except (FileNotFoundError, DatasetSchemaError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"  features used ({len(FEATURES)}): {FEATURES}")
    for col, why in EXCLUDED_COLUMNS.items():
        print(f"  {col}: {why}")

    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.seed)
    print(f"\n  train rows: {len(X_train):,} (positives {y_train.mean():.2%})"
          f"   test rows: {len(X_test):,} (positives {y_test.mean():.2%})")

    print("\n", "=" * 78, "\n2) CROSS-VALIDATION ON TRAINING DATA ONLY\n", "=" * 78, sep="")
    X_cv, y_cv = subsample(X_train, y_train, args.cv_rows, args.seed)
    print(f"  CV uses {len(X_cv):,} training rows, {args.cv_folds} folds\n")
    cv_results, oof_scores = {}, {}
    for algo, strat in product(ALGORITHMS, STRATEGIES):
        res, oof = cross_validate_config(algo, strat, X_cv, y_cv, args.cv_folds, args.seed)
        key = f"{algo}|{strat}"
        cv_results[key], oof_scores[key] = res, oof
        print(f"  {algo:<20} {strat:<9} AUC={res['roc_auc_mean']:.4f}±{res['roc_auc_std']:.4f} "
              f"recall@0.5={res['recall_mean@0.5']:.3f} precision@0.5={res['precision_mean@0.5']:.3f}")

    best_key = max(cv_results, key=lambda k: cv_results[k]["roc_auc_mean"])
    best_algo, best_strat = best_key.split("|")
    print(f"\n  SELECTED (highest mean CV ROC-AUC): {best_algo} / imbalance strategy '{best_strat}'")

    print("\n", "=" * 78, "\n3) DECISION THRESHOLDS (from out-of-fold scores, not the test set)\n", "=" * 78, sep="")
    oof = oof_scores[best_key]
    t_elev = threshold_for_recall(y_cv, oof, args.recall_target)
    t_border = threshold_for_recall(y_cv, oof, args.borderline_recall)
    t_border = min(t_border, t_elev)
    print(f"  'elevated'   score >= {t_elev:.4f}  (targets recall {args.recall_target:.0%} in CV)")
    print(f"  'borderline' score >= {t_border:.4f}  (targets recall {args.borderline_recall:.0%} in CV)")

    print("\n", "=" * 78, "\n4) FINAL FIT + ONE EVALUATION ON THE HELD-OUT TEST SET\n", "=" * 78, sep="")
    X_fit, y_fit = subsample(X_train, y_train, args.max_train_rows, args.seed)
    print(f"  final fits use {len(X_fit):,} training rows\n")
    comparison, fitted = [], {}
    for algo, strat in product(ALGORITHMS, STRATEGIES):
        pipe = make_pipeline(algo, strat, args.seed)
        fit_pipeline(pipe, X_fit, y_fit, algo, strat)
        scores = pipe.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test, scores, 0.5)
        fitted[f"{algo}|{strat}"] = (pipe, scores)
        comparison.append({"algorithm": algo, "imbalance": strat, **{
            k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_negative_rate")}})
    table = pd.DataFrame(comparison).round(4)
    print("  Test-set metrics at the default 0.5 cut-off (comparison only):\n")
    print(table.to_string(index=False))

    pipe, scores = fitted[best_key]
    m_elev = compute_metrics(y_test, scores, t_elev)
    m_border = compute_metrics(y_test, scores, t_border)
    print(f"\n  Selected model on test set at 'elevated' threshold {t_elev:.4f}:")
    print("   ", {k: round(v, 4) for k, v in m_elev.items() if k != 'confusion_matrix'})
    print("    confusion matrix:", m_elev["confusion_matrix"], "(tn, fp, fn, tp)")
    print(f"  At 'borderline' threshold {t_border:.4f}:")
    print("   ", {k: round(v, 4) for k, v in m_border.items() if k != 'confusion_matrix'})

    brier = float(brier_score_loss(y_test, scores))
    calib = calibration_table(y_test, scores)
    print(f"\n  Brier score (lower is better): {brier:.4f}")
    print("  Calibration by score decile (mean score vs. observed rate):")
    for row in calib:
        print(f"    score {row['mean_score']:.3f}  ->  observed {row['observed_rate']:.3f}  (n={row['n']})")
    print("  If these columns differ a lot, the score is NOT a real probability.")

    version = f"{best_algo}-{best_strat}-{datetime.now(timezone.utc):%Y%m%d%H%M}"
    meta = {
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": best_algo,
        "imbalance_strategy": best_strat,
        "target": "HeartDisease (self-reported, Yes=1)",
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "numeric_limits": NUMERIC_LIMITS,
        "categorical_choices": CATEGORICAL_CHOICES,
        "thresholds": {"elevated": t_elev, "borderline": t_border},
        "threshold_targets": {"elevated_recall": args.recall_target, "borderline_recall": args.borderline_recall},
        "data_report": report,
        "split": {"test_size": args.test_size, "seed": args.seed},
        "cv": {"folds": args.cv_folds, "rows": int(len(X_cv)), "results": cv_results},
        "test_comparison_at_0.5": comparison,
        "test_metrics_elevated_threshold": m_elev,
        "test_metrics_borderline_threshold": m_border,
        "test_metrics_at_0.5": compute_metrics(y_test, scores, 0.5),
        "brier_score": brier,
        "calibration_deciles": calib,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    save_artifact(pipe, meta)
    out = get_model_dir()
    table.to_csv(out / "model_comparison.csv", index=False)
    print(f"\nSaved: {out / 'model.joblib'}\n       {out / 'model_meta.json'}\n       {out / 'model_comparison.csv'}")
    print("Restart the Django server so it picks up the new model.")


if __name__ == "__main__":
    main()
