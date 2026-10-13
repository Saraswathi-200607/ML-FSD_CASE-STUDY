"""Evaluation helpers."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


def compute_metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_for_recall(y_true, scores, target_recall):
    """Largest score threshold whose recall is still >= target_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(thresholds[0])
    return float(thresholds[ok[-1]])


def calibration_table(y_true, scores, bins=10):
    """Mean predicted score vs. observed positive rate per score-decile."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(scores)
    rows = []
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        rows.append({
            "mean_score": float(scores[chunk].mean()),
            "observed_rate": float(y_true[chunk].mean()),
            "n": int(len(chunk)),
        })
    return rows
