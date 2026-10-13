"""
Helpers for the AUTOMATED TESTS ONLY.

They build a tiny model from SYNTHETIC random data so tests can run without
the Kaggle dataset. The app itself never uses this module; real predictions
come only from a model trained with ml/train_model.py on real data.
"""
import os
import shutil
import tempfile
from datetime import datetime, timezone
from unittest import mock

import numpy as np
import pandas as pd
import sklearn

from . import artifacts
from .config import CATEGORICAL_CHOICES, FEATURES, NUMERIC_FEATURES, NUMERIC_LIMITS
from .pipeline import fit_pipeline, make_pipeline


def make_synthetic_frame(n=800, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "BMI": rng.uniform(16, 45, n),
        "PhysicalHealth": rng.integers(0, 31, n),
        "MentalHealth": rng.integers(0, 31, n),
        "SleepTime": rng.integers(4, 11, n),
    })
    for col, choices in CATEGORICAL_CHOICES.items():
        df[col] = rng.choice(choices, n)
    risk = 0.05 * df["BMI"] + 0.04 * df["PhysicalHealth"] + (df["Smoking"] == "Yes") * 1.0
    p = 1 / (1 + np.exp(-(risk - risk.mean()) * 1.5 - 1.5))
    y = (rng.random(n) < p).astype(int)
    return df[FEATURES], y


def train_tiny_artifact(model_dir):
    X, y = make_synthetic_frame()
    pipe = make_pipeline("logistic_regression", "balanced")
    fit_pipeline(pipe, X, pd.Series(y), "logistic_regression", "balanced")
    scores = pipe.predict_proba(X)[:, 1]
    meta = {
        "model_version": "test-synthetic-model",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "logistic_regression", "imbalance_strategy": "balanced",
        "features": FEATURES, "numeric_features": NUMERIC_FEATURES,
        "numeric_limits": NUMERIC_LIMITS, "categorical_choices": CATEGORICAL_CHOICES,
        "thresholds": {"elevated": float(np.quantile(scores, 0.8)),
                       "borderline": float(np.quantile(scores, 0.5))},
        "threshold_targets": {"elevated_recall": 0.8, "borderline_recall": 0.95},
        "test_metrics_elevated_threshold": {"recall": 0.0, "precision": 0.0, "f1": 0.0,
                                            "roc_auc": 0.0, "accuracy": 0.0,
                                            "false_negative_rate": 0.0},
        "test_comparison_at_0.5": [], "brier_score": 0.0, "calibration_deciles": [],
        "data_report": {}, "sklearn_version": sklearn.__version__,
    }
    artifacts.save_artifact(pipe, meta, model_dir)


class TempModelMixin:
    """Mixin for TestCase: points MODEL_DIR at a temp folder with a tiny
    synthetic model (set with_model = False for the 'missing model' case)."""
    with_model = True

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.mkdtemp()
        if self.with_model:
            train_tiny_artifact(self._tmp)
        patcher = mock.patch.dict(os.environ, {"MODEL_DIR": self._tmp})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        artifacts._CACHE.clear()


VALID_ASSESSMENT = {
    "age": 52, "sex": "Male", "height_cm": 172, "weight_kg": 82,
    "smoking": "Yes", "alcohol_drinking": "No", "stroke": "No",
    "diff_walking": "No", "physical_activity": "No", "asthma": "No",
    "kidney_disease": "No", "skin_cancer": "No", "diabetic": "No",
    "gen_health": "Good", "physical_health_days": 3, "mental_health_days": 2,
    "sleep_hours": 6, "fruit_veg_servings": 2, "fried_food_frequency": "daily",
}
