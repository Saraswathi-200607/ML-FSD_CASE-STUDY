"""Inference: validate inputs, run the saved pipeline, return a result dict."""
import pandas as pd

from .artifacts import ModelNotAvailableError, load_artifact  # noqa: F401
from .config import CATEGORICAL_CHOICES, FEATURES, NUMERIC_FEATURES, NUMERIC_LIMITS


def validate_inputs(inputs: dict) -> dict:
    """Reject anything the model was not trained to handle."""
    clean = {}
    for col in FEATURES:
        if col not in inputs or inputs[col] in (None, ""):
            raise ValueError(f"Missing required model input: {col}")
        value = inputs[col]
        if col in NUMERIC_FEATURES:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{col} must be a number.")
            lo, hi = NUMERIC_LIMITS[col]
            if not lo <= value <= hi:
                raise ValueError(f"{col} must be between {lo} and {hi}.")
        else:
            if value not in CATEGORICAL_CHOICES[col]:
                raise ValueError(f"Unsupported value for {col}: {value!r}")
        clean[col] = value
    return clean


def categorize(score, meta):
    t = meta["thresholds"]
    if score >= t["elevated"]:
        return "elevated"
    if score >= t["borderline"]:
        return "borderline"
    return "lower"


def predict_one(inputs: dict) -> dict:
    pipeline, meta = load_artifact()
    clean = validate_inputs(inputs)
    frame = pd.DataFrame([clean])[FEATURES]
    score = float(pipeline.predict_proba(frame)[0, 1])
    return {
        "score": score,
        "category": categorize(score, meta),
        "model_version": meta["model_version"],
        "thresholds": meta["thresholds"],
    }
