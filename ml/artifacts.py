"""Saving / loading the trained pipeline (+ metadata) with Joblib."""
import json
from pathlib import Path

import joblib

from .config import META_FILE, MODEL_FILE, get_model_dir


class ModelNotAvailableError(Exception):
    """Trained model is missing or unusable. Message is safe to show to users."""


_CACHE = {}  # (path, mtime) -> (pipeline, meta): loaded once, not per request


def save_artifact(pipeline, meta, model_dir=None):
    model_dir = Path(model_dir or get_model_dir())
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / MODEL_FILE, compress=3)
    (model_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _CACHE.clear()


def load_artifact(model_dir=None):
    import sklearn

    model_dir = Path(model_dir or get_model_dir())
    model_path = model_dir / MODEL_FILE
    meta_path = model_dir / META_FILE
    if not model_path.exists() or not meta_path.exists():
        raise ModelNotAvailableError(
            "The prediction model has not been trained yet. "
            "Run 'python ml/train_model.py' (see README) and restart the server."
        )
    key = (str(model_path), model_path.stat().st_mtime_ns)
    if key in _CACHE:
        return _CACHE[key]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        trained_with = meta.get("sklearn_version", "")
        if trained_with.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            raise ModelNotAvailableError(
                f"The model was trained with scikit-learn {trained_with} but this "
                f"server runs {sklearn.__version__}. Please retrain the model."
            )
        pipeline = joblib.load(model_path)
    except ModelNotAvailableError:
        raise
    except Exception as exc:  # corrupted file etc. - never leak internals
        raise ModelNotAvailableError(
            "The model file could not be loaded. Please retrain the model."
        ) from exc
    _CACHE.clear()
    _CACHE[key] = (pipeline, meta)
    return pipeline, meta
