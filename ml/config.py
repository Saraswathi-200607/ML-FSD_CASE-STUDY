"""
Single source of truth for the ML feature schema.

The training script, the saved model, the Django form and the prediction
service ALL import from here, so the web form can never drift away from the
features the model was trained on.

Column names below are those of the Kaggle file `heart_2020_cleaned.csv`
(Personal Key Indicators of Heart Disease). `ml/data.py` verifies at runtime
that your CSV really contains them and stops with a clear message if not.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_CSV = DATA_DIR / "heart_2020_cleaned.csv"

MODEL_FILE = "model.joblib"
META_FILE = "model_meta.json"


def get_model_dir() -> Path:
    """Folder holding the trained artifacts. Override with env var MODEL_DIR
    (e.g. a Render persistent-disk mount path)."""
    return Path(os.environ.get("MODEL_DIR") or (BASE_DIR / "model_artifacts"))


TARGET = "HeartDisease"
TARGET_MAP = {"No": 0, "Yes": 1}

NUMERIC_FEATURES = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

# Hard limits: rows outside these are dropped in training and rejected at
# prediction time, so the model never sees values it was not trained on.
NUMERIC_LIMITS = {
    "BMI": (10.0, 80.0),
    "PhysicalHealth": (0, 30),
    "MentalHealth": (0, 30),
    "SleepTime": (1, 24),
}

YES_NO = ["No", "Yes"]
BINARY_FEATURES = [
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "PhysicalActivity",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

AGE_CATEGORIES = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
    "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older",
]
_AGE_LOWER_BOUNDS = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

CATEGORICAL_CHOICES = {
    "Sex": ["Female", "Male"],
    "AgeCategory": AGE_CATEGORIES,
    "Diabetic": ["No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)"],
    "GenHealth": ["Excellent", "Very good", "Good", "Fair", "Poor"],
}
for _col in BINARY_FEATURES:
    CATEGORICAL_CHOICES[_col] = YES_NO

CATEGORICAL_FEATURES = list(CATEGORICAL_CHOICES)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns in the CSV that we deliberately do NOT use, with the reason.
EXCLUDED_COLUMNS = {
    "Race": (
        "Excluded: US-specific race categories do not map to Indian users, "
        "and using them in a health-risk tool is ethically problematic."
    ),
}


def age_to_category(age: int) -> str:
    """Map an age in years to the dataset's AgeCategory label."""
    if age < 18:
        raise ValueError("The model only supports adults (18+).")
    label = AGE_CATEGORIES[0]
    for lower, cat in zip(_AGE_LOWER_BOUNDS, AGE_CATEGORIES):
        if age >= lower:
            label = cat
    return label
