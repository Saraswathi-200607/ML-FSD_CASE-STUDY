"""Glue between Django (assessment form/model) and the ML package."""
from ml.config import age_to_category
from ml.predict import ModelNotAvailableError, predict_one  # noqa: F401

# Django field name -> model feature name (ml/config.py FEATURES)
FIELD_TO_FEATURE = {
    "bmi": "BMI",
    "physical_health_days": "PhysicalHealth",
    "mental_health_days": "MentalHealth",
    "sleep_hours": "SleepTime",
    "sex": "Sex",
    "age_category": "AgeCategory",
    "smoking": "Smoking",
    "alcohol_drinking": "AlcoholDrinking",
    "stroke": "Stroke",
    "diff_walking": "DiffWalking",
    "physical_activity": "PhysicalActivity",
    "asthma": "Asthma",
    "kidney_disease": "KidneyDisease",
    "skin_cancer": "SkinCancer",
    "diabetic": "Diabetic",
    "gen_health": "GenHealth",
}


def build_model_input(assessment) -> dict:
    """Read the model's features from an Assessment (saved or unsaved)."""
    if not assessment.age_category:
        assessment.age_category = age_to_category(assessment.age)
    return {feat: getattr(assessment, field) for field, feat in FIELD_TO_FEATURE.items()}


def run_prediction(assessment) -> dict:
    """Raises ModelNotAvailableError if the model is not trained/loadable."""
    return predict_one(build_model_input(assessment))
