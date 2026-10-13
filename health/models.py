from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

from ml.config import CATEGORICAL_CHOICES

YES_NO = [("No", "No"), ("Yes", "Yes")]
SEX = [(c, c) for c in CATEGORICAL_CHOICES["Sex"]]
DIABETIC = [(c, c) for c in CATEGORICAL_CHOICES["Diabetic"]]
GEN_HEALTH = [(c, c) for c in CATEGORICAL_CHOICES["GenHealth"]]
FRIED = [("rarely", "Rarely (less than once a week)"),
         ("weekly", "A few times a week"),
         ("daily", "Daily or almost daily")]
RISK_CATEGORIES = [("lower", "Lower"), ("borderline", "Borderline"), ("elevated", "Elevated")]


class Assessment(models.Model):
    """One health assessment = inputs + model output + recommendations.
    Always query with user=request.user (see health/views.py)."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="assessments")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    # ---- inputs used by the ML model ----
    age = models.PositiveSmallIntegerField(
        "Age (years)", validators=[MinValueValidator(18), MaxValueValidator(120)])
    age_category = models.CharField(max_length=20, blank=True)
    sex = models.CharField(max_length=6, choices=SEX)
    height_cm = models.FloatField(
        "Height (cm)", validators=[MinValueValidator(100), MaxValueValidator(250)])
    weight_kg = models.FloatField(
        "Weight (kg)", validators=[MinValueValidator(25), MaxValueValidator(300)])
    bmi = models.FloatField(
        validators=[MinValueValidator(10), MaxValueValidator(80)])
    smoking = models.CharField(max_length=3, choices=YES_NO)
    alcohol_drinking = models.CharField(max_length=3, choices=YES_NO)
    stroke = models.CharField(max_length=3, choices=YES_NO)
    diff_walking = models.CharField(max_length=3, choices=YES_NO)
    physical_activity = models.CharField(max_length=3, choices=YES_NO)
    asthma = models.CharField(max_length=3, choices=YES_NO)
    kidney_disease = models.CharField(max_length=3, choices=YES_NO)
    skin_cancer = models.CharField(max_length=3, choices=YES_NO)
    diabetic = models.CharField(max_length=30, choices=DIABETIC)
    gen_health = models.CharField(max_length=12, choices=GEN_HEALTH)
    physical_health_days = models.PositiveSmallIntegerField(
        validators=[MaxValueValidator(30)])
    mental_health_days = models.PositiveSmallIntegerField(
        validators=[MaxValueValidator(30)])
    sleep_hours = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(24)])

    # ---- optional inputs used ONLY by the recommendation engine ----
    fruit_veg_servings = models.PositiveSmallIntegerField(
        null=True, blank=True, validators=[MaxValueValidator(30)])
    fried_food_frequency = models.CharField(max_length=10, choices=FRIED, blank=True)

    # ---- outputs ----
    model_score = models.FloatField(null=True, blank=True)
    risk_category = models.CharField(max_length=12, choices=RISK_CATEGORIES, blank=True)
    model_version = models.CharField(max_length=80, blank=True)
    recommendations = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["user", "-created_at"])]

    def __str__(self):
        return f"Assessment {self.pk} by {self.user} on {self.created_at:%Y-%m-%d}"

    def recommendation_input(self) -> dict:
        """Dict for recommendations.engine."""
        fields = ["age", "bmi", "smoking", "alcohol_drinking", "stroke", "diff_walking",
                  "physical_activity", "asthma", "kidney_disease", "skin_cancer",
                  "diabetic", "gen_health", "physical_health_days",
                  "mental_health_days", "sleep_hours", "fruit_veg_servings",
                  "fried_food_frequency"]
        return {f: getattr(self, f) for f in fields}
