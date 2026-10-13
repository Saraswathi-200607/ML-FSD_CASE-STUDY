from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Profile(models.Model):
    """Optional profile data used only to pre-fill the assessment form.
    Data minimisation: nothing here is required."""

    SEX_CHOICES = [("Female", "Female"), ("Male", "Male")]

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                related_name="profile")
    age = models.PositiveSmallIntegerField(
        null=True, blank=True, validators=[MinValueValidator(18), MaxValueValidator(120)])
    sex = models.CharField(max_length=6, choices=SEX_CHOICES, blank=True)
    height_cm = models.FloatField(null=True, blank=True,
                                  validators=[MinValueValidator(100), MaxValueValidator(250)])
    weight_kg = models.FloatField(null=True, blank=True,
                                  validators=[MinValueValidator(25), MaxValueValidator(300)])
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profile of {self.user}"
