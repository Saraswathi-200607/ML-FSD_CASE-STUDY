from django import forms

from accounts.forms import BootstrapMixin

from .models import Assessment


class AssessmentForm(BootstrapMixin, forms.ModelForm):
    class Meta:
        model = Assessment
        fields = [
            "age", "sex", "height_cm", "weight_kg",
            "smoking", "alcohol_drinking", "physical_activity",
            "stroke", "diff_walking", "asthma", "kidney_disease", "skin_cancer",
            "diabetic", "gen_health",
            "physical_health_days", "mental_health_days", "sleep_hours",
            "fruit_veg_servings", "fried_food_frequency",
        ]
        labels = {
            "smoking": "Have you smoked at least 100 cigarettes in your life?",
            "alcohol_drinking": "Heavy alcohol drinking?",
            "physical_activity": "Any physical activity or exercise in the past 30 days (outside your regular job)?",
            "stroke": "Ever told you had a stroke?",
            "diff_walking": "Serious difficulty walking or climbing stairs?",
            "asthma": "Ever told you had asthma?",
            "kidney_disease": "Ever told you had kidney disease (not kidney stones)?",
            "skin_cancer": "Ever told you had skin cancer?",
            "diabetic": "Diabetes status",
            "gen_health": "In general, your health is",
            "physical_health_days": "Days in the past 30 your physical health was not good",
            "mental_health_days": "Days in the past 30 your mental health was not good",
            "sleep_hours": "Average hours of sleep in 24 hours",
            "fruit_veg_servings": "Servings of fruit and vegetables per day (optional)",
            "fried_food_frequency": "How often do you eat fried / heavily processed food? (optional)",
        }
        help_texts = {
            "age": "18-120. Required.",
            "height_cm": "In centimetres, 100-250. Required.",
            "weight_kg": "In kilograms, 25-300. Required.",
            "alcohol_drinking": "Commonly defined as more than 14 drinks/week (men) or 7 (women).",
            "physical_health_days": "0-30. Include illness and injury.",
            "mental_health_days": "0-30. Include stress, depression and emotional problems.",
            "sleep_hours": "Whole hours, 1-24.",
            "fruit_veg_servings": "0-30. Used only for lifestyle tips, not by the prediction model.",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._style()
        for name in ("fruit_veg_servings", "fried_food_frequency"):
            self.fields[name].required = False
        self.fields["fried_food_frequency"].choices = [("", "Prefer not to say")] + list(
            self.fields["fried_food_frequency"].choices)[1:]
        self.computed_bmi = None

    def clean(self):
        data = super().clean()
        h, w = data.get("height_cm"), data.get("weight_kg")
        if h and w:
            bmi = round(w / ((h / 100) ** 2), 1)
            if not 10 <= bmi <= 80:
                raise forms.ValidationError(
                    f"The height and weight you entered give a BMI of {bmi}, which is "
                    "outside the supported range (10-80). Please check both values.")
            self.computed_bmi = bmi
        return data
