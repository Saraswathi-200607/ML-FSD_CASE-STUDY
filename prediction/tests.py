from types import SimpleNamespace

from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from ml.testing_utils import TempModelMixin

from .services import build_model_input, run_prediction


def fake_assessment(**kw):
    base = dict(bmi=26.0, physical_health_days=1, mental_health_days=1, sleep_hours=7,
                sex="Male", age=45, age_category="", smoking="No", alcohol_drinking="No",
                stroke="No", diff_walking="No", physical_activity="Yes", asthma="No",
                kidney_disease="No", skin_cancer="No", diabetic="No", gen_health="Good")
    base.update(kw)
    return SimpleNamespace(**base)


class ServiceTests(TempModelMixin, SimpleTestCase):
    def test_build_input_maps_features_and_age_category(self):
        data = build_model_input(fake_assessment(age=52))
        self.assertEqual(data["AgeCategory"], "50-54")
        self.assertEqual(data["BMI"], 26.0)

    def test_run_prediction(self):
        r = run_prediction(fake_assessment())
        self.assertIn(r["category"], {"lower", "borderline", "elevated"})


class ModelInfoViewTests(TempModelMixin, TestCase):
    def test_page_with_model(self):
        r = self.client.get(reverse("prediction:info"))
        self.assertContains(r, "test-synthetic-model")


class ModelInfoMissingTests(TempModelMixin, TestCase):
    with_model = False

    def test_page_without_model_shows_setup_message(self):
        r = self.client.get(reverse("prediction:info"))
        self.assertContains(r, "not been trained")
