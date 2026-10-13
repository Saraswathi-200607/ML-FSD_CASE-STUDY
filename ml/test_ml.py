import tempfile
from unittest import mock
import os

from django.test import SimpleTestCase

from ml import artifacts
from ml.config import age_to_category
from ml.predict import ModelNotAvailableError, predict_one, validate_inputs
from ml.testing_utils import TempModelMixin

GOOD = {
    "BMI": 27.0, "PhysicalHealth": 2, "MentalHealth": 3, "SleepTime": 7,
    "Smoking": "No", "AlcoholDrinking": "No", "Stroke": "No", "DiffWalking": "No",
    "PhysicalActivity": "Yes", "Asthma": "No", "KidneyDisease": "No",
    "SkinCancer": "No", "Sex": "Female", "AgeCategory": "50-54",
    "Diabetic": "No", "GenHealth": "Good",
}


class AgeCategoryTests(SimpleTestCase):
    def test_mapping(self):
        self.assertEqual(age_to_category(18), "18-24")
        self.assertEqual(age_to_category(24), "18-24")
        self.assertEqual(age_to_category(25), "25-29")
        self.assertEqual(age_to_category(52), "50-54")
        self.assertEqual(age_to_category(79), "75-79")
        self.assertEqual(age_to_category(95), "80 or older")

    def test_minor_rejected(self):
        with self.assertRaises(ValueError):
            age_to_category(17)


class PredictionTests(TempModelMixin, SimpleTestCase):
    def test_prediction_shape(self):
        r = predict_one(GOOD)
        self.assertTrue(0.0 <= r["score"] <= 1.0)
        self.assertIn(r["category"], {"lower", "borderline", "elevated"})
        self.assertEqual(r["model_version"], "test-synthetic-model")

    def test_model_loaded_once(self):
        predict_one(GOOD)
        with mock.patch("ml.artifacts.joblib.load") as loader:
            predict_one(GOOD)
            loader.assert_not_called()

    def test_invalid_category_rejected(self):
        with self.assertRaises(ValueError):
            validate_inputs({**GOOD, "Smoking": "Maybe"})

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            validate_inputs({**GOOD, "BMI": 500})

    def test_missing_feature_rejected(self):
        bad = dict(GOOD)
        del bad["Sex"]
        with self.assertRaises(ValueError):
            validate_inputs(bad)


class MissingModelTests(TempModelMixin, SimpleTestCase):
    with_model = False

    def test_missing_model_raises_clear_error(self):
        with self.assertRaises(ModelNotAvailableError):
            predict_one(GOOD)

    def test_corrupt_model_raises_clear_error(self):
        tmp = os.environ["MODEL_DIR"]
        open(os.path.join(tmp, "model.joblib"), "wb").write(b"not a model")
        open(os.path.join(tmp, "model_meta.json"), "w").write(
            '{"sklearn_version": "%s"}' % __import__("sklearn").__version__)
        artifacts._CACHE.clear()
        with self.assertRaises(ModelNotAvailableError):
            predict_one(GOOD)
