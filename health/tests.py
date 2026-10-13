from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from ml.testing_utils import VALID_ASSESSMENT, TempModelMixin

from .forms import AssessmentForm
from .models import Assessment


def make_user(name="alice", password="S3cure-pass-phrase"):
    return User.objects.create_user(name, f"{name}@example.com", password)


class FormValidationTests(TestCase):
    def test_valid_form_computes_bmi(self):
        f = AssessmentForm(VALID_ASSESSMENT)
        self.assertTrue(f.is_valid(), f.errors)
        self.assertAlmostEqual(f.computed_bmi, 27.7, places=1)

    def test_optional_fields_may_be_blank(self):
        data = {**VALID_ASSESSMENT, "fruit_veg_servings": "", "fried_food_frequency": ""}
        self.assertTrue(AssessmentForm(data).is_valid())

    def test_required_field_missing(self):
        data = dict(VALID_ASSESSMENT)
        del data["smoking"]
        f = AssessmentForm(data)
        self.assertFalse(f.is_valid())
        self.assertIn("smoking", f.errors)

    def test_age_range(self):
        for bad in (17, 121):
            self.assertIn("age", AssessmentForm({**VALID_ASSESSMENT, "age": bad}).errors)

    def test_days_range(self):
        self.assertIn("physical_health_days",
                      AssessmentForm({**VALID_ASSESSMENT, "physical_health_days": 31}).errors)

    def test_unsupported_choice_rejected(self):
        self.assertIn("smoking", AssessmentForm({**VALID_ASSESSMENT, "smoking": "Sometimes"}).errors)

    def test_impossible_bmi_rejected(self):
        f = AssessmentForm({**VALID_ASSESSMENT, "height_cm": 100, "weight_kg": 300})
        self.assertFalse(f.is_valid())
        self.assertTrue(f.non_field_errors())


class AssessmentFlowTests(TempModelMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.user = make_user()
        self.client.force_login(self.user)

    def test_login_required(self):
        self.client.logout()
        for name in ("health:new", "health:history", "health:dashboard"):
            r = self.client.get(reverse(name))
            self.assertEqual(r.status_code, 302)
            self.assertIn("/accounts/login/", r["Location"])

    def test_submit_saves_prediction_and_recommendations(self):
        r = self.client.post(reverse("health:new"), VALID_ASSESSMENT)
        a = Assessment.objects.get(user=self.user)
        self.assertRedirects(r, reverse("health:detail", args=[a.pk]))
        self.assertEqual(a.model_version, "test-synthetic-model")
        self.assertIn(a.risk_category, {"lower", "borderline", "elevated"})
        self.assertEqual(a.age_category, "50-54")
        self.assertTrue(a.recommendations)
        self.assertEqual(a.recommendations[-1]["id"], "disclaimer")

    def test_invalid_submission_not_saved(self):
        r = self.client.post(reverse("health:new"), {**VALID_ASSESSMENT, "age": 5})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(Assessment.objects.count(), 0)

    def test_history_and_dashboard_show_own_data(self):
        self.client.post(reverse("health:new"), VALID_ASSESSMENT)
        self.client.post(reverse("health:new"), {**VALID_ASSESSMENT, "weight_kg": 80})
        self.assertContains(self.client.get(reverse("health:history")), "Assessment history")
        r = self.client.get(reverse("health:dashboard"))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "bmiChart")

    def test_delete_assessment(self):
        self.client.post(reverse("health:new"), VALID_ASSESSMENT)
        a = Assessment.objects.get()
        self.client.post(reverse("health:delete", args=[a.pk]))
        self.assertEqual(Assessment.objects.count(), 0)


class AccessControlTests(TempModelMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.alice, self.bob = make_user("alice"), make_user("bob")
        self.client.force_login(self.alice)
        self.client.post(reverse("health:new"), VALID_ASSESSMENT)
        self.alice_assessment = Assessment.objects.get(user=self.alice)
        self.client.logout()
        self.client.force_login(self.bob)

    def test_other_user_cannot_view_detail(self):
        r = self.client.get(reverse("health:detail", args=[self.alice_assessment.pk]))
        self.assertEqual(r.status_code, 404)

    def test_other_user_cannot_delete(self):
        self.client.post(reverse("health:delete", args=[self.alice_assessment.pk]))
        self.assertTrue(Assessment.objects.filter(pk=self.alice_assessment.pk).exists())

    def test_history_excludes_other_users_records(self):
        r = self.client.get(reverse("health:history"))
        self.assertEqual(len(r.context["page"].object_list), 0)

    def test_dashboard_excludes_other_users_records(self):
        r = self.client.get(reverse("health:dashboard"))
        self.assertIsNone(r.context["latest"])


class MissingModelTests(TempModelMixin, TestCase):
    with_model = False

    def test_clear_setup_error_and_nothing_saved(self):
        user = make_user()
        self.client.force_login(user)
        r = self.client.post(reverse("health:new"), VALID_ASSESSMENT)
        self.assertEqual(r.status_code, 503)
        self.assertContains(r, "not been trained", status_code=503)
        self.assertEqual(Assessment.objects.count(), 0)
