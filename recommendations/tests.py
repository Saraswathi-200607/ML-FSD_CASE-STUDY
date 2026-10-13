from django.test import SimpleTestCase

from .engine import generate_recommendations

IDS = lambda recs: {r["id"] for r in recs}


class RecommendationTests(SimpleTestCase):
    def test_smoker_gets_smoking_tip_with_reason(self):
        recs = generate_recommendations({"smoking": "Yes"})
        self.assertIn("smoking", IDS(recs))
        self.assertTrue(all(r["reason"] for r in recs))

    def test_inactive_user_gets_activity_tip(self):
        self.assertIn("activity", IDS(generate_recommendations({"physical_activity": "No"})))

    def test_bmi_uses_south_asian_cutoffs(self):
        self.assertIn("bmi-mid", IDS(generate_recommendations({"bmi": 24.0})))
        self.assertIn("bmi-high", IDS(generate_recommendations({"bmi": 28.0})))
        self.assertIn("bmi-ok", IDS(generate_recommendations({"bmi": 21.0})))
        self.assertIn("bmi-low", IDS(generate_recommendations({"bmi": 17.0})))

    def test_short_sleep(self):
        self.assertIn("sleep-low", IDS(generate_recommendations({"sleep_hours": 5})))

    def test_empty_input_does_not_crash(self):
        recs = generate_recommendations({})
        self.assertIn("disclaimer", IDS(recs))
        self.assertIn("diet-missing", IDS(recs))

    def test_missing_values_skip_rules(self):
        recs = generate_recommendations({"smoking": None, "bmi": "", "sleep_hours": None})
        self.assertNotIn("smoking", IDS(recs))
        self.assertNotIn("bmi-ok", IDS(recs))

    def test_model_flag_only_when_elevated_or_borderline(self):
        self.assertIn("model-flag", IDS(generate_recommendations({}, "elevated")))
        self.assertNotIn("model-flag", IDS(generate_recommendations({}, "lower")))

    def test_history_gives_followup_not_medication(self):
        recs = generate_recommendations({"diabetic": "Yes", "kidney_disease": "Yes"})
        text = " ".join(r["advice"].lower() for r in recs)
        self.assertIn("diabetes", IDS(recs))
        self.assertNotIn("mg", text.split())

    def test_high_priority_sorted_first(self):
        recs = generate_recommendations({"smoking": "Yes", "sleep_hours": 8})
        self.assertEqual(recs[0]["priority"], "high")
        self.assertEqual(recs[-1]["priority"], "info")
