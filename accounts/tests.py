from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from health.models import Assessment

PW = "Correct-Horse-9-Battery"


class RegistrationTests(TestCase):
    def data(self, **kw):
        d = {"username": "newuser", "email": "new@example.com",
             "password1": PW, "password2": PW}
        d.update(kw)
        return d

    def test_register_logs_in_and_redirects(self):
        r = self.client.post(reverse("accounts:register"), self.data())
        self.assertRedirects(r, reverse("health:dashboard"))
        self.assertTrue(User.objects.filter(username="newuser").exists())
        self.assertIn("_auth_user_id", self.client.session)

    def test_password_is_hashed(self):
        self.client.post(reverse("accounts:register"), self.data())
        u = User.objects.get(username="newuser")
        self.assertNotEqual(u.password, PW)
        self.assertTrue(u.password.startswith(("pbkdf2_", "argon2", "bcrypt")))

    def test_weak_passwords_rejected(self):
        for weak in ("short1", "password1234", "1234567890123"):
            r = self.client.post(reverse("accounts:register"),
                                 self.data(password1=weak, password2=weak))
            self.assertEqual(r.status_code, 200)
            self.assertFalse(User.objects.filter(username="newuser").exists())

    def test_mismatched_passwords_rejected(self):
        r = self.client.post(reverse("accounts:register"), self.data(password2="Different-9-Pass!"))
        self.assertEqual(r.status_code, 200)
        self.assertFalse(User.objects.exists())

    def test_duplicate_email_rejected(self):
        User.objects.create_user("x", "new@example.com", PW)
        r = self.client.post(reverse("accounts:register"), self.data())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(User.objects.count(), 1)


class LoginLogoutTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("bob", "bob@example.com", PW)

    def test_login_success_and_failure(self):
        bad = self.client.post(reverse("accounts:login"), {"username": "bob", "password": "nope"})
        self.assertEqual(bad.status_code, 200)
        ok = self.client.post(reverse("accounts:login"), {"username": "bob", "password": PW})
        self.assertRedirects(ok, reverse("health:dashboard"))

    def test_logout_requires_post(self):
        self.client.force_login(self.user)
        self.assertEqual(self.client.get(reverse("accounts:logout")).status_code, 405)
        self.client.post(reverse("accounts:logout"))
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_profile_requires_login(self):
        self.assertEqual(self.client.get(reverse("accounts:profile")).status_code, 302)

    def test_profile_save(self):
        self.client.force_login(self.user)
        r = self.client.post(reverse("accounts:profile"),
                             {"age": 40, "sex": "Male", "height_cm": 170, "weight_kg": 70})
        self.assertRedirects(r, reverse("accounts:profile"))
        self.user.refresh_from_db()
        self.assertEqual(self.user.profile.age, 40)

    def test_delete_account_removes_data(self):
        self.client.force_login(self.user)
        Assessment.objects.create(
            user=self.user, age=40, sex="Male", height_cm=170, weight_kg=70, bmi=24.2,
            smoking="No", alcohol_drinking="No", stroke="No", diff_walking="No",
            physical_activity="Yes", asthma="No", kidney_disease="No", skin_cancer="No",
            diabetic="No", gen_health="Good", physical_health_days=0,
            mental_health_days=0, sleep_hours=7)
        self.client.post(reverse("accounts:delete"))
        self.assertFalse(User.objects.filter(pk=self.user.pk).exists())
        self.assertEqual(Assessment.objects.count(), 0)
