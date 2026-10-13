from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

from .models import Profile


class BootstrapMixin:
    """Add Bootstrap CSS classes to every widget."""

    def _style(self):
        for field in self.fields.values():
            w = field.widget
            css = "form-select" if isinstance(w, forms.Select) else "form-control"
            w.attrs["class"] = f"{w.attrs.get('class', '')} {css}".strip()


class RegisterForm(BootstrapMixin, UserCreationForm):
    email = forms.EmailField(required=True, help_text="Used only for your account.")

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._style()

    def clean_email(self):
        email = self.cleaned_data["email"].lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email


class LoginForm(BootstrapMixin, AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._style()


class ProfileForm(BootstrapMixin, forms.ModelForm):
    class Meta:
        model = Profile
        fields = ("age", "sex", "height_cm", "weight_kg")
        labels = {"age": "Age (years)", "height_cm": "Height (cm)", "weight_kg": "Weight (kg)"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._style()
