from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.shortcuts import redirect, render

from .forms import LoginForm, ProfileForm, RegisterForm
from .models import Profile


class UserLoginView(LoginView):
    template_name = "accounts/login.html"
    authentication_form = LoginForm
    redirect_authenticated_user = True


def register(request):
    if request.user.is_authenticated:
        return redirect("health:dashboard")
    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save(commit=False)
        user.email = form.cleaned_data["email"]
        user.save()
        login(request, user)
        messages.success(request, "Welcome! Your account has been created.")
        return redirect("health:dashboard")
    return render(request, "accounts/register.html", {"form": form})


@login_required
def profile(request):
    # Always the logged-in user's own profile - never looked up by an ID in the URL.
    obj, _ = Profile.objects.get_or_create(user=request.user)
    form = ProfileForm(request.POST or None, instance=obj)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Profile saved.")
        return redirect("accounts:profile")
    return render(request, "accounts/profile.html", {"form": form})


@login_required
def delete_account(request):
    """Right to deletion: removes the user and (via CASCADE) all their data."""
    if request.method == "POST":
        user = request.user
        logout(request)
        user.delete()
        messages.success(request, "Your account and all stored health data were deleted.")
        return redirect("home")
    return render(request, "accounts/delete_account.html")
