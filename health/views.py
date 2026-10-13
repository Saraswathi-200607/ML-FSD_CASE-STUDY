from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, redirect, render

from accounts.models import Profile
from ml.config import age_to_category
from prediction.services import ModelNotAvailableError, run_prediction
from recommendations.engine import generate_recommendations

from .forms import AssessmentForm
from .models import Assessment


def home(request):
    return render(request, "home.html")


def user_assessments(request):
    """THE ONLY entry point for reading assessments: always scoped to the
    logged-in user, so one user can never reach another user's records."""
    return Assessment.objects.filter(user=request.user)


@login_required
def dashboard(request):
    qs = user_assessments(request)
    latest = qs.first()
    recent = list(qs[1:6])
    history = list(qs.order_by("created_at")[:50])
    chart = {
        "labels": [a.created_at.strftime("%d %b %Y") for a in history],
        "bmi": [a.bmi for a in history],
        "weight": [a.weight_kg for a in history],
        "sleep": [a.sleep_hours for a in history],
        "physical_days": [a.physical_health_days for a in history],
    }
    return render(request, "dashboard/dashboard.html", {
        "latest": latest, "recent": recent, "chart": chart,
        "total": qs.count(),
    })


@login_required
def assessment_new(request):
    if request.method == "POST":
        form = AssessmentForm(request.POST)
        if form.is_valid():
            a = form.save(commit=False)
            a.user = request.user
            a.bmi = form.computed_bmi
            a.age_category = age_to_category(a.age)
            try:
                result = run_prediction(a)
            except ModelNotAvailableError as exc:
                # nothing is saved and no fake prediction is made
                return render(request, "health/assessment_form.html",
                              {"form": form, "setup_error": str(exc)}, status=503)
            a.model_score = result["score"]
            a.risk_category = result["category"]
            a.model_version = result["model_version"]
            a.recommendations = generate_recommendations(
                a.recommendation_input(), a.risk_category)
            a.save()
            messages.success(request, "Assessment saved.")
            return redirect("health:detail", pk=a.pk)
    else:
        initial = {}
        p = Profile.objects.filter(user=request.user).first()
        if p:
            initial = {k: v for k, v in {"age": p.age, "sex": p.sex or None,
                                         "height_cm": p.height_cm,
                                         "weight_kg": p.weight_kg}.items() if v}
        form = AssessmentForm(initial=initial)
    return render(request, "health/assessment_form.html", {"form": form})


@login_required
def assessment_detail(request, pk):
    a = get_object_or_404(user_assessments(request), pk=pk)  # 404 for other users' records
    return render(request, "health/assessment_detail.html", {"a": a})


@login_required
def assessment_history(request):
    page = Paginator(user_assessments(request), 10).get_page(request.GET.get("page"))
    return render(request, "health/history.html", {"page": page})


@login_required
def assessment_delete(request, pk):
    a = get_object_or_404(user_assessments(request), pk=pk)
    if request.method == "POST":
        a.delete()
        messages.success(request, "Assessment deleted.")
        return redirect("health:history")
    return render(request, "health/confirm_delete.html", {"a": a})


def error_404(request, exception):
    return render(request, "404.html", status=404)


def error_500(request):
    return render(request, "500.html", status=500)
