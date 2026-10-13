from django.urls import path

from . import views

app_name = "health"

urlpatterns = [
    path("dashboard/", views.dashboard, name="dashboard"),
    path("assessment/new/", views.assessment_new, name="new"),
    path("assessment/<int:pk>/", views.assessment_detail, name="detail"),
    path("assessment/<int:pk>/delete/", views.assessment_delete, name="delete"),
    path("history/", views.assessment_history, name="history"),
]
