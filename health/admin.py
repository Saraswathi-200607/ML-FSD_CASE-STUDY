from django.contrib import admin

from .models import Assessment


@admin.register(Assessment)
class AssessmentAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "created_at", "risk_category", "model_version")
    list_filter = ("risk_category",)
    # Health data is sensitive: the admin is read-only.
    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
