from django.contrib import admin
from django.urls import include, path

from health import views as health_views

handler404 = "health.views.error_404"
handler500 = "health.views.error_500"

urlpatterns = [
    path("", health_views.home, name="home"),
    path("admin/", admin.site.urls),
    path("accounts/", include("accounts.urls")),
    path("health/", include("health.urls")),
    path("model/", include("prediction.urls")),
]
