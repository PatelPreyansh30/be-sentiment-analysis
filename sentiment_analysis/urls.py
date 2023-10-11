# from django.contrib import admin
from django.urls import path
from django.contrib import admin
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('analyze/', views.SentimentAnalysisView.as_view()),
    path('bulk_analyze/', views.BulkSentimentAnalysisView.as_view()),
]
