from django.urls import path
from api import api_views

urlpatterns = [
    path('predict/', api_views.PredictAPIView.as_view(), name='predict'),
]