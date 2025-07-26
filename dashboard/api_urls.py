from django.urls import path
from . import views

# API-only URL patterns (without 'api/' prefix)
urlpatterns = [
    path('predictions/', views.PredictionAPIView.as_view(), name='predictions_api'),
    path('historical/', views.HistoricalDataView.as_view(), name='historical_api'),
    path('administrative/', views.AdministrativeLayersView.as_view(), name='administrative_api'),
    path('raster-predictions/', views.RasterPredictionView.as_view(), name='raster_predictions_api'),
    path('regional-statistics/', views.RegionalStatisticsView.as_view(), name='regional_statistics_api'),
    path('agricultural-advisory/', views.AgriculturalAdvisoryView.as_view(), name='agricultural_advisory_api'),
    path('debug/', views.DebugView.as_view(), name='debug_api'),
]
