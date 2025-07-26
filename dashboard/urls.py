from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.DashboardView.as_view(), name='index'),
    path('agricultural-advisory/', views.AgriculturalAdvisoryPageView.as_view(), name='agricultural_advisory'),
    path('api/predictions/', views.PredictionAPIView.as_view(), name='predictions_api'),
    path('api/historical/', views.HistoricalDataView.as_view(), name='historical_api'),
    path('api/administrative/', views.AdministrativeLayersView.as_view(), name='administrative_api'),
    path('api/raster-predictions/', views.RasterPredictionView.as_view(), name='raster_predictions_api'),
    path('api/regional-statistics/', views.RegionalStatisticsView.as_view(), name='regional_statistics_api'),
    path('api/agricultural-advisory/', views.AgriculturalAdvisoryView.as_view(), name='agricultural_advisory_api'),
    path('api/debug/', views.DebugView.as_view(), name='debug_api'),
]

