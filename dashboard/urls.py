from django.urls import path
from . import views
from . import test_views

app_name = 'dashboard'

urlpatterns = [
    path('', views.DashboardView.as_view(), name='index'),
    path('agricultural-advisory/', views.AgriculturalAdvisoryPageView.as_view(), name='agricultural_advisory'),
    path('drought-alerts/', views.DroughtAlertsView.as_view(), name='drought_alerts'),
    path('api/predictions/', views.PredictionAPIView.as_view(), name='predictions_api'),
    path('api/predictions-enhanced/', views.EnhancedPredictionAPIView.as_view(), name='enhanced_prediction_api'),
    path('api/administrative-layers/', views.AdministrativeLayersView.as_view(), name='administrative_layers'),
    path('api/test-enhanced-forecast/', views.TestEnhancedForecastView.as_view(), name='test_enhanced_forecast'),
    path('api/historical/', views.HistoricalDataView.as_view(), name='historical_api'),
    path('api/administrative/', views.AdministrativeLayersView.as_view(), name='administrative_api'),
    path('api/raster-predictions/', views.RasterPredictionView.as_view(), name='raster_predictions_api'),
    path('api/regional-statistics/', views.RegionalStatisticsView.as_view(), name='regional_statistics_api'),
    path('api/agricultural-advisory/', views.AgriculturalAdvisoryView.as_view(), name='agricultural_advisory_api'),
    path('api/enhanced-drought-monitoring/', views.EnhancedDroughtMonitoringView.as_view(), name='enhanced_drought_monitoring_api'),
    path('api/agricultural-intelligence/', views.AgriculturalIntelligenceView.as_view(), name='agricultural_intelligence_api'),
    path('api/debug/', views.DebugView.as_view(), name='debug_api'),
    path('test-drought/', views.TestDroughtMonitoringView.as_view(), name='test_drought_monitoring'),
    # Test endpoints for debugging
    path('api/test-drought-monitoring/', test_views.TestDroughtMonitoringView.as_view(), name='test_drought_monitoring'),
    path('api/test-agricultural-intelligence/', test_views.TestAgriculturalIntelligenceView.as_view(), name='test_agricultural_intelligence'),
]

