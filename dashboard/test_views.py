"""
Test views for debugging drought monitoring functionality
Uses real Google Earth Engine data - no mock data.
"""

from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime, timedelta


class TestDroughtMonitoringView(View):
    """Test endpoint for drought monitoring with real GEE data"""
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def get(self, request):
        """Test drought monitoring with real satellite data"""
        try:
            import ee
            from .gee_auth import authenticate_gee
            from .gee_data_processing import get_tanzania_boundary
            
            authenticate_gee()
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Test real GEE connectivity
            tanzania = get_tanzania_boundary()
            region_count = tanzania.size().getInfo()
            
            # Test NDVI data availability
            s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(tanzania.geometry()) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            
            s2_count = s2_collection.size().getInfo()
            
            return JsonResponse({
                'status': 'success',
                'test_type': 'drought_monitoring_real_data',
                'data': {
                    'gee_authenticated': True,
                    'tanzania_boundary_features': region_count,
                    'sentinel2_images_available': s2_count,
                    'date_range': {'start': start_date, 'end': end_date},
                    'message': 'Real GEE data connection verified successfully'
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Test failed: {str(e)}'
            }, status=500)


class TestAgriculturalIntelligenceView(View):
    """Test endpoint for agricultural intelligence with real GEE data"""
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def get(self, request):
        """Test agricultural intelligence with real satellite data"""
        try:
            import ee
            from .gee_auth import authenticate_gee
            from .gee_data_processing import get_tanzania_boundary
            
            authenticate_gee()
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            tanzania = get_tanzania_boundary()
            geometry = tanzania.geometry()
            
            # Test NDVI computation
            s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(geometry) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            
            image_count = s2_collection.size().getInfo()
            
            if image_count > 0:
                ndvi = s2_collection.median().normalizedDifference(['B8', 'B4']).rename('NDVI')
                ndvi_stats = ndvi.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=5000,
                    maxPixels=1e8
                ).getInfo()
                
                avg_ndvi = ndvi_stats.get('NDVI', 0)
            else:
                avg_ndvi = None
            
            return JsonResponse({
                'status': 'success',
                'test_type': 'agricultural_intelligence_real_data',
                'data': {
                    'gee_authenticated': True,
                    'sentinel2_images': image_count,
                    'average_ndvi': round(avg_ndvi, 4) if avg_ndvi else None,
                    'date_range': {'start': start_date, 'end': end_date},
                    'message': 'Real satellite data processing verified successfully'
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Test failed: {str(e)}'
            }, status=500)
