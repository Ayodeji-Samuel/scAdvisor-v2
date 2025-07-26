from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from datetime import datetime, timedelta
import json

# Safe imports with fallbacks
try:
    import ee
except ImportError:
    ee = None

from .models import Prediction, PredictionMetadata
from .gee_auth import authenticate_gee
from .gee_data_processing import get_tanzania_boundary
from .gee_ml_models import train_flood_classifier, train_drought_classifier, predict_flood, predict_drought

# Safe imports for optional components
try:
    from .geoserver_utils import GeoServerManager, get_tanzania_administrative_layers
except ImportError:
    GeoServerManager = None
    get_tanzania_administrative_layers = None

try:
    from .raster_prediction import RasterPredictor
except ImportError:
    RasterPredictor = None

class DashboardView(LoginRequiredMixin, View):
    """Main dashboard view - requires authentication"""
    
    def get(self, request):
        """Render the main dashboard page"""
        return render(request, 'dashboard/index.html')

class PredictionAPIView(View):
    """API view for getting predictions - public access for map data"""
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def get(self, request):
        """Get predictions for all forecast periods"""
        try:
            # Initialize GEE
            authenticate_gee()
            
            # Get Tanzania boundary
            tanzania_boundary = get_tanzania_boundary().geometry()
            
            # Train classifiers (in production, these would be pre-trained and cached)
            flood_classifier = train_flood_classifier(tanzania_boundary)
            drought_classifier = train_drought_classifier(tanzania_boundary)
            
            # Generate predictions for all forecast periods
            today = datetime.now().date()
            forecast_periods = [0, 7, 14, 21]
            
            predictions_data = {
                'flood': {},
                'drought': {}
            }
            
            for period in forecast_periods:
                target_date = today + timedelta(days=period)
                
                # Generate flood prediction
                flood_prediction = predict_flood(flood_classifier, target_date, tanzania_boundary)
                flood_tile_url = self._get_tile_url(flood_prediction, 'flood')
                
                # Generate drought prediction
                drought_prediction = predict_drought(drought_classifier, target_date, tanzania_boundary)
                drought_tile_url = self._get_tile_url(drought_prediction, 'drought')
                
                # Store in database
                flood_pred, created = Prediction.objects.get_or_create(
                    prediction_type='flood',
                    forecast_period=period,
                    prediction_date=target_date,
                    defaults={
                        'tile_url': flood_tile_url,
                        'severity_level': 'moderate'  # Placeholder
                    }
                )
                
                drought_pred, created = Prediction.objects.get_or_create(
                    prediction_type='drought',
                    forecast_period=period,
                    prediction_date=target_date,
                    defaults={
                        'tile_url': drought_tile_url,
                        'severity_level': 'moderate'  # Placeholder
                    }
                )
                
                predictions_data['flood'][period] = {
                    'tile_url': flood_tile_url,
                    'date': target_date.strftime('%Y-%m-%d'),
                    'severity': flood_pred.severity_level
                }
                
                predictions_data['drought'][period] = {
                    'tile_url': drought_tile_url,
                    'date': target_date.strftime('%Y-%m-%d'),
                    'severity': drought_pred.severity_level
                }
            
            return JsonResponse({
                'status': 'success',
                'data': predictions_data
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    def _get_tile_url(self, image, prediction_type):
        """Generate tile URL for GEE image"""
        try:
            if prediction_type == 'flood':
                vis_params = {
                    'min': 0,
                    'max': 1,
                    'palette': ['white', 'blue']
                }
            else:  # drought
                vis_params = {
                    'min': 0,
                    'max': 1,
                    'palette': ['green', 'yellow', 'red']
                }
            
            # Get the tile URL from GEE
            map_id = image.getMapId(vis_params)
            return map_id['tile_fetcher'].url_format
            
        except Exception as e:
            print(f"Error generating tile URL: {e}")
            return None

class HistoricalDataView(View):
    """API view for historical data - public access for map data"""
    
    def get(self, request):
        """Get historical predictions"""
        prediction_type = request.GET.get('type', 'flood')
        limit = int(request.GET.get('limit', 30))
        
        predictions = Prediction.objects.filter(
            prediction_type=prediction_type
        ).order_by('-created_at')[:limit]
        
        data = []
        for pred in predictions:
            data.append({
                'date': pred.prediction_date.strftime('%Y-%m-%d'),
                'forecast_period': pred.forecast_period,
                'severity': pred.severity_level,
                'affected_area': pred.affected_area_km2,
                'created_at': pred.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return JsonResponse({
            'status': 'success',
            'data': data
        })

class DebugView(View):
    """Debug view to check system status"""
    
    def get(self, request):
        """Get system status and available modules"""
        status = {
            'django': True,
            'earth_engine': ee is not None,
            'raster_predictor': RasterPredictor is not None,
            'geoserver_utils': GeoServerManager is not None,
        }
        
        # Test Earth Engine authentication
        ee_status = 'not_available'
        if ee:
            try:
                authenticate_gee()
                # Simple test
                test_point = ee.Geometry.Point([35.0, -6.0])
                ee_status = 'authenticated'
            except Exception as e:
                ee_status = f'error: {str(e)}'
        
        return JsonResponse({
            'status': 'success',
            'system_status': status,
            'earth_engine_status': ee_status,
            'available_endpoints': [
                '/api/predictions/',
                '/api/administrative/',
                '/api/raster-predictions/',
                '/api/regional-statistics/',
                '/api/historical/',
                '/api/debug/'
            ]
        })

class AdministrativeLayersView(View):
    """API view for administrative layers - public access for map data"""
    
    def get(self, request):
        """Get available administrative layers"""
        try:
            if not get_tanzania_administrative_layers:
                return JsonResponse({
                    'status': 'error',
                    'message': 'GeoServer utilities not available',
                    'layers': []
                })
            
            layers = get_tanzania_administrative_layers()
            return JsonResponse({
                'status': 'success',
                'layers': layers
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e),
                'layers': []
            }, status=500)

class RasterPredictionView(View):
    """API view for raster-based predictions - public access for map data"""
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def get(self, request):
        """Get raster prediction layers"""
        try:
            # Check if required modules are available
            if not ee:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Google Earth Engine not available'
                }, status=500)
            
            if not RasterPredictor:
                return JsonResponse({
                    'status': 'error',
                    'message': 'RasterPredictor module not available',
                    'fallback_data': self.get_fallback_predictions('', 0)
                })
            
            # Initialize GEE
            authenticate_gee()
            
            # Get parameters
            target_date = request.GET.get('date', datetime.now().strftime('%Y-%m-%d'))
            forecast_days = int(request.GET.get('forecast_days', 7))
            
            target_datetime = datetime.strptime(target_date, '%Y-%m-%d').date()
            
            # Create raster predictor
            predictor = RasterPredictor()
            
            # Generate prediction layers
            prediction_layers = predictor.create_prediction_layers(
                target_datetime, 
                forecast_days
            )
            
            return JsonResponse({
                'status': 'success',
                'data': prediction_layers,
                'metadata': {
                    'date': target_date,
                    'forecast_days': forecast_days,
                    'generated_at': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error generating predictions: {str(e)}',
                'debug_info': {
                    'target_date': request.GET.get('date', 'not provided'),
                    'forecast_days': request.GET.get('forecast_days', 'not provided')
                }
            }, status=500)
    
    def get_fallback_predictions(self, target_date, forecast_days):
        """Fallback prediction data when raster prediction fails"""
        return {
            'flood': {
                'tile_url': None,
                'legend': {
                    'title': 'Flood Risk (Fallback)',
                    'items': [
                        {'color': '#0066CC', 'label': 'Very Low'},
                        {'color': '#66B2FF', 'label': 'Low'},
                        {'color': '#FFFF66', 'label': 'Moderate'},
                        {'color': '#FF6600', 'label': 'High'},
                        {'color': '#CC0000', 'label': 'Very High'}
                    ]
                }
            },
            'drought': {
                'tile_url': None,
                'legend': {
                    'title': 'Drought Risk (Fallback)',
                    'items': [
                        {'color': '#228B22', 'label': 'Very Low'},
                        {'color': '#90EE90', 'label': 'Low'},
                        {'color': '#FFFF66', 'label': 'Moderate'},
                        {'color': '#FF6600', 'label': 'High'},
                        {'color': '#8B0000', 'label': 'Very High'}
                    ]
                }
            }
        }

class RegionalStatisticsView(View):
    """API view for regional statistics - public access for map data"""
    
    def get(self, request):
        """Get statistics for each region/district"""
        try:
            # Initialize GEE
            authenticate_gee()
            
            region_type = request.GET.get('type', 'regions')  # regions or districts
            prediction_type = request.GET.get('prediction', 'flood')  # flood or drought
            target_date = request.GET.get('date', datetime.now().strftime('%Y-%m-%d'))
            forecast_days = int(request.GET.get('forecast_days', 0))
            
            if not RasterPredictor:
                return JsonResponse({
                    'status': 'error',
                    'message': 'RasterPredictor module not available'
                }, status=500)
            
            predictor = RasterPredictor()
            
            # Get administrative boundaries
            if region_type == 'districts':
                boundaries = predictor.get_tanzania_districts()
            else:
                boundaries = predictor.get_tanzania_regions()
            
            if not boundaries:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Could not load administrative boundaries'
                }, status=500)
            
            # Parse target date
            try:
                target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
            except:
                target_date_obj = datetime.now().date()
            
            # Generate prediction raster using the same method as the map
            current_date = datetime.now().date()
            historical_end_date = current_date
            historical_start_date = current_date - timedelta(days=30)
            
            start_date_str = historical_start_date.strftime('%Y-%m-%d')
            end_date_str = historical_end_date.strftime('%Y-%m-%d')
            
            # Create the same prediction raster that's used for the map
            if prediction_type == 'flood':
                raster = predictor.create_flood_risk_raster(start_date_str, end_date_str, forecast_days=forecast_days)
            else:
                raster = predictor.create_drought_risk_raster(start_date_str, end_date_str, forecast_days=forecast_days)
            
            if not raster:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Could not generate prediction raster'
                }, status=500)
            
            # Calculate actual zonal statistics
            statistics = predictor.calculate_regional_statistics(
                raster, boundaries, prediction_type, forecast_days
            )
            
            return JsonResponse({
                'status': 'success',
                'statistics': statistics,
                'metadata': {
                    'region_type': region_type,
                    'prediction_type': prediction_type,
                    'target_date': target_date,
                    'forecast_days': forecast_days,
                    'generated_at': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)


class AgriculturalAdvisoryView(View):
    """API view for agricultural advisory - public access for advisory data"""
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def get(self, request):
        """Get agricultural recommendations based on current predictions"""
        try:
            # Get parameters from request
            prediction_type = request.GET.get('prediction_type', 'drought')
            forecast_days = int(request.GET.get('forecast_days', 7))
            target_date_str = request.GET.get('target_date')
            region_name = request.GET.get('region', None)
            
            # Parse target date
            if target_date_str:
                target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
            else:
                target_date = datetime.now().date()
            
            # Initialize predictor
            if not RasterPredictor:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Raster prediction module not available'
                }, status=500)
            
            authenticate_gee()
            predictor = RasterPredictor()
            
            # Get agricultural recommendations
            recommendations = predictor.get_agricultural_recommendations(
                prediction_type, forecast_days, target_date, region_name
            )
            
            return JsonResponse({
                'status': 'success',
                'recommendations': recommendations,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'valid_until': (datetime.now() + timedelta(days=7)).isoformat(),
                    'disclaimer': 'These recommendations are based on predictive models and should be combined with local knowledge and current field conditions.'
                }
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)


class AgriculturalAdvisoryPageView(LoginRequiredMixin, View):
    """View for the agricultural advisory page - requires authentication"""
    
    def get(self, request):
        """Render the agricultural advisory page"""
        return render(request, 'dashboard/agricultural_advisory.html')

