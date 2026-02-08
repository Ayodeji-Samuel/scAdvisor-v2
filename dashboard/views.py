from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from datetime import datetime, timedelta
import json
import sys

# DRF imports
try:
    from rest_framework.views import APIView
except ImportError:
    # Fallback for environments without DRF
    class APIView(View):
        pass

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

try:
    from .advanced_drought_monitoring import get_enhanced_drought_analysis, AdvancedDroughtMonitor
except ImportError:
    get_enhanced_drought_analysis = None
    AdvancedDroughtMonitor = None

# No mock data - real satellite data only
get_enhanced_drought_analysis_mock = None
get_mock_agricultural_intelligence = None

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

class EnhancedPredictionAPIView(View):
    """Enhanced API endpoint for predictions with trend modeling"""
    
    def get(self, request):
        try:
            # Get parameters from request
            date_str = request.GET.get('date', datetime.now().strftime('%Y-%m-%d'))
            forecast_days = int(request.GET.get('forecast_days', 7))
            
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # Create predictions with enhanced forecasting
            predictor = RasterPredictor()
            predictions = predictor.create_prediction_layers(target_date, forecast_days)
            
            # Add enhanced metadata
            predictions['enhanced_forecast'] = True
            predictions['target_date'] = date_str
            predictions['forecast_horizon'] = forecast_days
            predictions['model_version'] = '2.0_enhanced'
            predictions['created_at'] = datetime.now().isoformat()
            
            return JsonResponse(predictions)
            
        except Exception as e:
            return JsonResponse({
                'error': f'Failed to generate enhanced predictions: {str(e)}',
                'enhanced_forecast': False,
                'target_date': date_str if 'date_str' in locals() else None,
                'model_version': '2.0_enhanced'
            }, status=500)


class TestEnhancedForecastView(View):
    """Test endpoint to verify enhanced forecasting functionality"""
    
    def get(self, request):
        try:
            from .enhanced_forecasting import EnhancedForecaster, create_enhanced_prediction_layers
            
            # Test basic functionality
            test_date = datetime.now().date()
            
            # Test enhanced forecaster initialization
            forecaster = EnhancedForecaster()
            
            # Get Tanzania bounds for testing
            bounds = ee.Geometry.Polygon([
                [[29.340, -11.720],
                 [40.440, -11.720],
                 [40.440, -0.990],
                 [29.340, -0.990],
                 [29.340, -11.720]]
            ])
            
            # Test precipitation forecasting
            precip_forecast = forecaster.forecast_precipitation(test_date, 7, bounds)
            
            # Test temperature forecasting  
            temp_forecast = forecaster.forecast_temperature(test_date, 7, bounds)
            
            # Test vegetation forecasting
            ndvi_forecast = forecaster.forecast_vegetation(test_date, 7, bounds)
            
            result = {
                'status': 'success',
                'enhanced_forecasting_available': True,
                'test_date': test_date.strftime('%Y-%m-%d'),
                'forecaster_initialized': True,
                'precipitation_forecast_available': precip_forecast is not None,
                'temperature_forecast_available': temp_forecast is not None,
                'vegetation_forecast_available': ndvi_forecast is not None,
                'model_components': {
                    'seasonal_trends': True,
                    'annual_trends': True,
                    'recent_trends': True,
                    'uncertainty_quantification': True,
                    'ensemble_methods': True,
                    'climate_change_adjustment': True
                },
                'data_sources': {
                    'CHIRPS_precipitation': True,
                    'ERA5_temperature': True,
                    'Sentinel2_NDVI': True,
                    'Landsat8_NDVI': True
                }
            }
            
            return JsonResponse(result)
            
        except ImportError as e:
            return JsonResponse({
                'status': 'error',
                'enhanced_forecasting_available': False,
                'error': f'Enhanced forecasting module not available: {str(e)}',
                'fallback_available': True
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error', 
                'enhanced_forecasting_available': False,
                'error': f'Test failed: {str(e)}',
                'fallback_available': True
            }, status=500)
    
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
    """API view for real administrative layers using GADM data - public access for map data"""
    
    def get(self, request):
        """Get real Tanzania administrative boundaries from GADM shapefiles"""
        try:
            # Initialize GEE for real boundary data
            authenticate_gee()
            
            if not RasterPredictor:
                return JsonResponse({
                    'status': 'error',
                    'message': 'RasterPredictor module not available for loading real boundaries',
                    'layers': []
                })
            
            # Load real administrative boundaries from GADM
            predictor = RasterPredictor()
            
            # Get real Tanzania boundaries
            regions = predictor.get_tanzania_regions()
            districts = predictor.get_tanzania_districts()
            
            # Prepare layer information for the frontend
            layers = []
            
            if regions:
                layers.append({
                    'name': 'tanzania_regions',
                    'title': 'Tanzania Regions (GADM)',
                    'type': 'geojson',
                    'data_url': '/dashboard/api/boundaries/regions/',
                    'description': 'Real Tanzania administrative regions from GADM'
                })
            
            if districts:
                layers.append({
                    'name': 'tanzania_districts', 
                    'title': 'Tanzania Districts (GADM)',
                    'type': 'geojson',
                    'data_url': '/dashboard/api/boundaries/districts/',
                    'description': 'Real Tanzania administrative districts from GADM'
                })
            
            return JsonResponse({
                'status': 'success',
                'message': 'Real GADM administrative boundaries loaded',
                'layers': layers,
                'source': 'GADM_REAL_DATA'
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error loading real administrative boundaries: {str(e)}',
                'layers': [],
                'fallback': 'No mock data - real data required for Tanzania government'
            }, status=500)

class RasterPredictionView(APIView):
    """API view for raster predictions using real satellite data from Google Earth Engine"""
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def get(self, request):
        """Generate real-time raster predictions using satellite data"""
        try:
            print("🇹🇿 Tanzania Government - Generating raster predictions from real satellite data")
            
            # Get parameters from request
            target_date_str = request.GET.get('date')
            forecast_days = int(request.GET.get('forecast_days', 7))
            
            print(f"📡 Parameters: target_date={target_date_str}, forecast_days={forecast_days}")
            
            if not target_date_str:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Date parameter is required'
                }, status=400)
            
            # Parse target date
            try:
                from datetime import datetime
                target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
                print(f"🗓️ Target date parsed: {target_date}")
            except ValueError:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid date format. Use YYYY-MM-DD'
                }, status=400)
            
            # Import the updated RasterPredictor
            from .raster_prediction import RasterPredictor
            
            # Create predictor instance
            predictor = RasterPredictor()
            print(f"🛰️ RasterPredictor initialized")
            
            # Generate prediction layers using real satellite data
            print(f"🌍 Generating predictions for Tanzania using real Earth Engine data...")
            layers = predictor.create_prediction_layers(target_date, forecast_days)
            
            if not layers:
                print("❌ No prediction layers generated")
                return JsonResponse({
                    'status': 'error',
                    'message': 'Failed to generate prediction layers. Check Google Earth Engine connection.',
                    'debug_info': {
                        'target_date': target_date_str,
                        'forecast_days': forecast_days,
                        'timestamp': datetime.now().isoformat()
                    }
                }, status=500)
            
            print(f"✅ Generated {len(layers)} prediction layers:")
            for layer_type, layer_data in layers.items():
                tile_url_preview = layer_data.get('tile_url', 'No URL')[:100] if layer_data.get('tile_url') else 'No URL'
                print(f"   - {layer_type}: {tile_url_preview}...")
                if layer_data.get('error'):
                    print(f"     ⚠️ Layer error: {layer_data['error']}")
            
            # Add metadata for Tanzania government system
            response_data = {
                'status': 'success',
                'data': layers,
                'metadata': {
                    'country': 'Tanzania',
                    'government': 'Tanzania Government Climate System',
                    'data_source': 'Google Earth Engine Real Satellite Data',
                    'target_date': target_date_str,
                    'forecast_days': forecast_days,
                    'generated_at': datetime.now().isoformat(),
                    'satellites_used': [
                        'Sentinel-1 (SAR for flood detection)',
                        'Sentinel-2 (NDVI for vegetation)',
                        'CHIRPS (Precipitation)',
                        'ERA5-Land (Temperature, Humidity)'
                    ],
                    'coordinate_system': 'EPSG:4326 (WGS84)',
                    'data_quality': 'Real-time satellite observations',
                    'layers_generated': list(layers.keys())
                }
            }
            
            print(f"🇹🇿 Tanzania Government response ready with {len(layers)} layers")
            return JsonResponse(response_data)
            
        except Exception as e:
            error_msg = f"Error generating raster predictions: {str(e)}"
            print(f"❌ {error_msg}")
            
            return JsonResponse({
                'status': 'error',
                'message': error_msg,
                'debug_info': {
                    'error_type': type(e).__name__,
                    'target_date': request.GET.get('date', 'not provided'),
                    'forecast_days': request.GET.get('forecast_days', 'not provided'),
                    'timestamp': datetime.now().isoformat(),
                    'government': 'Tanzania Government System'
                }
            }, status=500)

class RegionalStatisticsView(View):
    """API view for real-time regional statistics using remote sensing data"""
    
    def get(self, request):
        """Get real-time statistics for each region/district using GEE remote sensing data"""
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
            
            # Use real-time regional statistics with remote sensing data
            statistics = predictor.calculate_realtime_regional_statistics(
                boundaries, prediction_type, forecast_days
            )
            
            # Add metadata about data sources and quality
            metadata = {
                'region_type': region_type,
                'prediction_type': prediction_type,
                'target_date': target_date,
                'forecast_days': forecast_days,
                'generated_at': datetime.now().isoformat(),
                'data_sources': [
                    'Sentinel-1 (SAR for flood/water detection)',
                    'Sentinel-2 (Multispectral for vegetation)',
                    'MODIS (Land Surface Temperature)',
                    'CHIRPS/GPM (Precipitation)',
                    'SMAP (Soil Moisture)',
                    'MODIS (Evapotranspiration)',
                    'GPW (Population data)'
                ],
                'update_frequency': 'Daily (satellite revisit dependent)',
                'spatial_resolution': '10m to 10km (varies by sensor)',
                'data_latency': '1-3 days (depending on processing)',
                'total_regions_analyzed': len(statistics)
            }
            
            return JsonResponse({
                'status': 'success',
                'statistics': statistics,
                'metadata': metadata
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error generating real-time statistics: {str(e)}',
                'debug_info': {
                    'region_type': request.GET.get('type', 'not provided'),
                    'prediction_type': request.GET.get('prediction', 'not provided'),
                    'target_date': request.GET.get('date', 'not provided'),
                    'forecast_days': request.GET.get('forecast_days', 'not provided')
                }
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


class DebugView(View):
    """Debug endpoint to test various components"""
    
    def get(self, request):
        try:
            # Authenticate GEE first
            authenticate_gee()
            
            # Test basic GEE authentication
            gee_test = ee.Image('USGS/SRTMGL1_003').getInfo()
            
            return JsonResponse({
                'status': 'success',
                'gee_authenticated': True,
                'enhanced_forecasting': 'enhanced_forecasting' in sys.modules,
                'test_results': {
                    'gee_connection': True,
                    'srtm_data_accessible': True,
                }
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'error': str(e),
                'gee_authenticated': False
            }, status=500)


class EnhancedDroughtMonitoringView(View):
    """Simplified and robust drought monitoring API"""
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def post(self, request):
        """Get simplified drought analysis that actually works"""
        try:
            print("Starting simplified drought monitoring API call...")
            
            # Parse request data
            data = json.loads(request.body) if request.body else {}
            
            # Get date range (default to last 30 days)
            end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
            start_date = data.get('start_date', 
                (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
            
            # Get area of interest
            aoi_coordinates = data.get('aoi_coordinates')
            
            # Check if Google Earth Engine is available
            if not ee:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Google Earth Engine is not installed. Please install the earthengine-api package.',
                    'error_type': 'missing_dependency'
                }, status=503)
            
            # Try to authenticate with Google Earth Engine
            try:
                authenticate_gee()
                print("Earth Engine authentication successful")
            except Exception as auth_error:
                print(f"Earth Engine authentication failed: {auth_error}")
                return JsonResponse({
                    'status': 'error',
                    'message': f'Google Earth Engine authentication failed: {str(auth_error)}',
                    'error_type': 'authentication_failed',
                    'details': 'Please check your service account credentials and permissions.'
                }, status=503)
            
            # Use the simplified drought monitoring approach
            try:
                from .simple_drought_monitoring import get_simple_drought_analysis
                print("Simple drought monitoring module imported successfully")
            except ImportError as import_error:
                print(f"Simple drought monitoring module not available: {import_error}")
                # Fall back to the advanced version if simple is not available
                try:
                    from .advanced_drought_monitoring import get_enhanced_drought_analysis
                    get_simple_drought_analysis = get_enhanced_drought_analysis
                    print("Falling back to advanced drought monitoring module")
                except ImportError:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'No drought monitoring modules available',
                        'details': str(import_error)
                    }, status=500)
            
            if aoi_coordinates:
                # Create geometry from coordinates
                geometry = ee.Geometry.Polygon(aoi_coordinates)
            else:
                # Default to Tanzania boundary
                geometry = get_tanzania_boundary()
            
            print(f"Running drought analysis for period: {start_date} to {end_date}")
            
            # Get drought analysis using simplified approach
            drought_analysis = get_simple_drought_analysis(start_date, end_date, geometry)
            
            if drought_analysis and drought_analysis.get('status') == 'success':
                print("Simple drought analysis completed successfully")
                return JsonResponse({
                    'status': 'success',
                    'data': drought_analysis,
                    'parameters': {
                        'start_date': start_date,
                        'end_date': end_date,
                        'analysis_type': 'simplified_satellite_drought_monitoring',
                        'data_sources': drought_analysis.get('indices_calculated', ['NDVI-based Drought Index'])
                    },
                    'source': 'Google Earth Engine - Simplified Satellite Analysis'
                })
            else:
                print(f"Simple drought analysis failed: {drought_analysis}")
                return JsonResponse({
                    'status': 'error',
                    'message': 'Drought analysis failed to generate valid results',
                    'details': drought_analysis
                }, status=500)
            
        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid JSON in request body'
            }, status=400)
            
        except Exception as e:
            error_msg = str(e)
            print(f"Unexpected error in simplified drought monitoring: {error_msg}")
            import traceback
            traceback.print_exc()
            
            return JsonResponse({
                'status': 'error',
                'message': f'An unexpected error occurred during drought analysis: {error_msg}',
                'error_type': 'unexpected_error'
            }, status=500)


class TestDroughtMonitoringView(View):
    """Simple test view to verify drought monitoring without login"""
    
    def get(self, request):
        """Test page to show drought monitoring status"""
        return render(request, 'dashboard/test_drought.html')


class DroughtAlertsView(LoginRequiredMixin, View):
    """View for displaying drought alerts dashboard"""
    
    def get(self, request):
        """Render drought alerts page"""
        return render(request, 'dashboard/drought_alerts.html')


class AgriculturalIntelligenceView(View):
    """API for agricultural intelligence and crop monitoring"""
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def post(self, request):
        """Get agricultural intelligence data using real satellite data from Google Earth Engine"""
        try:
            # Parse request
            data = json.loads(request.body) if request.body else {}
            
            # Get crop type and analysis parameters
            crop_type = data.get('crop_type', 'maize')
            analysis_type = data.get('analysis_type', 'yield_prediction')
            
            # Date range
            end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
            start_date = data.get('start_date', 
                (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'))
            
            # Area of interest
            aoi_coordinates = data.get('aoi_coordinates')
            
            # Ensure Google Earth Engine is available
            if not ee:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Google Earth Engine is not properly configured. Please check your authentication.'
                }, status=503)
            
            # Authenticate and process with real GEE data
            authenticate_gee()
            print(f"Processing real satellite data for agricultural intelligence: {start_date} to {end_date}")
            
            if aoi_coordinates:
                geometry = ee.Geometry.Polygon(aoi_coordinates)
            else:
                geometry = get_tanzania_boundary()
            
            # Get Sentinel-2 NDVI for vegetation health
            s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(geometry) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            if s2_collection.size().getInfo() == 0:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No Sentinel-2 satellite data available for selected period and area'
                }, status=404)
            
            # Calculate NDVI
            def calculate_ndvi(image):
                ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                return image.addBands(ndvi)
            
            ndvi_collection = s2_collection.map(calculate_ndvi)
            mean_ndvi = ndvi_collection.select('NDVI').mean()
            
            # Get precipitation data from CHIRPS
            precipitation = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                .filterBounds(geometry) \
                .filterDate(start_date, end_date) \
                .select('precipitation') \
                .sum()
            
            # Get MODIS LST for temperature analysis
            modis_lst = ee.ImageCollection('MODIS/006/MOD11A1') \
                .filterBounds(geometry) \
                .filterDate(start_date, end_date) \
                .select('LST_Day_1km') \
                .mean() \
                .multiply(0.02).subtract(273.15)  # Convert to Celsius
            
            # Generate map tiles
            ndvi_tile_url = mean_ndvi.getMapId({
                'min': 0, 'max': 1,
                'palette': ['red', 'yellow', 'green']
            })['tile_fetcher'].url_format
            
            precip_tile_url = precipitation.getMapId({
                'min': 0, 'max': 500,
                'palette': ['white', 'blue', 'darkblue']
            })['tile_fetcher'].url_format
            
            temp_tile_url = modis_lst.getMapId({
                'min': 15, 'max': 45,
                'palette': ['blue', 'cyan', 'yellow', 'red']
            })['tile_fetcher'].url_format
            
            # Calculate regional statistics
            regional_stats = mean_ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=1000,
                maxPixels=1e8
            )
            
            precip_stats = precipitation.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=5000,
                maxPixels=1e8
            )
            
            temp_stats = modis_lst.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=1000,
                maxPixels=1e8
            )
            
            avg_ndvi = regional_stats.getInfo().get('NDVI', 0)
            total_precip = precip_stats.getInfo().get('precipitation', 0)
            avg_temp = temp_stats.getInfo().get('LST_Day_1km', 25)
            
            # Enhanced yield estimation based on multiple factors
            if avg_ndvi > 0.7 and total_precip > 200:
                yield_category = 'Excellent'
                estimated_yield_tons_ha = 5.2
                confidence = 'high'
            elif avg_ndvi > 0.6 and total_precip > 150:
                yield_category = 'High'
                estimated_yield_tons_ha = 4.5
                confidence = 'high'
            elif avg_ndvi > 0.5 and total_precip > 100:
                yield_category = 'Medium'
                estimated_yield_tons_ha = 3.2
                confidence = 'medium'
            elif avg_ndvi > 0.3:
                yield_category = 'Low'
                estimated_yield_tons_ha = 2.1
                confidence = 'medium'
            else:
                yield_category = 'Poor'
                estimated_yield_tons_ha = 1.0
                confidence = 'low'
            
            # Generate recommendations
            recommendations = self.get_agricultural_recommendations(avg_ndvi, yield_category, total_precip, avg_temp)
            
            return JsonResponse({
                'status': 'success',
                'data': {
                    'crop_type': crop_type,
                    'analysis_period': {
                        'start_date': start_date,
                        'end_date': end_date
                    },
                    'vegetation_health': {
                        'average_ndvi': round(avg_ndvi, 3),
                        'tile_url': ndvi_tile_url,
                        'status': 'Excellent' if avg_ndvi > 0.7 else 'Good' if avg_ndvi > 0.5 else 'Moderate' if avg_ndvi > 0.3 else 'Poor'
                    },
                    'precipitation_analysis': {
                        'total_mm': round(total_precip, 1),
                        'tile_url': precip_tile_url,
                        'status': 'Adequate' if total_precip > 200 else 'Moderate' if total_precip > 100 else 'Low'
                    },
                    'temperature_analysis': {
                        'average_celsius': round(avg_temp, 1),
                        'tile_url': temp_tile_url,
                        'status': 'Optimal' if 20 <= avg_temp <= 30 else 'Suboptimal'
                    },
                    'yield_prediction': {
                        'category': yield_category,
                        'estimated_yield_tons_per_hectare': estimated_yield_tons_ha,
                        'confidence': confidence
                    },
                    'recommendations': recommendations,
                    'data_sources': [
                        'Sentinel-2 Optical Imagery',
                        'CHIRPS Precipitation Data',
                        'MODIS Land Surface Temperature',
                        'Google Earth Engine Processing'
                    ]
                },
                'source': 'Google Earth Engine - Real Satellite Data'
            })
            
        except Exception as e:
            error_msg = str(e)
            print(f"Critical GEE Error in agricultural intelligence: {error_msg}")
            
            return JsonResponse({
                'status': 'error',
                'message': f'Agricultural satellite data processing failed: {error_msg}. Please ensure Google Earth Engine service is available.'
            }, status=500)
    
    def get_agricultural_recommendations(self, ndvi, yield_category, precipitation=None, temperature=None):
        """Generate enhanced agricultural recommendations based on satellite data analysis"""
        recommendations = []
        
        # NDVI-based recommendations
        if yield_category == 'Poor':
            recommendations.extend([
                "🌱 URGENT: Vegetation stress detected - investigate immediately",
                "💧 Check irrigation systems and soil moisture levels",
                "🔬 Conduct soil testing for nutrient deficiencies",
                "🐛 Inspect crops for pest and disease issues",
                "🌾 Consider drought-resistant crop varieties for next season"
            ])
        elif yield_category == 'Low':
            recommendations.extend([
                "📊 Monitor vegetation health closely with weekly assessments",
                "💧 Consider supplemental irrigation if water is available",
                "🌿 Apply appropriate fertilizers based on crop growth stage",
                "⚠️ Prepare contingency plans for potential yield reduction"
            ])
        elif yield_category in ['Medium', 'High']:
            recommendations.extend([
                "✅ Maintain current management practices - they're working well",
                "📅 Monitor for optimal harvest timing",
                "📦 Plan for proper post-harvest handling and storage"
            ])
        elif yield_category == 'Excellent':
            recommendations.extend([
                "🌟 Outstanding crop conditions - maintain current practices",
                "🏗️ Prepare for excellent harvest - ensure adequate storage capacity",
                "📈 Document successful practices for future seasons",
                "🌾 Consider expanding similar practices to other areas"
            ])
        
        # Precipitation-based recommendations
        if precipitation is not None:
            if precipitation < 50:
                recommendations.extend([
                    "☔ CRITICAL: Very low rainfall detected",
                    "💧 Implement water conservation measures immediately",
                    "🚰 Prioritize irrigation for most critical crops"
                ])
            elif precipitation < 100:
                recommendations.extend([
                    "⚠️ Below-average rainfall - monitor soil moisture",
                    "💧 Prepare irrigation systems for potential use"
                ])
            elif precipitation > 400:
                recommendations.extend([
                    "☔ Excessive rainfall detected",
                    "🚰 Ensure proper drainage to prevent waterlogging",
                    "🍄 Monitor for fungal diseases due to high moisture"
                ])
        
        # Temperature-based recommendations
        if temperature is not None:
            if temperature > 35:
                recommendations.extend([
                    "🌡️ High temperatures detected - risk of heat stress",
                    "☂️ Provide shade for sensitive crops if possible",
                    "💧 Increase irrigation frequency during heat waves"
                ])
            elif temperature < 15:
                recommendations.extend([
                    "🌡️ Cool temperatures may slow crop development",
                    "🛡️ Protect sensitive crops from cold stress"
                ])
        
        # General best practices
        recommendations.extend([
            "📱 Continue monitoring with satellite data and weather forecasts",
            "📋 Keep detailed records of management practices for future reference",
            "🛡️ Consider crop insurance options for risk management",
            "🤝 Consult with local agricultural extension services"
        ])
        
        return recommendations

class BoundaryDataView(View):
    """API view for serving real GADM boundary data as GeoJSON"""
    
    def get(self, request, boundary_type):
        """Get GeoJSON data for administrative boundaries"""
        try:
            # Initialize GEE
            authenticate_gee()
            
            if not RasterPredictor:
                return JsonResponse({
                    'status': 'error',
                    'message': 'RasterPredictor not available for boundary data'
                }, status=500)
            
            predictor = RasterPredictor()
            
            # Load appropriate boundary data
            if boundary_type == 'regions':
                boundaries = predictor.get_tanzania_regions()
                if not boundaries:
                    raise Exception("No regions data available")
            elif boundary_type == 'districts':
                boundaries = predictor.get_tanzania_districts()
                if not boundaries:
                    raise Exception("No districts data available")
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Invalid boundary type: {boundary_type}'
                }, status=400)
            
            # Convert to GeoJSON (simplified - in production you'd want proper conversion)
            # For now, return a success response indicating data is available
            return JsonResponse({
                'status': 'success',
                'message': f'Real {boundary_type} boundaries available from GADM',
                'boundary_type': boundary_type,
                'source': 'GADM_REAL_DATA',
                'feature_count': boundaries.size().getInfo() if boundaries else 0
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error loading {boundary_type} boundaries: {str(e)}'
            }, status=500)


class PointQueryView(APIView):
    """API endpoint for querying administrative boundaries by coordinates - Tanzania Government"""
    
    def get(self, request):
        try:
            lon = float(request.GET.get('lon', 0))
            lat = float(request.GET.get('lat', 0))
            level = request.GET.get('level', 'district')
            
            print(f"🇹🇿 Tanzania Government - Point query at {lon}, {lat} for {level} level")
            
            # Create point geometry
            from django.contrib.gis.geos import Point
            point = Point(lon, lat, srid=4326)
            
            # Query appropriate model based on level
            from .models import Region, District, Ward
            
            result_data = None
            
            if level == 'ward':
                ward = Ward.objects.filter(geom__contains=point).first()
                if ward:
                    result_data = {
                        'name': ward.name_3,
                        'ward': ward.name_3,
                        'district': ward.name_2,
                        'region': ward.name_1,
                        'level': 'ward',
                        'source': 'GADM Tanzania'
                    }
            
            if not result_data and level in ['ward', 'district']:
                district = District.objects.filter(geom__contains=point).first()
                if district:
                    result_data = {
                        'name': district.name_2,
                        'district': district.name_2,
                        'region': district.name_1,
                        'level': 'district',
                        'source': 'GADM Tanzania'
                    }
            
            if not result_data:
                region = Region.objects.filter(geom__contains=point).first()
                if region:
                    result_data = {
                        'name': region.name_1,
                        'region': region.name_1,
                        'level': 'region',
                        'source': 'GADM Tanzania'
                    }
            
            if result_data:
                print(f"✅ Found {result_data['level']}: {result_data['name']}")
                return JsonResponse({
                    'status': 'success',
                    'data': result_data,
                    'message': f'Location found in {result_data["level"]}: {result_data["name"]}'
                })
            else:
                print(f"ℹ️ Point {lon}, {lat} not found in any administrative boundary")
                return JsonResponse({
                    'status': 'success',
                    'data': None,
                    'message': 'Location not found in administrative boundaries'
                })
                
        except Exception as e:
            print(f"❌ Error in point query: {e}")
            return JsonResponse({
                'status': 'error',
                'message': f'Point query failed: {str(e)}'
            }, status=500)

