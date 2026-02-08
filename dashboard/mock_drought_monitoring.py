"""
Mock Drought Monitoring for Demo Purposes
This module provides mock data when Google Earth Engine is not available
"""

from datetime import datetime, timedelta
import json
import random

class MockDroughtMonitor:
    """Mock drought monitoring system for demo purposes"""
    
    def __init__(self):
        self.regions = [
            'Arusha', 'Dar es Salaam', 'Dodoma', 'Geita', 'Iringa', 'Kagera',
            'Katavi', 'Kigoma', 'Kilimanjaro', 'Lindi', 'Manyara', 'Mara',
            'Mbeya', 'Morogoro', 'Mtwara', 'Mwanza', 'Njombe', 'Pemba North',
            'Pemba South', 'Pwani', 'Rukwa', 'Ruvuma', 'Shinyanga', 'Simiyu',
            'Singida', 'Tabora', 'Tanga', 'Zanzibar Central/South', 'Zanzibar North',
            'Zanzibar Urban/West'
        ]
    
    def generate_mock_drought_analysis(self, start_date, end_date, geometry=None):
        """Generate mock drought analysis data"""
        try:
            # Generate mock alerts
            alerts = []
            for i, region in enumerate(self.regions[:10]):  # Top 10 regions
                # Generate random but realistic severity data
                severe_percentage = random.uniform(5, 85)
                
                if severe_percentage > 70:
                    alert_level = "CRITICAL"
                elif severe_percentage > 50:
                    alert_level = "HIGH"
                elif severe_percentage > 25:
                    alert_level = "MODERATE"
                else:
                    alert_level = "LOW"
                
                alerts.append({
                    'region': region,
                    'alert_level': alert_level,
                    'severe_percentage': round(severe_percentage, 1),
                    'total_area': random.randint(5000, 50000),
                    'drought_distribution': {
                        '1': random.randint(100, 1000),   # No drought
                        '2': random.randint(50, 500),     # Mild
                        '3': random.randint(20, 300),     # Moderate
                        '4': random.randint(10, 200),     # Severe
                        '5': random.randint(5, 100)       # Extreme
                    }
                })
            
            # Sort by severity
            alerts.sort(key=lambda x: x['severe_percentage'], reverse=True)
            
            # Generate mock tile URLs (these would be real in production)
            composite_tile_url = self.generate_mock_tile_url('composite')
            drought_classes_tile_url = self.generate_mock_tile_url('classes')
            
            # Generate mock indices
            indices_calculated = ['vci', 'tci', 'spi', 'smi', 'evi']
            
            return {
                'composite_index_url': composite_tile_url,
                'drought_classes_url': drought_classes_tile_url,
                'alerts': alerts,
                'indices_calculated': indices_calculated,
                'status': 'success',
                'data_source': 'mock',
                'message': 'Mock data for demonstration purposes'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error generating mock data: {str(e)}'
            }
    
    def generate_mock_tile_url(self, layer_type):
        """Generate mock tile URL for demonstration"""
        # In a real implementation, this would be a Google Earth Engine tile URL
        # For demo, we'll use a placeholder that shows the concept
        base_url = "https://earthengine.googleapis.com/v1alpha/projects/earthengine-legacy/maps"
        mock_map_id = f"mock-{layer_type}-{random.randint(1000, 9999)}"
        mock_token = f"token-{random.randint(10000, 99999)}"
        
        return f"{base_url}/{mock_map_id}/tiles/{{z}}/{{x}}/{{y}}?token={mock_token}"

def get_enhanced_drought_analysis_mock(start_date, end_date, geometry=None):
    """Mock version of enhanced drought analysis"""
    try:
        monitor = MockDroughtMonitor()
        return monitor.generate_mock_drought_analysis(start_date, end_date, geometry)
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Mock drought analysis failed: {str(e)}'
        }

def get_mock_agricultural_intelligence(crop_type='maize', analysis_type='yield_prediction', 
                                     start_date=None, end_date=None, geometry=None):
    """Generate mock agricultural intelligence data"""
    try:
        # Mock NDVI value
        avg_ndvi = random.uniform(0.2, 0.8)
        
        # Mock yield estimation
        if avg_ndvi > 0.7:
            yield_category = 'High'
            estimated_yield = random.uniform(4.0, 5.0)
        elif avg_ndvi > 0.5:
            yield_category = 'Medium'
            estimated_yield = random.uniform(3.0, 4.0)
        elif avg_ndvi > 0.3:
            yield_category = 'Low'
            estimated_yield = random.uniform(2.0, 3.0)
        else:
            yield_category = 'Poor'
            estimated_yield = random.uniform(1.0, 2.0)
        
        # Mock tile URLs
        ndvi_tile_url = f"https://earthengine.googleapis.com/mock/ndvi/tiles/{{z}}/{{x}}/{{y}}?token=mock-{random.randint(1000, 9999)}"
        precip_tile_url = f"https://earthengine.googleapis.com/mock/precipitation/tiles/{{z}}/{{x}}/{{y}}?token=mock-{random.randint(1000, 9999)}"
        
        recommendations = get_mock_agricultural_recommendations(avg_ndvi, yield_category)
        
        return {
            'status': 'success',
            'data': {
                'crop_type': crop_type,
                'analysis_period': {
                    'start_date': start_date or (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                    'end_date': end_date or datetime.now().strftime('%Y-%m-%d')
                },
                'vegetation_health': {
                    'average_ndvi': round(avg_ndvi, 3),
                    'tile_url': ndvi_tile_url
                },
                'precipitation_analysis': {
                    'tile_url': precip_tile_url
                },
                'yield_prediction': {
                    'category': yield_category,
                    'estimated_yield_tons_per_hectare': round(estimated_yield, 1),
                    'confidence': 'medium'
                },
                'recommendations': recommendations
            },
            'data_source': 'mock',
            'message': 'Mock agricultural data for demonstration'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Mock agricultural analysis failed: {str(e)}'
        }

def get_mock_agricultural_recommendations(ndvi, yield_category):
    """Generate mock agricultural recommendations"""
    recommendations = []
    
    if yield_category == 'Poor':
        recommendations = [
            "Consider soil testing and nutrient supplementation",
            "Evaluate irrigation needs - crops may be water stressed", 
            "Check for pest and disease issues",
            "Consider drought-resistant crop varieties for next season"
        ]
    elif yield_category == 'Low':
        recommendations = [
            "Monitor soil moisture levels closely",
            "Consider supplemental irrigation if available",
            "Apply appropriate fertilizers based on crop growth stage"
        ]
    elif yield_category == 'Medium':
        recommendations = [
            "Maintain current management practices",
            "Monitor for optimal harvest timing",
            "Plan for proper post-harvest handling"
        ]
    else:  # High
        recommendations = [
            "Excellent crop conditions - maintain current practices",
            "Prepare for good harvest - ensure adequate storage",
            "Consider expanding similar practices to other areas"
        ]
    
    # Add general recommendations
    recommendations.extend([
        "Continue monitoring weather forecasts",
        "Keep records of management practices for future reference",
        "Consider crop insurance options"
    ])
    
    return recommendations
