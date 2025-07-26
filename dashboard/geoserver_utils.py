"""
GeoServer integration utilities for Tanzania administrative boundaries
"""
import requests
import json
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# GeoServer configuration
GEOSERVER_BASE_URL = getattr(settings, 'GEOSERVER_BASE_URL', 'http://localhost:8080/geoserver')
GEOSERVER_USERNAME = getattr(settings, 'GEOSERVER_USERNAME', 'admin')
GEOSERVER_PASSWORD = getattr(settings, 'GEOSERVER_PASSWORD', 'geoserver')
GEOSERVER_WORKSPACE = getattr(settings, 'GEOSERVER_WORKSPACE', 'tanzania')

class GeoServerManager:
    """Manages GeoServer operations for Tanzania administrative data"""
    
    def __init__(self):
        self.base_url = GEOSERVER_BASE_URL
        self.auth = (GEOSERVER_USERNAME, GEOSERVER_PASSWORD)
        self.workspace = GEOSERVER_WORKSPACE
        
    def create_workspace(self):
        """Create workspace for Tanzania data"""
        url = f"{self.base_url}/rest/workspaces"
        data = {
            "workspace": {
                "name": self.workspace,
                "dataStores": []
            }
        }
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(url, json=data, headers=headers, auth=self.auth)
            if response.status_code in [201, 409]:  # 409 means already exists
                logger.info(f"Workspace '{self.workspace}' created or already exists")
                return True
            else:
                logger.error(f"Failed to create workspace: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error creating workspace: {str(e)}")
            return False
    
    def upload_shapefile(self, shapefile_path, layer_name, title=None):
        """Upload shapefile to GeoServer"""
        if not title:
            title = layer_name.replace('_', ' ').title()
            
        # Create datastore
        datastore_url = f"{self.base_url}/rest/workspaces/{self.workspace}/datastores"
        datastore_data = {
            "dataStore": {
                "name": layer_name,
                "type": "Shapefile",
                "enabled": True,
                "connectionParameters": {
                    "url": f"file:{shapefile_path}"
                }
            }
        }
        
        try:
            response = requests.post(
                datastore_url, 
                json=datastore_data, 
                headers={'Content-Type': 'application/json'}, 
                auth=self.auth
            )
            
            if response.status_code in [201, 409]:
                logger.info(f"Datastore '{layer_name}' created successfully")
                
                # Create featuretype (layer)
                featuretype_url = f"{datastore_url}/{layer_name}/featuretypes"
                featuretype_data = {
                    "featureType": {
                        "name": layer_name,
                        "title": title,
                        "enabled": True,
                        "srs": "EPSG:4326"
                    }
                }
                
                ft_response = requests.post(
                    featuretype_url,
                    json=featuretype_data,
                    headers={'Content-Type': 'application/json'},
                    auth=self.auth
                )
                
                if ft_response.status_code in [201, 409]:
                    logger.info(f"Layer '{layer_name}' created successfully")
                    return True
                else:
                    logger.error(f"Failed to create layer: {ft_response.status_code}")
                    return False
            else:
                logger.error(f"Failed to create datastore: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error uploading shapefile: {str(e)}")
            return False
    
    def get_wms_url(self, layer_name):
        """Get WMS URL for a layer"""
        return f"{self.base_url}/{self.workspace}/wms"
    
    def get_wfs_url(self, layer_name):
        """Get WFS URL for a layer"""
        return f"{self.base_url}/{self.workspace}/wfs"
    
    def get_layer_capabilities(self, layer_name):
        """Get layer capabilities and metadata"""
        url = f"{self.base_url}/rest/workspaces/{self.workspace}/layers/{layer_name}"
        try:
            response = requests.get(url, auth=self.auth, headers={'Accept': 'application/json'})
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error getting layer capabilities: {str(e)}")
            return None
    
    def get_feature_info(self, layer_name, bbox, width=512, height=512, x=256, y=256):
        """Get feature info for a point"""
        wms_url = self.get_wms_url(layer_name)
        params = {
            'SERVICE': 'WMS',
            'VERSION': '1.1.1',
            'REQUEST': 'GetFeatureInfo',
            'LAYERS': f"{self.workspace}:{layer_name}",
            'QUERY_LAYERS': f"{self.workspace}:{layer_name}",
            'INFO_FORMAT': 'application/json',
            'BBOX': bbox,
            'WIDTH': width,
            'HEIGHT': height,
            'X': x,
            'Y': y,
            'SRS': 'EPSG:4326'
        }
        
        try:
            response = requests.get(wms_url, params=params, auth=self.auth)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error getting feature info: {str(e)}")
            return None

def get_tanzania_administrative_layers():
    """Get available Tanzania administrative layers from GeoServer"""
    geoserver = GeoServerManager()
    
    # Define expected layers
    expected_layers = [
        'tanzania_regions',
        'tanzania_districts',
        'tanzania_wards',
        'tanzania_boundaries'
    ]
    
    available_layers = []
    for layer in expected_layers:
        capabilities = geoserver.get_layer_capabilities(layer)
        if capabilities:
            available_layers.append({
                'name': layer,
                'title': layer.replace('_', ' ').title(),
                'wms_url': geoserver.get_wms_url(layer),
                'wfs_url': geoserver.get_wfs_url(layer)
            })
    
    return available_layers

def setup_tanzania_layers():
    """Setup Tanzania administrative layers in GeoServer"""
    geoserver = GeoServerManager()
    
    # Create workspace
    geoserver.create_workspace()
    
    # This would be called when shapefiles are available
    # geoserver.upload_shapefile('/path/to/tanzania_regions.shp', 'tanzania_regions')
    # geoserver.upload_shapefile('/path/to/tanzania_districts.shp', 'tanzania_districts')
    
    return True
