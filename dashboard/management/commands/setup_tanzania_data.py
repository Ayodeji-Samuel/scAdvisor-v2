"""
Django management command to setup Tanzania GIS data and generate administrative boundaries
"""

from django.core.management.base import BaseCommand
from django.conf import settings
import os
import requests
import zipfile
import tempfile
from dashboard.raster_prediction import export_administrative_boundaries
from dashboard.geoserver_utils import GeoServerManager
from dashboard.gee_auth import authenticate_gee


class Command(BaseCommand):
    help = 'Setup Tanzania administrative boundaries and GIS data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--download-gadm',
            action='store_true',
            help='Download GADM administrative boundaries',
        )
        parser.add_argument(
            '--export-gee',
            action='store_true',
            help='Export boundaries from Google Earth Engine',
        )
        parser.add_argument(
            '--setup-geoserver',
            action='store_true',
            help='Setup GeoServer with administrative layers',
        )
        parser.add_argument(
            '--data-dir',
            type=str,
            default='data/shapefiles',
            help='Directory to store shapefile data',
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Setting up Tanzania GIS data...')
        )

        data_dir = options['data_dir']
        os.makedirs(data_dir, exist_ok=True)

        if options['download_gadm']:
            self.download_gadm_data(data_dir)

        if options['export_gee']:
            self.export_gee_boundaries()

        if options['setup_geoserver']:
            self.setup_geoserver_layers(data_dir)

        self.stdout.write(
            self.style.SUCCESS('Tanzania GIS data setup completed!')
        )

    def download_gadm_data(self, data_dir):
        """Download Tanzania administrative boundaries from GADM"""
        self.stdout.write('Downloading GADM data for Tanzania...')
        
        gadm_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_TZA_shp.zip"
        
        try:
            # Download file
            response = requests.get(gadm_url, stream=True)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            
            # Extract to data directory
            gadm_dir = os.path.join(data_dir, 'gadm')
            os.makedirs(gadm_dir, exist_ok=True)
            
            with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
                zip_ref.extractall(gadm_dir)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            self.stdout.write(
                self.style.SUCCESS(f'GADM data downloaded to {gadm_dir}')
            )
            
            # List available files
            for file in os.listdir(gadm_dir):
                if file.endswith('.shp'):
                    self.stdout.write(f'  - {file}')
                    
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error downloading GADM data: {str(e)}')
            )

    def export_gee_boundaries(self):
        """Export administrative boundaries from Google Earth Engine"""
        self.stdout.write('Exporting boundaries from Google Earth Engine...')
        
        try:
            # Authenticate with Google Earth Engine
            authenticate_gee()
            
            # Export administrative boundaries
            tasks = export_administrative_boundaries()
            
            if tasks:
                self.stdout.write(
                    self.style.SUCCESS('Export tasks started:')
                )
                for name, task in tasks.items():
                    self.stdout.write(f'  - {name}: {task.id}')
                    
                self.stdout.write(
                    'Note: Check your Google Drive for exported shapefiles'
                )
            else:
                self.stdout.write(
                    self.style.WARNING('No export tasks were created')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error exporting from GEE: {str(e)}')
            )

    def setup_geoserver_layers(self, data_dir):
        """Setup GeoServer with administrative layers"""
        self.stdout.write('Setting up GeoServer layers...')
        
        try:
            geoserver = GeoServerManager()
            
            # Create workspace
            if geoserver.create_workspace():
                self.stdout.write('GeoServer workspace created')
            
            # Look for shapefiles in data directory
            gadm_dir = os.path.join(data_dir, 'gadm')
            
            # Define shapefile mappings
            shapefiles = {
                'tanzania_regions': ('gadm41_TZA_1.shp', 'Tanzania Regions'),
                'tanzania_districts': ('gadm41_TZA_2.shp', 'Tanzania Districts'),
                'tanzania_wards': ('gadm41_TZA_3.shp', 'Tanzania Wards'),
            }
            
            for layer_name, (shapefile, title) in shapefiles.items():
                shapefile_path = os.path.join(gadm_dir, shapefile)
                
                if os.path.exists(shapefile_path):
                    if geoserver.upload_shapefile(shapefile_path, layer_name, title):
                        self.stdout.write(f'  ✓ {layer_name} uploaded successfully')
                    else:
                        self.stdout.write(f'  ✗ Failed to upload {layer_name}')
                else:
                    self.stdout.write(f'  - {shapefile} not found, skipping {layer_name}')
            
            self.stdout.write(
                self.style.SUCCESS('GeoServer setup completed')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error setting up GeoServer: {str(e)}')
            )

    def download_natural_earth_data(self, data_dir):
        """Download Natural Earth administrative boundaries"""
        self.stdout.write('Downloading Natural Earth data...')
        
        ne_url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip"
        
        try:
            response = requests.get(ne_url, stream=True)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            
            ne_dir = os.path.join(data_dir, 'natural_earth')
            os.makedirs(ne_dir, exist_ok=True)
            
            with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
                zip_ref.extractall(ne_dir)
            
            os.unlink(tmp_file_path)
            
            self.stdout.write(
                self.style.SUCCESS(f'Natural Earth data downloaded to {ne_dir}')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error downloading Natural Earth data: {str(e)}')
            )
