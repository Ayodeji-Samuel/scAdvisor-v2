import ee
import os
from django.conf import settings

def get_tanzania_boundary():
    """Returns the boundary of Tanzania using local GADM shapefiles or fallback to GEE datasets."""
    try:
        # First try to use uploaded GADM assets if available
        # You can upload your local shapefiles to GEE as assets
        # For now, we'll use the GADM dataset available in GEE
        return ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level0').filter(
            ee.Filter.eq('ADM0_NAME', 'United Republic of Tanzania')
        )
    except:
        # Fallback to alternative datasets
        try:
            return ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(
                ee.Filter.eq('country_na', 'Tanzania')
            )
        except:
            # Final fallback
            return ee.FeatureCollection('projects/google/charts/features/countries').filter(
                ee.Filter.eq('country', 'TZ')
            )

def get_tanzania_regions():
    """Returns Tanzania regions (administrative level 1) using GADM data."""
    try:
        # Use GADM level 1 for regions
        return ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level1').filter(
            ee.Filter.eq('ADM0_NAME', 'United Republic of Tanzania')
        )
    except:
        # Fallback: create synthetic regions from country boundary
        country = get_tanzania_boundary()
        return country.geometry().coveringGrid(ee.Projection('EPSG:4326'), 100000)

def get_tanzania_districts():
    """Returns Tanzania districts (administrative level 2) using GADM data."""
    try:
        # Use GADM level 2 for districts
        return ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level2').filter(
            ee.Filter.eq('ADM0_NAME', 'United Republic of Tanzania')
        )
    except:
        # Fallback to regions
        return get_tanzania_regions()

def upload_local_shapefiles_to_gee():
    """
    Helper function to upload local GADM shapefiles to Google Earth Engine as assets.
    This should be run once to upload your local shapefiles to GEE.
    Note: This requires geemap and proper authentication.
    """
    try:
        import geemap
        
        # Paths to your local shapefiles
        base_path = os.path.join(settings.BASE_DIR, 'data', 'shapefiles', 'gadm')
        
        shapefiles = {
            'tanzania_country': os.path.join(base_path, 'gadm41_TZA_0.shp'),
            'tanzania_regions': os.path.join(base_path, 'gadm41_TZA_1.shp'),
            'tanzania_districts': os.path.join(base_path, 'gadm41_TZA_2.shp'),
            'tanzania_wards': os.path.join(base_path, 'gadm41_TZA_3.shp'),
        }
        
        # Upload each shapefile as a GEE asset
        for asset_name, shapefile_path in shapefiles.items():
            if os.path.exists(shapefile_path):
                try:
                    # This would upload to your GEE assets
                    # geemap.shp_to_ee(shapefile_path, asset_id=f'users/your_username/{asset_name}')
                    print(f"Shapefile {asset_name} ready for upload: {shapefile_path}")
                except Exception as e:
                    print(f"Error uploading {asset_name}: {e}")
            else:
                print(f"Shapefile not found: {shapefile_path}")
                
    except ImportError:
        print("geemap not available for shapefile upload")
    except Exception as e:
        print(f"Error in shapefile upload: {e}")

def get_sentinel1_flood_data(start_date, end_date, geometry):
    """Fetches and pre-processes Sentinel-1 data for flood detection.
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        geometry (ee.Geometry): Region of interest.
    Returns:
        ee.Image: Pre-processed Sentinel-1 image collection.
    """
    collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
        .filter(ee.Filter.eq('resolution_meters', 10)) \
        .filter(ee.Filter.eq('SPATIAL_REF_ACCURACY', 'HIGH')) \
        .select(['VV', 'VH'])

    # Apply speckle filter (e.g., Refined Lee)
    def apply_speckle_filter(image):
        return image.focal_median(3, 'square', 'pixels')

    filtered_collection = collection.map(apply_speckle_filter)

    # Calculate flood proxy (e.g., simple thresholding or change detection)
    # This is a placeholder. A proper flood detection algorithm would be more complex.
    # For example, using a pre-event image and post-event image for change detection.
    # For now, let's just return the median of the collection.
    return filtered_collection.median()

def get_optical_drought_data(start_date, end_date, geometry):
    """Fetches and pre-processes Sentinel-2 data for drought detection.
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        geometry (ee.Geometry): Region of interest.
    Returns:
        ee.Image: Pre-processed Sentinel-2 image with NDVI.
    """
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    # Apply cloud mask and add NDVI
    def mask_s2_clouds(image):
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask).divide(10000)

    processed_collection = collection.map(mask_s2_clouds).map(add_ndvi)

    # For drought, we often need a time series or a composite. Let's return the median NDVI.
    return processed_collection.select('NDVI').median()

# Example usage (for testing purposes, not to be run directly in Django views)
if __name__ == '__main__':
    from gee_auth import authenticate_gee
    authenticate_gee()

    tanzania = get_tanzania_boundary()
    print(f"Tanzania boundary type: {tanzania.first().geometry().type().getInfo()}")

    # Example flood data for a specific date range
    flood_image = get_sentinel1_flood_data('2024-01-01', '2024-01-31', tanzania.geometry())
    print(f"Flood image bands: {flood_image.bandNames().getInfo()}")

    # Example drought data for a specific date range
    drought_image = get_optical_drought_data('2024-01-01', '2024-01-31', tanzania.geometry())
    print(f"Drought image bands: {drought_image.bandNames().getInfo()}")


