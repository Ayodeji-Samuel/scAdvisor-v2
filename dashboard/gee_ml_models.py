import ee
from .gee_data_processing import get_tanzania_boundary, get_sentinel1_flood_data, get_optical_drought_data
from datetime import datetime, timedelta

def train_flood_classifier(geometry):
    """Trains a flood classifier using historical data.
    For a more realistic scenario, this would involve actual flood event data.
    Here, we simulate training data based on a simple water detection approach.
    """
    try:
        # Use a historical Sentinel-1 collection and get a median composite for training data generation
        # This approach is more robust than using a specific image ID that might not exist
        collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(geometry) \
            .filterDate('2014-10-03', '2025-07-18') \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
            .select(['VV', 'VH'])
        
        # Get median composite
        image = collection.median()
        
        # Simple water detection using VV polarization (threshold can be refined)
        water_mask = image.select("VV").lt(-18) # Adjust threshold as needed

        # Create training data: sample points from water and non-water areas
        # This is a highly simplified approach for generating training data.
        # In a real application, you would use actual ground truth data (e.g., digitized flood extents).
        water_points = image.select(['VV', 'VH']).sample(
            region=geometry,
            scale=30,
            numPixels=500,
            seed=0,
            geometries=True
        ).map(lambda f: f.set("class", 1)) # Class 1 for water

        land_points = image.select(['VV', 'VH']).sample(
            region=geometry,
            scale=30,
            numPixels=500,
            seed=1,
            geometries=True
        ).map(lambda f: f.set("class", 0)) # Class 0 for land

        training_points = water_points.merge(land_points)

        # Define the classifier (Random Forest is a good choice for remote sensing)
        classifier = ee.Classifier.smileRandomForest(10).train(
            features=training_points,
            classProperty='class',
            inputProperties=['VV', 'VH']
        )
        return classifier
    except Exception as e:
        print(f"Error training flood classifier: {e}")
        # Return a dummy classifier or handle error appropriately
        return ee.Classifier.smileRandomForest(1)

def train_drought_classifier(geometry):
    """Trains a drought classifier using historical data.
    For a more realistic scenario, this would involve actual drought event data
    and various drought indices.
    """
    try:
        # Use a historical Sentinel-2 collection and get a median composite for training data generation
        # This approach is more robust than using a specific image ID that might not exist
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(geometry) \
            .filterDate('2017-03-28', '2025-07-18') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
        # Get median composite and calculate NDVI
        image = collection.median()
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

        # Simple drought classification based on NDVI (thresholds can be refined)
        # 0: Healthy vegetation, 1: Mild drought, 2: Moderate drought, 3: Severe drought
        healthy_points = ndvi.sample(
            region=geometry,
            scale=30,
            numPixels=250,
            seed=0,
            geometries=True
        ).filter(ee.Filter.gt('NDVI', 0.5)).map(lambda f: f.set("class", 0))

        mild_drought_points = ndvi.sample(
            region=geometry,
            scale=30,
            numPixels=250,
            seed=1,
            geometries=True
        ).filter(ee.Filter.And(ee.Filter.gt('NDVI', 0.3), ee.Filter.lte('NDVI', 0.5))).map(lambda f: f.set("class", 1))

        moderate_drought_points = ndvi.sample(
            region=geometry,
            scale=30,
            numPixels=250,
            seed=2,
            geometries=True
        ).filter(ee.Filter.And(ee.Filter.gt('NDVI', 0.1), ee.Filter.lte('NDVI', 0.3))).map(lambda f: f.set("class", 2))

        severe_drought_points = ndvi.sample(
            region=geometry,
            scale=30,
            numPixels=250,
            seed=3,
            geometries=True
        ).filter(ee.Filter.lte('NDVI', 0.1)).map(lambda f: f.set("class", 3))

        training_points = healthy_points.merge(mild_drought_points).merge(moderate_drought_points).merge(severe_drought_points)

        # Define the classifier (Random Forest)
        classifier = ee.Classifier.smileRandomForest(10).train(
            features=training_points,
            classProperty='class',
            inputProperties=['NDVI']
        )
        return classifier
    except Exception as e:
        print(f"Error training drought classifier: {e}")
        # Return a dummy classifier or handle error appropriately
        return ee.Classifier.smileRandomForest(1)

def predict_flood(classifier, target_date, geometry):
    """Makes a flood prediction for a given date.
    Args:
        classifier (ee.Classifier): Trained GEE classifier.
        target_date (datetime.date): Date for prediction.
        geometry (ee.Geometry): Region of interest.
    Returns:
        ee.Image: Predicted flood map.
    """
    try:
        # For forecasting, this would ideally use forecasted data. For now, it uses historical data up to the target_date.
        # This is a significant simplification for the purpose of this task.
        start_date_str = (target_date - timedelta(days=7)).strftime("%Y-%m-%d") # Look back 7 days for data
        end_date_str = target_date.strftime("%Y-%m-%d")
        image = get_sentinel1_flood_data(start_date_str, end_date_str, geometry)
        if image.bandNames().getInfo(): # Check if image is not empty
            return image.classify(classifier)
        else:
            return ee.Image().byte()
    except Exception as e:
        print(f"Error predicting flood: {e}")
        return ee.Image().byte()

def predict_drought(classifier, target_date, geometry):
    """Makes a drought prediction for a given date.
    Args:
        classifier (ee.Classifier): Trained GEE classifier.
        target_date (datetime.date): Date for prediction.
        geometry (ee.Geometry): Region of interest.
    Returns:
        ee.Image: Predicted drought map.
    """
    try:
        # For forecasting, this would ideally use forecasted data. For now, it uses historical data up to the target_date.
        # This is a significant simplification for the purpose of this task.
        start_date_str = (target_date - timedelta(days=30)).strftime("%Y-%m-%d") # Look back 30 days for data
        end_date_str = target_date.strftime("%Y-%m-%d")
        image = get_optical_drought_data(start_date_str, end_date_str, geometry)
        if image.bandNames().getInfo(): # Check if image is not empty
            return image.classify(classifier)
        else:
            return ee.Image().byte()
    except Exception as e:
        print(f"Error predicting drought: {e}")
        return ee.Image().byte()

# Example usage (for testing purposes)
if __name__ == '__main__':
    from gee_auth import authenticate_gee
    authenticate_gee()

    tanzania_boundary = get_tanzania_boundary().geometry()

    # Train classifiers
    flood_classifier = train_flood_classifier(tanzania_boundary)
    drought_classifier = train_drought_classifier(tanzania_boundary)

    # Make predictions for today and future dates
    today = datetime.now().date()
    forecast_dates = [
        today,
        today + timedelta(days=7),
        today + timedelta(days=14),
        today + timedelta(days=21),
    ]

    for date in forecast_dates:
        print(f"Predicting for: {date.strftime('%Y-%m-%d')}")
        flood_prediction = predict_flood(flood_classifier, date, tanzania_boundary)
        drought_prediction = predict_drought(drought_classifier, date, tanzania_boundary)

        print("Flood prediction image:", flood_prediction.getInfo())
        print("Drought prediction image:", drought_prediction.getInfo())

