import ee
import os

# Path to the service account key file
SERVICE_ACCOUNT_KEY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ee-my-makinde-2b6858cddb01.json')
SERVICE_ACCOUNT_EMAIL = 'smart-climate@ee-my-makinde.iam.gserviceaccount.com'

def authenticate_gee():
    try:
        # Authenticate using the service account
        credentials = ee.ServiceAccountCredentials(
            SERVICE_ACCOUNT_EMAIL,
            SERVICE_ACCOUNT_KEY_PATH
        )
        ee.Initialize(credentials)
        print("Google Earth Engine initialized successfully using service account.")
    except Exception as e:
        print(f"Error initializing GEE with service account: {e}")
        print("Please ensure the service account key file is correct and has necessary permissions.")

if __name__ == '__main__':
    authenticate_gee()


