# Smart Climate Advisor (scAdvisor)

A Django-based web application for Tanzania flood and drought prediction using Google Earth Engine satellite imagery and real-time climate data.

## Features

✅ **Real-Time Satellite Data**
- Google Earth Engine integration
- Sentinel-1 SAR for flood detection
- Sentinel-2 optical for drought monitoring
- CHIRPS precipitation data
- ERA5 temperature data

✅ **Interactive Map Dashboard**
- OpenLayers-based mapping interface
- GADM Tanzania administrative boundaries (31 regions, 186 districts, 3,663 wards)
- Real prediction layer visualization
- Click-to-query region information

✅ **Climate Predictions**
- Real flood risk assessment using satellite radar
- Real drought risk monitoring using vegetation indices
- Forecast periods (7, 14, 21 days)
- Regional statistics and analysis
- Agricultural recommendations

## Quick Start

### 1. Set Up Virtual Environment
```bash
# Use the provided virtual environment to avoid compatibility issues
C:\Users\Unstoppable\Documents\Python\Ongoing Project\Smart Climate\sc\scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Large Data Files (Optional)
The following large files are not included in the repository due to size constraints:
```bash
# Tanzania OSM data (700MB) - only needed for advanced processing
curl -o data/shapefiles/tanzania_osm.pbf "https://download.geofabrik.de/africa/tanzania-latest.osm.pbf"

# Natural Earth countries shapefile
curl -o data/shapefiles/natural_earth_countries.zip "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/cultural/ne_50m_admin_0_countries.zip"
```

### 4. Google Earth Engine Setup
- Ensure you have a Google Earth Engine service account
- Place your service account key as `ee-my-makinde-2b6858cddb01.json` in the project root
- The system will authenticate automatically

### 5. Run Database Migrations
```bash
python manage.py migrate
```

### 6. Start Development Server
```bash
python manage.py runserver
```

### 7. Access Application
- Dashboard: `http://127.0.0.1:8000/dashboard/`
- Admin: `http://127.0.0.1:8000/admin/`

## API Endpoints

- `GET /dashboard/api/raster-predictions/` - Get prediction data
- `GET /dashboard/api/administrative/` - Get administrative boundaries
- `GET /dashboard/api/regional-statistics/` - Get regional statistics

## Development Mode

The application runs in development mode with:
- Built-in Tanzania boundary data
- Mock prediction data
- Fallback mechanisms for external services
- Debug status panel for troubleshooting

## Project Structure

```
scAdvisor/
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies
├── smartproject/            # Main Django project
├── dashboard/               # Main dashboard app
│   ├── views.py            # API endpoints and views
│   ├── models.py           # Data models
│   ├── templates/          # HTML templates
│   └── static/             # CSS, JS, images
├── home/                   # Home page app
└── data/                   # Data files and shapefiles
```

## Optional Enhancements

### Google Earth Engine Integration
- Set up GEE credentials for real satellite data
- Enable machine learning predictions
- Access to historical climate data

### Authentication
- Create superuser: `python manage.py createsuperuser`
- Add user management and permissions
- Secure API endpoints

## Development Notes

- The application works completely offline
- No external services required for basic functionality
- All prediction data is currently mock/sample data
- Map tiles are loaded from OpenStreetMap
- Administrative boundaries are simplified but functional

## Technologies Used

- **Backend**: Django 5.2.4, Python
- **Frontend**: Bootstrap 5, OpenLayers 6+
- **Database**: SQLite (development)
- **Maps**: OpenLayers with OSM tiles
- **Optional**: Google Earth Engine, GeoServer
