@echo off
REM Setup script for downloading large data files for scAdvisor (Windows)

echo Setting up scAdvisor large data files...

REM Create data directories if they don't exist
if not exist "data\shapefiles" mkdir data\shapefiles

REM Download Tanzania OSM data (700MB) - optional for advanced processing
echo Downloading Tanzania OSM data (700MB)...
if not exist "data\shapefiles\tanzania_osm.pbf" (
    curl -L -o data\shapefiles\tanzania_osm.pbf "https://download.geofabrik.de/africa/tanzania-latest.osm.pbf"
    echo ✓ Tanzania OSM data downloaded
) else (
    echo ✓ Tanzania OSM data already exists
)

REM Download Natural Earth countries shapefile
echo Downloading Natural Earth countries data...
if not exist "data\shapefiles\natural_earth_countries.zip" (
    curl -L -o data\shapefiles\natural_earth_countries.zip "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/cultural/ne_50m_admin_0_countries.zip"
    echo ✓ Natural Earth countries data downloaded
) else (
    echo ✓ Natural Earth countries data already exists
)

echo.
echo Data setup complete!
echo.
echo Note: The core functionality uses GADM administrative boundaries
echo which are already included in the repository.
echo.
echo Large files downloaded:
echo - data\shapefiles\tanzania_osm.pbf (optional)
echo - data\shapefiles\natural_earth_countries.zip (optional)

pause
