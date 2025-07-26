#!/bin/bash
# Setup script for downloading large data files for scAdvisor

echo "Setting up scAdvisor large data files..."

# Create data directories if they don't exist
mkdir -p data/shapefiles

# Download Tanzania OSM data (700MB) - optional for advanced processing
echo "Downloading Tanzania OSM data (700MB)..."
if [ ! -f "data/shapefiles/tanzania_osm.pbf" ]; then
    curl -L -o data/shapefiles/tanzania_osm.pbf "https://download.geofabrik.de/africa/tanzania-latest.osm.pbf"
    echo "✓ Tanzania OSM data downloaded"
else
    echo "✓ Tanzania OSM data already exists"
fi

# Download Natural Earth countries shapefile
echo "Downloading Natural Earth countries data..."
if [ ! -f "data/shapefiles/natural_earth_countries.zip" ]; then
    curl -L -o data/shapefiles/natural_earth_countries.zip "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/cultural/ne_50m_admin_0_countries.zip"
    echo "✓ Natural Earth countries data downloaded"
else
    echo "✓ Natural Earth countries data already exists"
fi

echo ""
echo "Data setup complete!"
echo ""
echo "Note: The core functionality uses GADM administrative boundaries"
echo "which are already included in the repository."
echo ""
echo "Large files downloaded:"
echo "- data/shapefiles/tanzania_osm.pbf (optional)"
echo "- data/shapefiles/natural_earth_countries.zip (optional)"
