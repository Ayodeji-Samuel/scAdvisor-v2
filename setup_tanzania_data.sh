#!/bin/bash

# Tanzania GIS Data Setup Script
# This script helps download and prepare Tanzania administrative boundaries

echo "Tanzania Flood & Drought Dashboard - GIS Data Setup"
echo "=================================================="

# Create directories
mkdir -p data/shapefiles
mkdir -p data/geoserver
cd data/shapefiles

echo "Downloading Tanzania administrative boundaries..."

# Download Tanzania boundaries from various sources
# Option 1: GADM (Global Administrative Areas)
echo "Downloading from GADM..."
curl -o tanzania_gadm.zip "https://biogeo.ucdavis.edu/data/gadm3.6/shp/gadm36_TZA_shp.zip"

# Option 2: Natural Earth (for country boundaries)
echo "Downloading from Natural Earth..."
curl -o natural_earth_countries.zip "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip"

# Option 3: OpenStreetMap (via geofabrik)
echo "Downloading from OpenStreetMap..."
curl -o tanzania_osm.pbf "https://download.geofabrik.de/africa/tanzania-latest.osm.pbf"

# Extract downloaded files
echo "Extracting files..."
if [ -f "tanzania_gadm.zip" ]; then
    unzip -o tanzania_gadm.zip -d gadm/
    echo "GADM files extracted to gadm/"
fi

if [ -f "natural_earth_countries.zip" ]; then
    unzip -o natural_earth_countries.zip -d natural_earth/
    echo "Natural Earth files extracted to natural_earth/"
fi

echo "Data download completed!"
echo ""
echo "Available Tanzania administrative files:"
echo "- GADM Level 0 (Country): gadm/gadm36_TZA_0.*"
echo "- GADM Level 1 (Regions): gadm/gadm36_TZA_1.*"
echo "- GADM Level 2 (Districts): gadm/gadm36_TZA_2.*"
echo "- GADM Level 3 (Wards): gadm/gadm36_TZA_3.*"
echo ""

# Create GeoServer workspace setup script
cat > ../geoserver/setup_geoserver.sh << 'EOF'
#!/bin/bash

# GeoServer Setup Script for Tanzania Data
echo "Setting up GeoServer for Tanzania data..."

GEOSERVER_URL="http://localhost:8080/geoserver"
GEOSERVER_USER="admin"
GEOSERVER_PASS="geoserver"
WORKSPACE="tanzania"

# Create workspace
echo "Creating workspace: $WORKSPACE"
curl -u $GEOSERVER_USER:$GEOSERVER_PASS -X POST \
  -H "Content-Type: application/json" \
  -d "{\"workspace\":{\"name\":\"$WORKSPACE\"}}" \
  $GEOSERVER_URL/rest/workspaces

# Upload shapefiles
echo "Uploading Tanzania regions shapefile..."
if [ -f "../shapefiles/gadm/gadm36_TZA_1.shp" ]; then
    zip -j tanzania_regions.zip ../shapefiles/gadm/gadm36_TZA_1.*
    
    curl -u $GEOSERVER_USER:$GEOSERVER_PASS -X PUT \
      -H "Content-Type: application/zip" \
      --data-binary @tanzania_regions.zip \
      $GEOSERVER_URL/rest/workspaces/$WORKSPACE/datastores/tanzania_regions/file.shp
fi

echo "Uploading Tanzania districts shapefile..."
if [ -f "../shapefiles/gadm/gadm36_TZA_2.shp" ]; then
    zip -j tanzania_districts.zip ../shapefiles/gadm/gadm36_TZA_2.*
    
    curl -u $GEOSERVER_USER:$GEOSERVER_PASS -X PUT \
      -H "Content-Type: application/zip" \
      --data-binary @tanzania_districts.zip \
      $GEOSERVER_URL/rest/workspaces/$WORKSPACE/datastores/tanzania_districts/file.shp
fi

echo "GeoServer setup completed!"
echo "Access your layers at: $GEOSERVER_URL/$WORKSPACE/wms"
EOF

chmod +x ../geoserver/setup_geoserver.sh

echo "GeoServer setup script created at: data/geoserver/setup_geoserver.sh"
echo ""
echo "Next steps:"
echo "1. Make sure GeoServer is running on localhost:8080"
echo "2. Run: bash data/geoserver/setup_geoserver.sh"
echo "3. Update your Django settings with GeoServer configuration"
echo ""
echo "Django settings example:"
echo "GEOSERVER_BASE_URL = 'http://localhost:8080/geoserver'"
echo "GEOSERVER_USERNAME = 'admin'"
echo "GEOSERVER_PASSWORD = 'geoserver'"
echo "GEOSERVER_WORKSPACE = 'tanzania'"
