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
