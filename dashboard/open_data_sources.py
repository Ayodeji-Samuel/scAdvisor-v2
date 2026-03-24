"""
Open Data Sources for Tanzania Flood & Drought Prediction
==========================================================
All datasets here are freely available via Google Earth Engine.

Datasets integrated:
- JRC Global Surface Water v1.4          (permanent / seasonal water extent)
- SRTM DEM 30m                           (topographic wetness index for flood routing)
- GPM IMERG v6                           (near-real-time precipitation, globally)
- CHIRPS Daily                           (long climate record precipitation)
- ESA WorldCover 2021                    (land cover / impervious surfaces)
- MODIS Terra MOD13A2                    (NDVI / EVI vegetation indices)
- MODIS MOD11A1                          (land surface temperature)
- MODIS MOD16A2                          (evapotranspiration — drought indicator)
- ECMWF ERA5-Land                        (meteorological reanalysis)
- NASA FLDAS NOAH v001                   (multi-model soil moisture & runoff)
- NASA SMAP SPL3SMP_E v005              (soil moisture active/passive, 9 km)
- NASA FIRMS (MODIS/VIIRS)               (active fire — drought proxy)
- Copernicus Sentinel-1 GRD              (SAR flood water detection)
- Copernicus Sentinel-2 SR               (optical NDVI/water indices)
- CIESIN GPW v4.11                       (population density)
- FAO GAUL 2015                          (administrative boundaries)
"""

from datetime import datetime, timedelta

try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    ee = None
    EE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tanzania region of interest
# ---------------------------------------------------------------------------
TANZANIA_BBOX = [29.33, -11.75, 40.45, -0.98]  # [minlon, minlat, maxlon, maxlat]

def get_tanzania_geometry():
    """Return Tanzania bounding-box geometry (fast; FAO boundary for precision)."""
    if not ee:
        return None
    return ee.Geometry.Rectangle(TANZANIA_BBOX)

def get_tanzania_boundary_precise():
    """FAO GAUL level-0 boundary for Tanzania (precise outline)."""
    if not ee:
        return None
    return (
        ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level0')
        .filter(ee.Filter.eq('ADM0_NAME', 'United Republic of Tanzania'))
        .first()
        .geometry()
    )


# ---------------------------------------------------------------------------
# Helper: safe ImageCollection accessor
# ---------------------------------------------------------------------------
def safe_collection_image(collection, reducer_fn, fallback_value, band_name):
    """
    Apply `reducer_fn` to `collection` and return the result renamed to `band_name`.
    If the collection is empty, return ee.Image.constant(fallback_value) instead.
    This eliminates 0-band image errors.
    """
    return ee.Image(ee.Algorithms.If(
        collection.size().gt(0),
        reducer_fn(collection).rename(band_name),
        ee.Image.constant(fallback_value).rename(band_name)
    ))


# ---------------------------------------------------------------------------
# 1. JRC Global Surface Water — permanent & seasonal water extent
# ---------------------------------------------------------------------------
def get_jrc_water_occurrence(geometry, min_occurrence=75):
    """
    Permanent water mask from JRC v1.4.
    min_occurrence: pixels with water >= this % of time are treated as permanently wet.
    Returns a 0/1 binary image clipped to geometry.
    """
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    permanent_water = jrc.select('occurrence').gte(min_occurrence).rename('permanent_water')
    return permanent_water.clip(geometry)

def get_jrc_seasonal_water(geometry):
    """
    Latest seasonal water extent from JRC monthly history.
    Returns a 0/1 image where 1 = water observed in latest month.
    """
    now = datetime.now()
    # JRC monthly history has ~2 month latency
    latest_month = now - timedelta(days=60)
    start = latest_month.replace(day=1).strftime('%Y-%m-%d')
    end = now.strftime('%Y-%m-%d')
    monthly = (
        ee.ImageCollection('JRC/GSW1_4/MonthlyHistory')
        .filterBounds(geometry)
        .filterDate(start, end)
        .select('water')
    )
    return safe_collection_image(
        monthly,
        lambda c: c.max().gte(2),  # 2 = water
        0,
        'seasonal_water'
    ).clip(geometry)

def get_jrc_water_change(geometry):
    """
    Water gain (1) or loss (0) from JRC v1.4 change summary.
    Useful for identifying new flood extents.
    """
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    # transition: 1=permanent water, 2=new permanent, 3=lost permanent, etc.
    transition = jrc.select('transition')
    # New water pixels (gained): transitions 2, 7, 8
    water_gain = transition.remap([2, 7, 8], [1, 1, 1], 0).rename('water_gain')
    return water_gain.clip(geometry)


# ---------------------------------------------------------------------------
# 2. SRTM DEM — topographic wetness index for flood susceptibility
# ---------------------------------------------------------------------------
def get_srtm_twi(geometry):
    """
    Topographic Wetness Index (TWI) from SRTM 30m DEM.
    TWI = ln(flow_accumulation / tan(slope))  (proxy using terrain ruggedness)
    Higher TWI = higher flood susceptibility.
    Returns a float image clipped to geometry.
    """
    srtm = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(geometry)
    terrain = ee.Algorithms.Terrain(srtm)
    slope_rad = terrain.select('slope').multiply(3.14159 / 180)
    tan_slope = slope_rad.tan().max(0.001)  # avoid division by zero

    # Use slope as proxy for flow accumulation; low slope → high TWI
    twi = tan_slope.pow(-1).log().rename('twi')
    return twi.clip(geometry)

def get_elevation(geometry):
    """Raw SRTM elevation in metres."""
    return ee.Image('USGS/SRTMGL1_003').select('elevation').clip(geometry).rename('elevation_m')

def get_flood_susceptibility_from_dem(geometry):
    """
    Flood susceptibility based on elevation percentile and slope.
    Low elevation + low slope = high susceptibility (0–1 scale).
    """
    srtm = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(geometry)
    terrain = ee.Algorithms.Terrain(srtm)
    slope = terrain.select('slope')

    # Normalise elevation to Tanzania range (~0–5895 m)
    elev_norm = srtm.unitScale(0, 5895).subtract(1).multiply(-1)  # invert: lower = more susceptible
    slope_norm = slope.unitScale(0, 45).subtract(1).multiply(-1)   # invert: flatter = more susceptible

    susceptibility = elev_norm.multiply(0.6).add(slope_norm.multiply(0.4)).rename('flood_susceptibility')
    return susceptibility.clamp(0, 1).clip(geometry)


# ---------------------------------------------------------------------------
# 3. GPM IMERG — near-real-time precipitation (better latency than CHIRPS)
# ---------------------------------------------------------------------------
def get_gpm_precipitation(start_date, end_date, geometry):
    """
    Total accumulated precipitation from GPM IMERG monthly composite.
    GPM has ~2-day latency globally; covers Tanzania well.
    Returns mm accumulated over the date range.
    """
    # Use IMERG Final (monthly) for longer periods; daily for short windows
    days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days

    if days > 25:
        # Monthly accumulation: multiply mm/hr × 24 × days_in_month
        gpm = (
            ee.ImageCollection('NASA/GPM_L3/IMERG_MONTHLY_V06')
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .select('precipitation')
        )
        return safe_collection_image(
            gpm,
            lambda c: c.sum().multiply(24).multiply(30),  # mm/hr → mm/month approx
            50.0,
            'gpm_precip_mm'
        ).clip(geometry)
    else:
        # Half-hourly → sum to daily using late-run product
        gpm = (
            ee.ImageCollection('NASA/GPM_L3/IMERG_V06')
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .select('precipitationCal')
        )
        return safe_collection_image(
            gpm,
            lambda c: c.sum().multiply(0.5),  # 30-min values → mm
            50.0,
            'gpm_precip_mm'
        ).clip(geometry)

def get_chirps_precipitation(start_date, end_date, geometry):
    """CHIRPS daily summed precipitation — long record (1981–present, ~1 month latency)."""
    chirps = (
        ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select('precipitation')
    )
    return safe_collection_image(
        chirps,
        lambda c: c.sum(),
        50.0,
        'chirps_precip_mm'
    ).clip(geometry)

def get_best_precipitation(start_date, end_date, geometry):
    """
    Return the best available precipitation estimate:
    GPM IMERG preferred (lower latency); falls back to CHIRPS.
    Combined band 'precip_mm' is normalised to mm total.
    """
    gpm = get_gpm_precipitation(start_date, end_date, geometry).rename('precip_mm')
    chirps = get_chirps_precipitation(start_date, end_date, geometry).rename('precip_mm')
    # If GPM has values, use it; blend with CHIRPS for reliability
    # A simple approach: unmask GPM → fill with CHIRPS
    combined = gpm.unmask(chirps).rename('precip_mm')
    return combined


# ---------------------------------------------------------------------------
# 4. ESA WorldCover 2021 — land cover (impervious → flood; bare soil → drought)
# ---------------------------------------------------------------------------
def get_land_cover(geometry):
    """
    ESA WorldCover 2021 at 10m resolution.
    Classes: 10=Tree cover, 20=Shrubland, 30=Grassland, 40=Cropland,
             50=Built-up, 60=Bare, 70=Snow/Ice, 80=Open water,
             90=Herbaceous wetland, 95=Mangroves, 100=Moss/Lichen
    """
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').clip(geometry)
    return wc

def get_impervious_fraction(geometry):
    """Fraction of built-up area from ESA WorldCover (high runoff → flood risk)."""
    wc = get_land_cover(geometry)
    impervious = wc.eq(50).rename('impervious_fraction').toFloat()
    return impervious.clip(geometry)

def get_bare_soil_fraction(geometry):
    """Fraction of bare/sparse land from WorldCover (drought & erosion indicator)."""
    wc = get_land_cover(geometry)
    bare = wc.eq(60).rename('bare_soil_fraction').toFloat()
    return bare.clip(geometry)

def get_cropland_mask(geometry):
    """Cropland pixels from ESA WorldCover (agricultural drought relevance)."""
    wc = get_land_cover(geometry)
    cropland = wc.eq(40).rename('cropland').toFloat()
    return cropland.clip(geometry)


# ---------------------------------------------------------------------------
# 5. MODIS Evapotranspiration — ET deficit as drought indicator
# ---------------------------------------------------------------------------
def get_modis_et(start_date, end_date, geometry):
    """
    MODIS MOD16A2 8-day ET (kg/m²/8day == mm/8day).
    ET deficit (low ET in growing season) → drought stress.
    Returns mean ET over the period scaled to mm/day.
    """
    et_col = (
        ee.ImageCollection('MODIS/061/MOD16A2')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select('ET')
    )
    return safe_collection_image(
        et_col,
        lambda c: c.mean().multiply(0.1),  # scale factor 0.1 (kg/m²/8day → mm/8day)
        3.0,
        'et_mm_8day'
    ).clip(geometry)

def get_modis_et_baseline(start_doy, end_doy, geometry):
    """Long-term mean ET for the same day-of-year window (2001–2020 baseline)."""
    historical = (
        ee.ImageCollection('MODIS/061/MOD16A2')
        .filterBounds(geometry)
        .filter(ee.Filter.calendarRange(start_doy, end_doy, 'day_of_year'))
        .filter(ee.Filter.calendarRange(2001, 2020, 'year'))
        .select('ET')
        .map(lambda img: img.multiply(0.1))
    )
    return safe_collection_image(
        historical,
        lambda c: c.mean(),
        3.0,
        'et_baseline_mm_8day'
    ).clip(geometry)


# ---------------------------------------------------------------------------
# 6. NASA FLDAS — multi-model land surface (soil moisture, runoff, ET)
# ---------------------------------------------------------------------------
def get_fldas_soil_moisture(start_date, end_date, geometry):
    """
    FLDAS NOAH monthly soil moisture (m³/m³) at 0.1° resolution.
    More recent and less deprecated than SMAP 10km product.
    Two layers: SoilMoi00_10cm_tavg (top layer) and SoilMoi10_40cm_tavg.
    """
    fldas = (
        ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M/V001')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select(['SoilMoi00_10cm_tavg', 'SoilMoi10_40cm_tavg'])
    )
    top = safe_collection_image(
        fldas,
        lambda c: c.select('SoilMoi00_10cm_tavg').mean(),
        0.25,
        'sm_top_layer'
    )
    sub = safe_collection_image(
        fldas,
        lambda c: c.select('SoilMoi10_40cm_tavg').mean(),
        0.25,
        'sm_sub_layer'
    )
    # Composite: weighted average (top layer slightly more weight)
    composite = top.multiply(0.6).add(sub.multiply(0.4)).rename('fldas_soil_moisture')
    return composite.clip(geometry)

def get_fldas_runoff(start_date, end_date, geometry):
    """FLDAS surface runoff (kg/m²/s) → high runoff = flood precursor."""
    fldas = (
        ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M/V001')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select('Qs_tavg')
    )
    return safe_collection_image(
        fldas,
        lambda c: c.mean().multiply(86400),  # kg/m²/s → mm/day
        0.5,
        'surface_runoff_mm_day'
    ).clip(geometry)


# ---------------------------------------------------------------------------
# 7. NASA SMAP SPL3SMP_E v005 — improved soil moisture (9 km)
# ---------------------------------------------------------------------------
def get_smap_soil_moisture(start_date, end_date, geometry):
    """
    NASA SMAP Enhanced L3 Radiometer (SPL3SMP_E v005) at 9 km.
    Replaces the deprecated NASA_USDA/HSL/SMAP10KM_soil_moisture product.
    Band: soil_moisture_am (morning overpass, more stable).
    """
    smap_v5 = (
        ee.ImageCollection('NASA/SMAP/SPL3SMP_E/005')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select('soil_moisture_am')
    )
    result = safe_collection_image(
        smap_v5,
        lambda c: c.median(),
        0.25,
        'smap_soil_moisture'
    )

    # Fallback to FLDAS if SMAP unavailable
    fldas = get_fldas_soil_moisture(start_date, end_date, geometry)
    return result.unmask(fldas.rename('smap_soil_moisture')).clip(geometry)


# ---------------------------------------------------------------------------
# 8. NASA FIRMS — active fires (drought stress & land degradation proxy)
# ---------------------------------------------------------------------------
def get_fire_density(start_date, end_date, geometry):
    """
    FIRMS MODIS active fires as fire occurrence density (0/1 per pixel summed).
    High fire density in non-agricultural areas → drought-stressed vegetation.
    """
    firms = (
        ee.ImageCollection('FIRMS')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select('T21')  # brightness temperature channel
    )
    # Count fire detections: T21 brightness temp > 300K = likely fire
    # Must map over each Image in the collection, not call .gt() on the collection itself
    fire_count = safe_collection_image(
        firms,
        lambda c: c.map(lambda img: img.gt(300)).sum().toFloat(),
        0,
        'fire_count'
    ).clip(geometry)
    return fire_count


# ---------------------------------------------------------------------------
# 9. ERA5-Land — meteorological reanalysis
# ---------------------------------------------------------------------------
def get_era5_temperature(start_date, end_date, geometry):
    """ERA5-Land daily 2m air temperature (K → °C)."""
    era5 = (
        ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select('temperature_2m')
    )
    return safe_collection_image(
        era5,
        lambda c: c.mean().subtract(273.15),  # K → °C
        28.0,
        'air_temp_celsius'
    ).clip(geometry)

def get_era5_wind_speed(start_date, end_date, geometry):
    """ERA5-Land 10m wind speed (m/s) — used for evaporation estimation."""
    era5 = (
        ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select(['u_component_of_wind_10m', 'v_component_of_wind_10m'])
    )
    def wind_speed(img):
        u = img.select('u_component_of_wind_10m')
        v = img.select('v_component_of_wind_10m')
        return u.pow(2).add(v.pow(2)).sqrt().rename('wind_speed_ms')

    wind_col = safe_collection_image(
        era5,
        lambda c: c.map(wind_speed).mean(),
        3.0,
        'wind_speed_ms'
    )
    return wind_col.clip(geometry)


# ---------------------------------------------------------------------------
# 10. Sentinel-1 SAR — real-time flood water detection
# ---------------------------------------------------------------------------
def get_sentinel1_flood_mask(start_date, end_date, geometry, vv_threshold=-18.0):
    """
    Sentinel-1 GRD IW VV polarisation flood water mask.
    Pixels with VV < threshold (dB) are classified as open water.
    Compared against JRC permanent water to isolate flood extent.
    Returns: water_mask (0/1), anomaly (flood above normal water)
    """
    s1 = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .select('VV')
    )

    current_water = safe_collection_image(
        s1,
        lambda c: c.median().lt(vv_threshold).toFloat(),
        0.0,
        'sar_water_mask'
    ).clip(geometry)

    # Remove permanent water bodies to get flood extent only
    permanent = get_jrc_water_occurrence(geometry, min_occurrence=80)
    flood_only = current_water.subtract(permanent.toFloat()).max(0).rename('flood_extent')

    return {
        'water_mask': current_water,
        'flood_extent': flood_only,
        'combined': current_water.rename('sar_combined_water')
    }


# ---------------------------------------------------------------------------
# 11. Composite flood risk image — multi-source fusion
# ---------------------------------------------------------------------------
def build_flood_risk_composite(start_date, end_date, geometry, forecast_days=0):
    """
    Build a comprehensive flood risk composite from multiple open data sources.

    Key calibration notes for Tanzania:
    - SAR-detected water is masked against permanent water bodies so that
      Lake Victoria / Rufiji delta do not permanently bias every region as high.
    - Precipitation thresholds use Tanzania long-term mean: ~80 mm/30-day in
      dry season, up to ~200 mm/30-day in peak wet season (Mar, May).
      We normalise against the 30-day extreme (400 mm) so moderate rain maps
      to moderate risk, not near-maximum.
    - TWI contribution is capped so flat coastal plains don't dominate.
    - The composite is *anomaly-aware*: only SAR-detected NEW water (above
      permanent baseline) contributes to risk score.

    Returns an ee.Image with bands:
      'flood_risk'  — 0 (very low) … 1 (very high), float
      'flood_class' — 1 (very low) … 5 (very high), byte
    """
    # ---- Raw satellite inputs ----
    sar_result = get_sentinel1_flood_mask(start_date, end_date, geometry)
    # flood_extent = new water above permanent baseline (anomaly only)
    flood_extent = sar_result['flood_extent']          # 0/1 anomaly water
    sar_combined = sar_result['combined']               # 0/1 all water

    permanent_water = get_jrc_water_occurrence(geometry, 80)   # 0/1
    precip = get_best_precipitation(start_date, end_date, geometry)  # mm accumulated
    twi = get_srtm_twi(geometry)
    impervious = get_impervious_fraction(geometry)      # 0/1
    runoff = get_fldas_runoff(start_date, end_date, geometry)   # mm/day
    sm = get_smap_soil_moisture(start_date, end_date, geometry).rename('sm')

    # ---- Normalise each component to 0–1 risk ----

    # FLOOD ANOMALY from SAR (only counts new water, not lakes/rivers)
    # Weight 0.30 — direct observation is most reliable
    sar_risk = flood_extent.rename('sar_risk').clamp(0, 1)

    # PRECIPITATION: Tanzania extreme 30-day = ~400 mm; dry baseline ~40 mm.
    # Use a sigmoid-like unitScale: 0 mm → 0.0, 200 mm → 0.5, 400 mm → 1.0
    precip_norm = precip.unitScale(0, 400).clamp(0, 1).rename('precip_risk')

    # TWI: use tighter normalisation (2–8); cap contribution so it only
    # modulates, not dominates. High TWI (flat, waterlogged terrain) = higher risk.
    twi_norm = twi.unitScale(2, 8).clamp(0, 1).rename('twi_risk')

    # RUNOFF: Tanzania extreme ~15 mm/day; normalise accordingly
    runoff_norm = runoff.unitScale(0, 15).clamp(0, 1).rename('runoff_risk')

    # SOIL MOISTURE: saturation threshold 0.35–0.50 m³/m³ for clay-loam soils
    sm_risk = sm.unitScale(0.25, 0.50).clamp(0, 1).rename('sm_risk')

    # IMPERVIOUS: urban fraction rarely > 30% outside Dar es Salaam
    impervious_norm = impervious.clamp(0, 1).rename('impervious_risk')

    # ---- Weighted composite ----
    # SAR anomaly (0.30) + Precip (0.28) + Runoff (0.18) +
    # SM saturation (0.12) + TWI (0.08) + Impervious (0.04)
    flood_risk = (
        sar_risk.multiply(0.30)
        .add(precip_norm.multiply(0.28))
        .add(runoff_norm.multiply(0.18))
        .add(sm_risk.multiply(0.12))
        .add(twi_norm.multiply(0.08))
        .add(impervious_norm.multiply(0.04))
    ).rename('flood_risk').clamp(0, 1)

    # Permanent water bodies are reference only — do NOT inflate risk score
    # (avoids Lake Victoria making all surrounding pixels look flooded)
    # Instead, set permanent water to a moderate 0.40 so they appear on map
    # but don't pull regional mean upward.
    flood_risk = flood_risk.where(permanent_water.eq(1), 0.40)

    # Forecast uncertainty: small conservative uplift for future windows
    if forecast_days > 0:
        uplift = min(1.05, 1.0 + forecast_days * 0.005)  # max +5% at 10 days
        flood_risk = flood_risk.multiply(uplift).clamp(0, 1)

    # Classify 1 (very low) → 5 (very high)
    flood_class = flood_risk.expression(
        "(b('flood_risk') < 0.20) ? 1"
        ": (b('flood_risk') < 0.40) ? 2"
        ": (b('flood_risk') < 0.60) ? 3"
        ": (b('flood_risk') < 0.80) ? 4"
        ": 5"
    ).rename('flood_class').byte()

    return ee.Image.cat([flood_risk, flood_class]).clip(geometry)


# ---------------------------------------------------------------------------
# 12. Composite drought risk image — multi-source fusion
# ---------------------------------------------------------------------------
def build_drought_risk_composite(start_date, end_date, geometry, forecast_days=0):
    """
    Build a calibrated drought risk composite for Tanzania.

    Calibration notes
    -----------------
    Tanzania's baseline is semi-arid to sub-humid (250–1800 mm/yr).
    A naive NDVI < 0.4 threshold would flag almost all rangeland as drought,
    which is incorrect — those are structurally sparse ecosystems.

    Strategy: compute *anomalies* relative to a long-term seasonal baseline
    (2001-2020) using pixelwise comparison.  Then classify drought only when
    the current period deviates significantly below the expected value.

    Components
    ----------
    1. NDVI anomaly   — current vs 20-year same-period median (VCI proxy)
    2. Precip anomaly — SPI-like: (current – LTA) / LTA_std
    3. Soil moisture  — absolute SMAP value (calibrated for Africa soils)
    4. ET deficit     — MODIS ET anomaly vs historical
    5. LST anomaly    — ERA5 temperature deviation from seasonal mean
    6. Fire density   — actively burning pixels (stress proxy, low weight)

    Returns ee.Image with:
      'drought_risk'   — 0 (no drought) … 1 (extreme drought)
      'drought_class'  — 1 (extreme) … 5 (no drought), byte
    """
    # ---- Current-period data ----
    ndvi_col = (
        ee.ImageCollection('MODIS/061/MOD13A2')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select('NDVI')
    )
    current_ndvi = safe_collection_image(
        ndvi_col,
        lambda c: c.median().multiply(0.0001),
        0.35,   # Tanzania rangeland typical median; NOT 0 — avoids false extreme
        'ndvi'
    )

    precip_current = get_chirps_precipitation(start_date, end_date, geometry)
    sm = get_smap_soil_moisture(start_date, end_date, geometry)
    et_current = get_modis_et(start_date, end_date, geometry)
    fires = get_fire_density(start_date, end_date, geometry)
    temp = get_era5_temperature(start_date, end_date, geometry)

    # ---- Historical baselines (same calendar window, 2001-2020) ----
    start_month = ee.Date(start_date).get('month')
    end_month = ee.Date(end_date).get('month')

    ndvi_hist = (
        ee.ImageCollection('MODIS/061/MOD13A2')
        .filterBounds(geometry)
        .filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
        .filter(ee.Filter.calendarRange(2001, 2020, 'year'))
        .select('NDVI')
        .map(lambda img: img.multiply(0.0001))
    )
    ndvi_lta_median = safe_collection_image(
        ndvi_hist, lambda c: c.median(), 0.35, 'ndvi_lta'
    )
    ndvi_lta_p10 = safe_collection_image(
        ndvi_hist,
        lambda c: c.reduce(ee.Reducer.percentile([10])),
        0.15,
        'ndvi_lta_p10'
    )

    # VCI = (current - min) / (max - min); use p10 as min, p90 as max
    ndvi_lta_p90 = safe_collection_image(
        ndvi_hist,
        lambda c: c.reduce(ee.Reducer.percentile([90])),
        0.65,
        'ndvi_lta_p90'
    )
    vci_range = ndvi_lta_p90.subtract(ndvi_lta_p10).max(0.05)
    vci = current_ndvi.subtract(ndvi_lta_p10).divide(vci_range).clamp(0, 1).rename('vci')
    # VCI 0 = extreme drought, 1 = excellent; invert for risk
    vci_risk = ee.Image.constant(1).subtract(vci).rename('vci_risk')

    # ---- Precipitation SPI anomaly ----
    def _yearly_precip_sum(year):
        y = ee.Number(year)
        s = ee.Date.fromYMD(y, ee.Date(start_date).get('month'), ee.Date(start_date).get('day'))
        e = s.advance(
            ee.Date(end_date).difference(ee.Date(start_date), 'day'),
            'day'
        )
        return (
            ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
            .filterBounds(geometry)
            .filterDate(s, e)
            .select('precipitation')
            .sum()
            .rename('precip')
        )

    years = ee.List.sequence(1981, 2020)
    annual_precip_ic = ee.ImageCollection(years.map(_yearly_precip_sum))
    precip_lta_mean = safe_collection_image(annual_precip_ic, lambda c: c.mean(), 80.0, 'precip_mean')
    precip_lta_std = safe_collection_image(
        annual_precip_ic,
        lambda c: c.reduce(ee.Reducer.stdDev()),
        35.0,
        'precip_std'
    )
    # Standardised Precipitation Index (SPI): negative = below average
    spi = (
        precip_current.rename('p')
        .subtract(precip_lta_mean.rename('p'))
        .divide(precip_lta_std.rename('p').max(5.0))
    )
    # Map SPI to 0-1 risk: SPI -2 → risk 1.0; SPI +2 → risk 0.0
    spi_risk = spi.multiply(-1).add(2).divide(4).clamp(0, 1).rename('spi_risk')

    # ---- Soil moisture anomaly ----
    # SMAP: Africa soils wilting point ~0.08, field capacity ~0.35
    # Risk = 1 when SM <= wilting point (0.08), 0 when SM >= 0.30
    sm_risk = sm.subtract(0.30).multiply(-1).divide(0.22).clamp(0, 1).rename('sm_risk')

    # ---- ET anomaly relative to seasonal baseline ----
    start_month_py = ee.Date(start_date).get('month')
    end_month_py = ee.Date(end_date).get('month')
    et_hist = (
        ee.ImageCollection('MODIS/061/MOD16A2')
        .filterBounds(geometry)
        .filter(ee.Filter.calendarRange(start_month_py, end_month_py, 'month'))
        .filter(ee.Filter.calendarRange(2001, 2020, 'year'))
        .select('ET')
        .map(lambda img: img.multiply(0.1))
    )
    et_lta = safe_collection_image(et_hist, lambda c: c.mean(), 3.0, 'et_lta')
    # ET deficit risk: when current ET < 50 % of baseline → high risk
    et_ratio = et_current.rename('et').divide(et_lta.rename('et').max(0.5))
    et_risk = ee.Image.constant(1).subtract(et_ratio).clamp(0, 1).rename('et_risk')

    # ---- LST anomaly ----
    # Tanzania: dry season mean ~28 °C, wet season ~24 °C.
    # Elevated temp > 3 °C above baseline signals heat stress.
    temp_hist = (
        ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
        .filterBounds(geometry)
        .filter(ee.Filter.calendarRange(start_month_py, end_month_py, 'month'))
        .filter(ee.Filter.calendarRange(2001, 2020, 'year'))
        .select('temperature_2m')
        .map(lambda img: img.subtract(273.15))
    )
    temp_lta = safe_collection_image(temp_hist, lambda c: c.mean(), 26.0, 'temp_lta')
    # Positive anomaly (hotter than normal) → higher drought risk
    temp_anomaly = temp.rename('t').subtract(temp_lta.rename('t'))
    # Cap at ±5 °C anomaly range; +5 → risk 1.0, -5 → risk 0.0
    temp_risk = temp_anomaly.add(5).divide(10).clamp(0, 1).rename('temp_risk')

    # ---- Fire density ----
    fire_risk = fires.unitScale(0, 3).clamp(0, 1).rename('fire_risk')

    # ---- Weighted composite (higher = more drought risk) ----
    # VCI (0.30) + SPI (0.28) + SM (0.20) + ET deficit (0.12) +
    # Temp anomaly (0.06) + Fire (0.04)
    drought_risk = (
        vci_risk.multiply(0.30)
        .add(spi_risk.multiply(0.28))
        .add(sm_risk.multiply(0.20))
        .add(et_risk.multiply(0.12))
        .add(temp_risk.multiply(0.06))
        .add(fire_risk.multiply(0.04))
    ).rename('drought_risk').clamp(0, 1)

    # Conservative forecast uplift (max +4 % at day 7)
    if forecast_days > 0:
        uplift = min(1.06, 1.0 + forecast_days * 0.006)
        drought_risk = drought_risk.multiply(uplift).clamp(0, 1)

    # Classify: 1=extreme drought (risk > 0.80) … 5=no drought (risk ≤ 0.20)
    drought_class = drought_risk.expression(
        "(b('drought_risk') > 0.80) ? 1"
        ": (b('drought_risk') > 0.60) ? 2"
        ": (b('drought_risk') > 0.40) ? 3"
        ": (b('drought_risk') > 0.20) ? 4"
        ": 5"
    ).rename('drought_class').byte()

    return ee.Image.cat([drought_risk, drought_class]).clip(geometry)


# ---------------------------------------------------------------------------
# 13. MODIS NDVI + EVI for vegetation health
# ---------------------------------------------------------------------------
def get_modis_ndvi_evi(start_date, end_date, geometry):
    """16-day MODIS NDVI and EVI composites."""
    col = (
        ee.ImageCollection('MODIS/061/MOD13A2')
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select(['NDVI', 'EVI'])
    )
    ndvi = safe_collection_image(
        col, lambda c: c.select('NDVI').median().multiply(0.0001), 0.4, 'ndvi'
    )
    evi = safe_collection_image(
        col, lambda c: c.select('EVI').median().multiply(0.0001), 0.3, 'evi'
    )
    return ee.Image.cat([ndvi, evi]).clip(geometry)


# ---------------------------------------------------------------------------
# 14. Convenience: full risk band package for regional reduction
# ---------------------------------------------------------------------------
def build_full_analysis_bands(start_date, end_date, geometry, prediction_type='flood'):
    """
    Build a multi-band image optimised for batch regional statistics.
    Returns an ee.Image with standardised bands regardless of prediction type.
    All bands are protected against empty-collection 0-band errors.
    """
    if prediction_type == 'flood':
        composite = build_flood_risk_composite(start_date, end_date, geometry)
    else:
        composite = build_drought_risk_composite(start_date, end_date, geometry)

    # Common ancillary bands
    precip = get_best_precipitation(start_date, end_date, geometry)
    ndvi_evi = get_modis_ndvi_evi(start_date, end_date, geometry)
    sm = get_smap_soil_moisture(start_date, end_date, geometry)
    area_km2 = ee.Image.pixelArea().divide(1e6).rename('area_km2')

    return ee.Image.cat([
        composite,
        precip,
        ndvi_evi,
        sm,
        area_km2,
    ])
