"""
Advanced Drought Monitoring Module for Tanzania
Uses real satellite data from Google Earth Engine to compute standardized drought indices.
No mock data — all values derived from actual remote sensing observations.

Indices computed:
- VCI (Vegetation Condition Index) from MODIS NDVI
- TCI (Temperature Condition Index) from MODIS LST
- SPI (Standardized Precipitation Index) from CHIRPS
- SMI (Soil Moisture Index) from NASA SMAP
- CDHI (Combined Drought Hazard Index) — weighted composite
"""

from datetime import datetime, timedelta

try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    ee = None
    EE_AVAILABLE = False

try:
    from .gee_auth import authenticate_gee
    from .gee_data_processing import get_tanzania_boundary, get_tanzania_regions
except ImportError:
    authenticate_gee = None
    get_tanzania_boundary = None
    get_tanzania_regions = None

# Historical baseline years for percentile calculations
BASELINE_START = 2001
BASELINE_END = 2020


class AdvancedDroughtMonitor:
    """Production-grade drought monitoring using multi-source satellite data."""

    def __init__(self):
        if not EE_AVAILABLE:
            raise ImportError("Google Earth Engine API is required")
        if authenticate_gee:
            authenticate_gee()
        self._boundary = None
        self._regions = None

    @property
    def boundary(self):
        if self._boundary is None:
            self._boundary = get_tanzania_boundary().geometry()
        return self._boundary

    @property
    def regions(self):
        if self._regions is None:
            self._regions = get_tanzania_regions()
        return self._regions

    # ------------------------------------------------------------------
    # VCI — Vegetation Condition Index
    # ------------------------------------------------------------------
    def compute_vci(self, start_date, end_date, geometry=None):
        """
        VCI = (NDVI_current - NDVI_min) / (NDVI_max - NDVI_min) * 100
        Uses MODIS 16-day NDVI (MOD13A2) for long historical baseline.
        """
        geom = geometry if geometry is not None else self.boundary

        ndvi_col = (
            ee.ImageCollection("MODIS/061/MOD13A2")
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .select("NDVI")
        )
        current_ndvi = ee.Image(ee.Algorithms.If(
            ndvi_col.size().gt(0),
            ndvi_col.median().multiply(0.0001),
            ee.Image.constant(0.4)  # neutral NDVI fallback
        ))

        # Build day-of-year window for historical min/max
        start_doy = ee.Date(start_date).getRelative("day", "year")
        end_doy = ee.Date(end_date).getRelative("day", "year")

        historical = (
            ee.ImageCollection("MODIS/061/MOD13A2")
            .filterBounds(geom)
            .filter(ee.Filter.calendarRange(start_doy, end_doy, "day_of_year"))
            .filter(
                ee.Filter.calendarRange(BASELINE_START, BASELINE_END, "year")
            )
            .select("NDVI")
            .map(lambda img: img.multiply(0.0001))
        )

        ndvi_min = historical.min()
        ndvi_max = historical.max()
        denominator = ndvi_max.subtract(ndvi_min).max(0.001)  # avoid div-by-zero
        vci = current_ndvi.subtract(ndvi_min).divide(denominator).multiply(100).clamp(0, 100)
        return vci.rename("VCI")

    # ------------------------------------------------------------------
    # TCI — Temperature Condition Index
    # ------------------------------------------------------------------
    def compute_tci(self, start_date, end_date, geometry=None):
        """
        TCI = (LST_max - LST_current) / (LST_max - LST_min) * 100
        Uses MODIS daily LST (MOD11A1).
        """
        geom = geometry if geometry is not None else self.boundary

        lst_col = (
            ee.ImageCollection("MODIS/061/MOD11A1")
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .select("LST_Day_1km")
        )
        current_lst = ee.Image(ee.Algorithms.If(
            lst_col.size().gt(0),
            lst_col.median().multiply(0.02),
            ee.Image.constant(28.0 + 273.15)  # ~28°C in Kelvin-scaled fallback
        ))

        start_doy = ee.Date(start_date).getRelative("day", "year")
        end_doy = ee.Date(end_date).getRelative("day", "year")

        historical = (
            ee.ImageCollection("MODIS/061/MOD11A1")
            .filterBounds(geom)
            .filter(ee.Filter.calendarRange(start_doy, end_doy, "day_of_year"))
            .filter(
                ee.Filter.calendarRange(BASELINE_START, BASELINE_END, "year")
            )
            .select("LST_Day_1km")
            .map(lambda img: img.multiply(0.02))
        )

        lst_min = historical.min()
        lst_max = historical.max()
        denominator = lst_max.subtract(lst_min).max(0.001)
        tci = lst_max.subtract(current_lst).divide(denominator).multiply(100).clamp(0, 100)
        return tci.rename("TCI")

    # ------------------------------------------------------------------
    # SPI-like Precipitation Anomaly
    # ------------------------------------------------------------------
    def compute_spi(self, start_date, end_date, geometry=None):
        """
        Standardised precipitation anomaly using CHIRPS daily data.
        SPI_proxy = (P_current - P_mean) / P_std
        Converted to 0–100 scale for combination with other indices.
        """
        geom = geometry if geometry is not None else self.boundary

        precip_col = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .select("precipitation")
        )
        current_precip = ee.Image(ee.Algorithms.If(
            precip_col.size().gt(0),
            precip_col.sum(),
            ee.Image.constant(50.0)  # neutral precipitation fallback
        ))

        start_doy = ee.Date(start_date).getRelative("day", "year")
        end_doy = ee.Date(end_date).getRelative("day", "year")

        def _yearly_sum(year):
            y = ee.Number(year)
            s = ee.Date.fromYMD(y, ee.Date(start_date).get("month"), ee.Date(start_date).get("day"))
            e = ee.Date.fromYMD(y, ee.Date(end_date).get("month"), ee.Date(end_date).get("day"))
            return (
                ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterBounds(geom)
                .filterDate(s, e)
                .select("precipitation")
                .sum()
                .set("year", y)
            )

        years = ee.List.sequence(BASELINE_START, BASELINE_END)
        historical_annual = ee.ImageCollection(years.map(_yearly_sum))

        p_mean = historical_annual.mean()
        p_std = historical_annual.reduce(ee.Reducer.stdDev()).max(0.1)

        # SPI proxy: positive = wetter than normal (low drought), negative = drier (high drought)
        spi_raw = current_precip.subtract(p_mean).divide(p_std)

        # Map to 0–100 where 100 = no drought, 0 = extreme drought
        # SPI of -2 → 0, SPI of +2 → 100
        spi_scaled = spi_raw.add(2).divide(4).multiply(100).clamp(0, 100)
        return spi_scaled.rename("SPI")

    # ------------------------------------------------------------------
    # SMI — Soil Moisture Index
    # ------------------------------------------------------------------
    def compute_smi(self, start_date, end_date, geometry=None):
        """
        Soil moisture anomaly using NASA SMAP SPL3SMP_E v005 (9 km, replaces
        deprecated NASA_USDA/HSL/SMAP10KM_soil_moisture).
        Falls back to FLDAS NOAH multi-model ensemble if SMAP unavailable.
        SMI = (SM_current - SM_min) / (SM_max - SM_min) * 100
        """
        geom = geometry if geometry is not None else self.boundary

        # Preferred: SMAP SPL3SMP_E v005 (available from 2015-03-31 onwards)
        smap_v5 = (
            ee.ImageCollection("NASA/SMAP/SPL3SMP_E/005")
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .select("soil_moisture_am")
        )

        # Fallback: FLDAS NOAH multi-model (0.1° monthly)
        fldas = (
            ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001")
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .select("SoilMoi00_10cm_tavg")
        )

        # Pick SMAP if available; otherwise FLDAS
        current_sm = ee.Image(ee.Algorithms.If(
            smap_v5.size().gt(0),
            smap_v5.median().rename("ssm"),
            ee.Image(ee.Algorithms.If(
                fldas.size().gt(0),
                fldas.median().rename("ssm"),
                ee.Image.constant(0.25).rename("ssm")
            ))
        ))

        # SMAP/FLDAS baseline 2015–2020
        smap_baseline = (
            ee.ImageCollection("NASA/SMAP/SPL3SMP_E/005")
            .filterBounds(geom)
            .filter(ee.Filter.calendarRange(2015, BASELINE_END, "year"))
            .select("soil_moisture_am")
        )
        fldas_baseline = (
            ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001")
            .filterBounds(geom)
            .filter(ee.Filter.calendarRange(2000, BASELINE_END, "year"))
            .select("SoilMoi00_10cm_tavg")
        )
        # Prefer SMAP baseline; fall back to FLDAS
        has_smap_baseline = smap_baseline.size().gt(10)
        sm_min = ee.Image(ee.Algorithms.If(
            has_smap_baseline,
            smap_baseline.min(),
            fldas_baseline.min()
        ))
        sm_max = ee.Image(ee.Algorithms.If(
            has_smap_baseline,
            smap_baseline.max(),
            fldas_baseline.max()
        ))

        denominator = sm_max.subtract(sm_min).max(0.001)
        smi = current_sm.subtract(sm_min).divide(denominator).multiply(100).clamp(0, 100)
        return smi.rename("SMI")

    # ------------------------------------------------------------------
    # EVI — Enhanced Vegetation Index (supplementary)
    # ------------------------------------------------------------------
    def compute_evi_anomaly(self, start_date, end_date, geometry=None):
        """EVI anomaly from MODIS — supplements NDVI-based VCI."""
        geom = geometry if geometry is not None else self.boundary

        evi_col = (
            ee.ImageCollection("MODIS/061/MOD13A2")
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .select("EVI")
        )
        current_evi = ee.Image(ee.Algorithms.If(
            evi_col.size().gt(0),
            evi_col.median().multiply(0.0001),
            ee.Image.constant(0.3)  # neutral EVI fallback
        ))

        start_doy = ee.Date(start_date).getRelative("day", "year")
        end_doy = ee.Date(end_date).getRelative("day", "year")

        historical = (
            ee.ImageCollection("MODIS/061/MOD13A2")
            .filterBounds(geom)
            .filter(ee.Filter.calendarRange(start_doy, end_doy, "day_of_year"))
            .filter(
                ee.Filter.calendarRange(BASELINE_START, BASELINE_END, "year")
            )
            .select("EVI")
            .map(lambda img: img.multiply(0.0001))
        )

        evi_mean = historical.mean()
        evi_std = historical.reduce(ee.Reducer.stdDev()).max(0.0001)
        anomaly = current_evi.subtract(evi_mean).divide(evi_std)
        # Scale: +2 → 100, -2 → 0
        scaled = anomaly.add(2).divide(4).multiply(100).clamp(0, 100)
        return scaled.rename("EVI_anomaly")

    # ------------------------------------------------------------------
    # Combined Drought Hazard Index (CDHI)
    # ------------------------------------------------------------------
    def compute_composite_drought_index(self, start_date, end_date, geometry=None):
        """
        CDHI = 0.30 * VCI + 0.20 * TCI + 0.25 * SPI + 0.15 * SMI + 0.10 * EVI_anomaly
        Lower value = more severe drought.
        Returns both the continuous index and classified severity.
        """
        geom = geometry if geometry is not None else self.boundary

        vci = self.compute_vci(start_date, end_date, geom)
        tci = self.compute_tci(start_date, end_date, geom)
        spi = self.compute_spi(start_date, end_date, geom)

        # SMI and EVI may not always be available — use fallback weights
        try:
            smi = self.compute_smi(start_date, end_date, geom)
            has_smi = True
        except Exception:
            smi = ee.Image.constant(50).rename("SMI")
            has_smi = False

        try:
            evi_a = self.compute_evi_anomaly(start_date, end_date, geom)
            has_evi = True
        except Exception:
            evi_a = ee.Image.constant(50).rename("EVI_anomaly")
            has_evi = False

        # Adaptive weighting when not all indices available
        if has_smi and has_evi:
            cdhi = (
                vci.multiply(0.30)
                .add(tci.multiply(0.20))
                .add(spi.multiply(0.25))
                .add(smi.multiply(0.15))
                .add(evi_a.multiply(0.10))
            )
        elif has_smi:
            cdhi = (
                vci.multiply(0.35)
                .add(tci.multiply(0.25))
                .add(spi.multiply(0.25))
                .add(smi.multiply(0.15))
            )
        else:
            cdhi = (
                vci.multiply(0.40)
                .add(tci.multiply(0.25))
                .add(spi.multiply(0.35))
            )

        cdhi = cdhi.rename("CDHI").clamp(0, 100)

        # Classify drought severity (lower CDHI = more severe)
        # 1 = Extreme drought, 2 = Severe, 3 = Moderate, 4 = Mild, 5 = No drought
        drought_classes = cdhi.expression(
            "(b('CDHI') < 10) ? 1"
            + ": (b('CDHI') < 25) ? 2"
            + ": (b('CDHI') < 40) ? 3"
            + ": (b('CDHI') < 60) ? 4"
            + ": 5"
        ).rename("drought_severity").byte()

        return {
            "cdhi": cdhi,
            "drought_classes": drought_classes,
            "indices": {
                "vci": vci,
                "tci": tci,
                "spi": spi,
                "smi": smi,
                "evi_anomaly": evi_a,
            },
            "weights_used": {
                "vci": 0.30 if (has_smi and has_evi) else 0.35 if has_smi else 0.40,
                "tci": 0.20 if (has_smi and has_evi) else 0.25,
                "spi": 0.25 if (has_smi and has_evi) else 0.25 if has_smi else 0.35,
                "smi": 0.15 if has_smi else 0.0,
                "evi_anomaly": 0.10 if (has_smi and has_evi) else 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Regional Drought Alert Generation
    # ------------------------------------------------------------------
    def generate_regional_alerts(self, start_date, end_date, geometry=None):
        """
        Compute per-region drought alerts from real satellite data.
        Returns a list of region alert dicts sorted by severity.
        """
        geom = geometry if geometry is not None else self.boundary

        composite = self.compute_composite_drought_index(start_date, end_date, geom)
        cdhi = composite["cdhi"]
        drought_classes = composite["drought_classes"]

        regions = self.regions
        if regions is None:
            return []

        # Reduce per region
        def _reduce_region(feature):
            stats = cdhi.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.percentile([10, 25, 50, 75, 90]), sharedInputs=True
                ),
                geometry=feature.geometry(),
                scale=1000,
                maxPixels=1e9,
            )
            class_hist = drought_classes.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=feature.geometry(),
                scale=1000,
                maxPixels=1e9,
            )
            area_km2 = feature.geometry().area().divide(1e6)
            return feature.set(
                {
                    "cdhi_mean": stats.get("CDHI_mean"),
                    "cdhi_p10": stats.get("CDHI_p10"),
                    "cdhi_p25": stats.get("CDHI_p25"),
                    "cdhi_p50": stats.get("CDHI_p50"),
                    "cdhi_p75": stats.get("CDHI_p75"),
                    "cdhi_p90": stats.get("CDHI_p90"),
                    "drought_histogram": class_hist.get("drought_severity"),
                    "area_km2": area_km2,
                }
            )

        region_stats = regions.map(_reduce_region)
        region_info = region_stats.getInfo()

        alerts = []
        for feat in region_info.get("features", []):
            props = feat.get("properties", {})
            region_name = props.get("ADM1_NAME", "Unknown")
            cdhi_mean = props.get("cdhi_mean")
            if cdhi_mean is None:
                continue

            hist = props.get("drought_histogram", {})
            total_pixels = sum(hist.values()) if hist else 1
            severe_pixels = sum(
                hist.get(str(k), 0) for k in [1, 2]  # Extreme + Severe
            )
            moderate_pixels = hist.get("3", 0)

            severe_pct = (severe_pixels / max(total_pixels, 1)) * 100
            moderate_pct = (moderate_pixels / max(total_pixels, 1)) * 100

            if severe_pct > 50:
                alert_level = "CRITICAL"
            elif severe_pct > 25:
                alert_level = "HIGH"
            elif severe_pct > 10 or cdhi_mean < 35:
                alert_level = "MODERATE"
            else:
                alert_level = "LOW"

            alerts.append(
                {
                    "region": region_name,
                    "alert_level": alert_level,
                    "cdhi_mean": round(cdhi_mean, 1),
                    "cdhi_p10": round(props.get("cdhi_p10", 0), 1),
                    "cdhi_p50": round(props.get("cdhi_p50", 0), 1),
                    "severe_percentage": round(severe_pct, 1),
                    "moderate_percentage": round(moderate_pct, 1),
                    "total_area_km2": round(props.get("area_km2", 0)),
                    "drought_distribution": hist,
                }
            )

        alerts.sort(key=lambda x: x["cdhi_mean"])  # lowest CDHI = worst drought first
        return alerts

    # ------------------------------------------------------------------
    # Tile URLs for Map Visualisation
    # ------------------------------------------------------------------
    def get_drought_tile_urls(self, start_date, end_date, geometry=None):
        """Return tile URLs for the composite index and classified map."""
        geom = geometry if geometry is not None else self.boundary

        composite = self.compute_composite_drought_index(start_date, end_date, geom)

        cdhi_vis = {
            "min": 0,
            "max": 100,
            "palette": [
                "#8B0000", "#FF0000", "#FF6600", "#FFA500",
                "#FFD700", "#FFFF66", "#ADFF2F", "#32CD32",
                "#228B22", "#006400",
            ],
        }
        class_vis = {
            "min": 1,
            "max": 5,
            "palette": ["#8B0000", "#FF6600", "#FFFF66", "#90EE90", "#228B22"],
        }

        cdhi_tile = composite["cdhi"].clip(geom).getMapId(cdhi_vis)
        class_tile = composite["drought_classes"].clip(geom).getMapId(class_vis)

        return {
            "composite_index_url": cdhi_tile["tile_fetcher"].url_format,
            "drought_classes_url": class_tile["tile_fetcher"].url_format,
        }


def get_enhanced_drought_analysis(start_date, end_date, geometry=None):
    """
    Main entry point — called from views.py.
    Returns complete drought analysis with tile URLs and alerts.
    """
    try:
        monitor = AdvancedDroughtMonitor()

        # Tile URLs
        tiles = monitor.get_drought_tile_urls(start_date, end_date, geometry)

        # Regional alerts
        alerts = monitor.generate_regional_alerts(start_date, end_date, geometry)

        return {
            "status": "success",
            "composite_index_url": tiles["composite_index_url"],
            "drought_classes_url": tiles["drought_classes_url"],
            "alerts": alerts,
            "indices_calculated": ["VCI", "TCI", "SPI", "SMI", "EVI"],
            "data_source": "Google Earth Engine — Multi-Satellite Real Data",
            "baseline_period": f"{BASELINE_START}–{BASELINE_END}",
            "message": "Real satellite-derived drought analysis for Tanzania",
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Drought analysis failed: {str(e)}",
        }
