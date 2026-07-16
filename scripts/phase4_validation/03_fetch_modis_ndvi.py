"""
Phase 4 — Dataset 4.4
MODIS NDVI/EVI Time Series (Crop Damage Validation)

Extracts NDVI time series for flooded cropland pixels at 10 validation events.
Confirms V-shaped dip-and-recovery pattern expected for crop damage.

Uses Google Earth Engine Python API (ee) for MODIS MOD13A1 (500m, 16-day).

Output → data/processed/modis_ndvi_validation.csv

Usage:
    python 03_fetch_modis_ndvi.py
    (requires authenticated GEE: run `earthengine authenticate` first)

Install if needed:
    pip install earthengine-api pandas geopandas
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = (
    "./"
    "Other computers/My Laptop/UDel/Taky_research/flood_food_security"
)
PROC_DIR = os.path.join(PROJECT, "outputs", "validation")
OUT_PATH = os.path.join(PROC_DIR, "modis_ndvi_validation.csv")
os.makedirs(PROC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Validation events (same 10 events as 4.2/4.3)
# Centroid of major flood-affected agricultural region per event.
# ---------------------------------------------------------------------------
VALIDATION_EVENTS = [
    # (label, year, month_flood_peak, lon_center, lat_center, radius_km)
    {"label": "Pakistan_2010",    "year": 2010, "peak_month": 8,
     "lon": 68.4,  "lat": 28.9,  "radius_km": 200, "crop": "wheat"},
    {"label": "Thailand_2011",    "year": 2011, "peak_month": 10,
     "lon": 100.6, "lat": 14.8,  "radius_km": 150, "crop": "rice"},
    {"label": "Nigeria_2012",     "year": 2012, "peak_month": 9,
     "lon": 6.4,   "lat": 6.5,   "radius_km": 200, "crop": "maize"},
    {"label": "Pakistan_2015",    "year": 2015, "peak_month": 9,
     "lon": 71.8,  "lat": 32.2,  "radius_km": 150, "crop": "wheat"},
    {"label": "Bangladesh_2017",  "year": 2017, "peak_month": 6,
     "lon": 90.8,  "lat": 24.5,  "radius_km": 150, "crop": "rice"},
    {"label": "India_2019",       "year": 2019, "peak_month": 7,
     "lon": 75.5,  "lat": 20.9,  "radius_km": 200, "crop": "wheat"},
    {"label": "India_2020",       "year": 2020, "peak_month": 8,
     "lon": 85.0,  "lat": 24.0,  "radius_km": 200, "crop": "rice"},
    {"label": "China_2016",       "year": 2016, "peak_month": 7,
     "lon": 115.8, "lat": 30.5,  "radius_km": 200, "crop": "maize"},
    {"label": "Indonesia_2013",   "year": 2013, "peak_month": 2,
     "lon": 107.0, "lat": -7.0,  "radius_km": 100, "crop": "rice"},
    {"label": "Myanmar_2015",     "year": 2015, "peak_month": 8,
     "lon": 95.0,  "lat": 19.5,  "radius_km": 150, "crop": "rice"},
]

# Window: 6 months before to 6 months after flood peak (for V-shape)
PRE_MONTHS  = 6
POST_MONTHS = 6

# MODIS product in GEE
MODIS_PRODUCT = "MODIS/061/MOD13A1"
NDVI_BAND     = "NDVI"
EVI_BAND      = "EVI"
SCALE_FACTOR  = 0.0001    # MODIS NDVI scale factor

# ---------------------------------------------------------------------------

def init_gee():
    """Initialise Google Earth Engine. Returns True if successful."""
    try:
        import ee
        ee.Initialize(project='t2m-precip-monthly-country')
        print("  GEE initialised successfully.")
        return True
    except Exception as e:
        print(f"  GEE initialisation failed: {e}")
        print("  Run: earthengine authenticate")
        return False


def extract_ndvi_timeseries(event):
    """
    Extract MODIS NDVI/EVI monthly means for a circular AOI around event centroid.
    Returns list of dicts with date, mean_ndvi, mean_evi, n_pixels.
    """
    import ee
    from datetime import date, timedelta
    import calendar

    # Date range
    yr   = event["year"]
    pm   = event["peak_month"]

    # Start 6 months before peak, end 6 months after
    start = date(yr - 1 if pm - PRE_MONTHS <= 0 else yr,
                 (pm - PRE_MONTHS - 1) % 12 + 1, 1)
    end_yr = yr + 1 if pm + POST_MONTHS > 12 else yr
    end_m  = (pm + POST_MONTHS - 1) % 12 + 1
    end    = date(end_yr, end_m,
                  calendar.monthrange(end_yr, end_m)[1])

    # AOI
    point  = ee.Geometry.Point([event["lon"], event["lat"]])
    aoi    = point.buffer(event["radius_km"] * 1000)

    # MODIS collection
    col = (
        ee.ImageCollection(MODIS_PRODUCT)
        .filterBounds(aoi)
        .filterDate(str(start), str(end))
        .select([NDVI_BAND, EVI_BAND])
    )

    def monthly_mean(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.count(), sharedInputs=True
            ),
            geometry=aoi,
            scale=500,
            maxPixels=1e8,
        )
        return img.set({
            "date_str"  : img.date().format("YYYY-MM-dd"),
            "mean_ndvi" : stats.get(NDVI_BAND + "_mean"),
            "mean_evi"  : stats.get(EVI_BAND + "_mean"),
            "n_pixels"  : stats.get(NDVI_BAND + "_count"),
        })

    col_with_stats = col.map(monthly_mean)

    # Parse results
    records = []
    try:
        dates  = col_with_stats.aggregate_array("date_str").getInfo()
        ndvis  = col_with_stats.aggregate_array("mean_ndvi").getInfo()
        evis   = col_with_stats.aggregate_array("mean_evi").getInfo()
        counts = col_with_stats.aggregate_array("n_pixels").getInfo()

        for d, n, e, c in zip(dates, ndvis, evis, counts):
            records.append({
                "label"    : event["label"],
                "year"     : event["year"],
                "crop"     : event["crop"],
                "peak_month": event["peak_month"],
                "date"     : d,
                "mean_ndvi": round(float(n) * SCALE_FACTOR, 4) if n else np.nan,
                "mean_evi" : round(float(e) * SCALE_FACTOR, 4) if e else np.nan,
                "n_pixels" : int(c) if c else 0,
            })
    except Exception as ex:
        print(f"    Parse error for {event['label']}: {ex}")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=== MODIS NDVI/EVI Validation Extraction (GEE) ===\n")

gee_ok = init_gee()

all_records = []

if gee_ok:
    for event in VALIDATION_EVENTS:
        print(f"  Processing: {event['label']} …", end=" ", flush=True)
        try:
            recs = extract_ndvi_timeseries(event)
            all_records.extend(recs)
            print(f"{len(recs)} observations.")
        except Exception as e:
            print(f"ERROR: {e}")

    if all_records:
        df = pd.DataFrame(all_records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["label", "date"])

        # Compute months relative to flood peak (negative = before, positive = after)
        df["month_relative"] = (
            (df["date"].dt.year - df["year"]) * 12 +
            (df["date"].dt.month - df["peak_month"])
        )

        df.to_csv(OUT_PATH, index=False)
        print(f"\nSaved: {OUT_PATH}  ({len(df):,} rows)")
        print("Events covered:", df["label"].unique().tolist())
    else:
        print("\nWARNING: No NDVI data extracted. Check GEE authentication.")
else:
    print(
        "\nGEE not available. Manual steps:\n"
        "  1. earthengine authenticate\n"
        "  2. Re-run this script\n"
        "  OR: use GEE Code Editor with MOD13A1 for each event centroid.\n"
        f"  Output target: {OUT_PATH}"
    )

print("\nTier 4 complete. Run Tier 5 projection scripts next.")
