"""
Phase 1 — Datasets 1.3 + 1.4
Harvested area × crop calendar preparation

Reads data IN PLACE — nothing is copied.
Outputs to outputs/fce/:
    harvested_area_by_country_crop.csv   (country × crop, hectares)
    crop_calendar_by_country_crop.csv    (country × crop, planting/harvest months)
    harvested_calendar_panel.csv         (country × crop panel: area km² + growing months)

5 crops: rice, wheat, maize, soybeans, oil_crops
ISIMIP3a yield anomalies moved to 03_prep_isimip_yield_anomalies.py

Install if needed:
    pip install xarray netCDF4 rioxarray rasterio geopandas pandas numpy tqdm
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths — all pointing to existing data, nothing copied
# ---------------------------------------------------------------------------
BASE = (
    "./"
    ".shortcut-targets-by-id/1-1FmczG81P-KFYKDBRJd2nYETWMtZ_GU/Crop Yields/data"
)

HARVESTED_DIR  = os.path.join(BASE, "harvested_area_grids")
CALENDAR_DIR   = os.path.join(BASE, "ALL_CROPS_netCDF_5min_filled")
ISIMIP_DIR     = os.path.join(BASE, "isimip3a")

OUT_DIR = (
    "./"
    "Other computers/My Laptop/UDel/Taky_research/flood_food_security/outputs/fce"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Crop mapping
# ---------------------------------------------------------------------------
# Harvested area files use IRC (irrigated) and RFC (rainfed) × crop number.
# Read readme to confirm numbering — printed below.
# Based on standard SPAM/Monfreda crop ordering:
CROP_MAP_HARVESTED = {
    "rice"     : {"irc": "CROP2",  "rfc": "CROP2"},
    "wheat"    : {"irc": "CROP1",  "rfc": "CROP1"},
    "maize"    : {"irc": "CROP3",  "rfc": "CROP3"},
    "soybeans" : {"irc": "CROP10", "rfc": "CROP10"},
    # Oil crops: sunflower (CROP15) + rapeseed (CROP16) + groundnut (CROP13)
    # Sum all three; each MapSPAM CROP number confirmed from SPAM2010 docs.
    "oil_crops": {"irc": ["CROP13", "CROP15", "CROP16"],
                  "rfc": ["CROP13", "CROP15", "CROP16"]},
}

# Crop calendar filenames (Sacks et al. 5-arcmin netCDF)
CALENDAR_MAP = {
    "rice"     : ["Rice.crop.calendar.fill.nc", "Rice.2.crop.calendar.fill.nc"],
    "wheat"    : ["Wheat.crop.calendar.fill.nc", "Wheat.Winter.crop.calendar.fill.nc"],
    "maize"    : ["Maize.crop.calendar.fill.nc"],
    "soybeans" : [],   # no calendar file — use default Apr–Oct
    # Oil crops: Sunflower absent on disk → use Rapeseed.Winter as proxy
    "oil_crops": ["Rapeseed.Winter.crop.calendar.fill.nc"],
}

# ---------------------------------------------------------------------------
# 1.7 Flood-adapted agriculture exclusion list
# ---------------------------------------------------------------------------
# These countries have significant flood-adapted or flood-dependent
# agricultural systems. Flooded pixels here should not be coded as damaged.
# Simple country-level exclusion — transparent and reviewer-auditable.
FLOOD_ADAPTED_COUNTRIES = {
    "Bangladesh": "deepwater rice (Boro/Aman flood-tolerant varieties)",
    "Mali"      : "Inner Niger Delta flood-recession farming",
    "Niger"     : "Inner Niger Delta flood-recession farming",
    "Vietnam"   : "Mekong Delta scheduled inundation rice",
    "Cambodia"  : "Tonle Sap flood-recession rice",
    "Myanmar"   : "Ayeyarwady Delta flood-tolerant rice (partial)",
}
print("Flood-adapted agriculture exclusions:")
for country, reason in FLOOD_ADAPTED_COUNTRIES.items():
    print(f"  {country}: {reason}")
print()

# ---------------------------------------------------------------------------
# Step 0 — Print readme so we can confirm crop numbering
# ---------------------------------------------------------------------------
readme_path = os.path.join(HARVESTED_DIR, "readme__harvested_area_grids.txt")
if os.path.exists(readme_path):
    print("=== Harvested area readme (crop numbering) ===")
    with open(readme_path) as f:
        print(f.read()[:2000])
    print("=== end readme ===\n")

# ---------------------------------------------------------------------------
# Step 1 — Load country boundaries
# ---------------------------------------------------------------------------
print("Loading country boundaries...")
import urllib.request, zipfile, io, pathlib

NE_DIR  = os.path.join(OUT_DIR.replace("outputs/fce", "data/country_shapes"))
NE_SHP  = os.path.join(NE_DIR, "ne_110m_admin_0_countries.shp")
os.makedirs(NE_DIR, exist_ok=True)

if not os.path.exists(NE_SHP):
    print("  Downloading Natural Earth country boundaries (~500KB)...")
    url = (
        "https://naciscdn.org/naturalearth/110m/cultural/"
        "ne_110m_admin_0_countries.zip"
    )
    with urllib.request.urlopen(url) as r:
        with zipfile.ZipFile(io.BytesIO(r.read())) as z:
            z.extractall(NE_DIR)
    print("  Downloaded.")

world = gpd.read_file(NE_SHP)
world = world[world.geometry.notna()].copy()
world = world.rename(columns={"NAME": "name", "ISO_A3": "iso3"})
world["name"] = world["name"].fillna(world.get("ADMIN", ""))
print(f"  {len(world)} countries.")

# ---------------------------------------------------------------------------
# Step 2 — Harvested area by country and crop
# ---------------------------------------------------------------------------
print("\nProcessing harvested area grids...")

def read_asc(path):
    """Read an ASC raster file, return (data_array, transform, nodata)."""
    with rasterio.open(path) as src:
        data      = src.read(1).astype(float)
        transform = src.transform
        nodata    = src.nodata
        crs       = src.crs
    if nodata is not None:
        data[data == nodata] = np.nan
    data[data < 0] = np.nan
    return data, transform, crs

def zonal_sum_asc(data, transform, crs, world_gdf):
    """
    Compute sum of raster values within each country polygon.
    Returns dict: country_name → sum.
    """
    from rasterio.transform import array_bounds
    from rasterio.features import rasterize
    from shapely.geometry import mapping

    h, w    = data.shape
    results = {}

    for _, row in world_gdf.iterrows():
        try:
            geom = [mapping(row.geometry)]
            mask = rasterio.features.geometry_mask(
                geom,
                transform = transform,
                invert    = True,
                out_shape = (h, w),
            )
            values = data[mask]
            values = values[~np.isnan(values)]
            results[row["name"]] = float(values.sum()) if len(values) > 0 else 0.0
        except Exception:
            results[row["name"]] = 0.0

    return results

harvested_records = []

for crop, codes in CROP_MAP_HARVESTED.items():
    # Normalise to lists so multi-crop groups work uniformly
    irc_codes = codes["irc"] if isinstance(codes["irc"], list) else [codes["irc"]]
    rfc_codes = codes["rfc"] if isinstance(codes["rfc"], list) else [codes["rfc"]]

    total_ha = None
    transform_out = None
    crs_out = None

    for code_list, irr_tag in [(irc_codes, "IRC"), (rfc_codes, "RFC")]:
        for code in code_list:
            fpath = os.path.join(
                HARVESTED_DIR,
                f"ANNUAL_AREA_HARVESTED_{irr_tag}_{code}_HA.ASC"
            )
            if not os.path.exists(fpath):
                continue
            arr, transform, crs = read_asc(fpath)
            arr = np.nan_to_num(arr, nan=0.0)
            if total_ha is None:
                total_ha = arr.copy()
                transform_out = transform
                crs_out = crs
            else:
                total_ha += arr

    if total_ha is None:
        print(f"  WARNING: No harvested area files found for {crop}")
        continue

    country_ha = zonal_sum_asc(total_ha, transform_out, crs_out, world)

    for country, ha in country_ha.items():
        excluded = country in FLOOD_ADAPTED_COUNTRIES
        harvested_records.append({
            "country"               : country,
            "crop"                  : crop,
            "harvested_ha"          : round(ha, 1),
            "harvested_km2"         : round(ha * 0.01, 4),   # 1 ha = 0.01 km²
            "flood_adapted_exclude" : excluded,
        })

    print(f"  {crop}: done")

harvested_df = pd.DataFrame(harvested_records)
harvested_df = harvested_df[harvested_df["harvested_ha"] > 0]

out_ha = os.path.join(OUT_DIR, "harvested_area_by_country_crop.csv")
harvested_df.to_csv(out_ha, index=False)
print(f"\nSaved: {out_ha}  ({len(harvested_df):,} rows)")

# ---------------------------------------------------------------------------
# Step 3 — Crop calendar: planting and harvest months by crop
# ---------------------------------------------------------------------------
print("\nProcessing crop calendars...")

def extract_calendar_country(nc_path, world_gdf):
    """
    Extract median planting month and harvest month per country
    from a crop calendar NetCDF file.
    Variables store day-of-year (1–365) — converted to month (1–12).
    """
    import math

    def doy_to_month(doy):
        """Convert day-of-year to calendar month (1–12)."""
        doy = np.clip(doy, 1, 365)
        return int(min(math.ceil(doy / 30.4375), 12))

    ds = xr.open_dataset(nc_path)

    # Detect variable names
    plant_var   = next((v for v in ds.data_vars if "plant" in v.lower()), None)
    harvest_var = next((v for v in ds.data_vars if "harv"  in v.lower()), None)

    if plant_var is None or harvest_var is None:
        print(f"    Could not find plant/harvest vars in {os.path.basename(nc_path)}")
        print(f"    Available vars: {list(ds.data_vars)}")
        ds.close()
        return {}

    plant_da   = ds[plant_var].squeeze()
    harvest_da = ds[harvest_var].squeeze()

    lats = plant_da.coords[
        next(c for c in plant_da.coords if "lat" in c.lower())
    ].values
    lons = plant_da.coords[
        next(c for c in plant_da.coords if "lon" in c.lower())
    ].values

    plant_arr   = plant_da.values.astype(float)
    harvest_arr = harvest_da.values.astype(float)
    plant_arr[plant_arr   <= 0] = np.nan
    harvest_arr[harvest_arr <= 0] = np.nan

    results = {}
    for _, row in world_gdf.iterrows():
        try:
            lon_mask = (lons >= row.geometry.bounds[0]) & (lons <= row.geometry.bounds[2])
            lat_mask = (lats >= row.geometry.bounds[1]) & (lats <= row.geometry.bounds[3])
            p_sub = plant_arr[np.ix_(lat_mask, lon_mask)]
            h_sub = harvest_arr[np.ix_(lat_mask, lon_mask)]
            p_vals = p_sub[~np.isnan(p_sub)]
            h_vals = h_sub[~np.isnan(h_sub)]
            if len(p_vals) > 0:
                results[row["name"]] = {
                    "plant_month"  : doy_to_month(np.nanmedian(p_vals)),
                    "harvest_month": doy_to_month(np.nanmedian(h_vals)),
                }
        except Exception:
            pass

    ds.close()
    return results

calendar_records = []

for crop, nc_files in CALENDAR_MAP.items():
    if not nc_files:
        # Default growing season: soybeans Apr–Oct, oil_crops May–Sep
        defaults = {
            "soybeans" : (4, 10),
            "oil_crops": (5,  9),
        }
        pm, hm = defaults.get(crop, (4, 10))
        for _, row in world.iterrows():
            calendar_records.append({
                "country"      : row["name"],
                "crop"         : crop,
                "plant_month"  : pm,
                "harvest_month": hm,
                "source"       : "default",
            })
        print(f"  {crop}: using default calendar ({pm}–{hm})")
        continue

    for nc_file in nc_files:
        nc_path = os.path.join(CALENDAR_DIR, nc_file)
        if not os.path.exists(nc_path):
            print(f"  WARNING: {nc_file} not found")
            continue

        country_cal = extract_calendar_country(nc_path, world)
        season_label = "season2" if "2" in nc_file else "season1"

        for country, months in country_cal.items():
            calendar_records.append({
                "country"      : country,
                "crop"         : crop,
                "plant_month"  : months["plant_month"],
                "harvest_month": months["harvest_month"],
                "source"       : os.path.basename(nc_file),
            })

    print(f"  {crop}: done")

calendar_df = pd.DataFrame(calendar_records)
out_cal = os.path.join(OUT_DIR, "crop_calendar_by_country_crop.csv")
calendar_df.to_csv(out_cal, index=False)
print(f"\nSaved: {out_cal}  ({len(calendar_df):,} rows)")

# ---------------------------------------------------------------------------
# Step 4 — Build merged harvested_calendar_panel.csv
# ---------------------------------------------------------------------------
# Country × crop panel: harvested area (ha and km²) + growing-season months.
# For crops with multiple seasons (rice, wheat), take the dominant season
# (season1 = max harvested area proxy → take season1 row when available).
print("\nBuilding merged harvested_calendar_panel.csv ...")

# Calendar: keep season1 preferentially (season2 is the secondary)
cal_primary = (
    calendar_df
    .sort_values("source")          # season1 sorts before season2 alphabetically
    .groupby(["country", "crop"])
    .first()
    .reset_index()
    [["country", "crop", "plant_month", "harvest_month", "source"]]
)

# Derive growing-season month list (plant → harvest, wrapping at year boundary)
def growing_months(pm, hm):
    if pm <= hm:
        return list(range(pm, hm + 1))
    else:                       # wraps across December
        return list(range(pm, 13)) + list(range(1, hm + 1))

cal_primary["growing_season_months"] = cal_primary.apply(
    lambda r: growing_months(int(r.plant_month), int(r.harvest_month)), axis=1
)
cal_primary["growing_season_length"] = cal_primary["growing_season_months"].apply(len)

# Merge area + calendar
panel = harvested_df[["country", "crop", "harvested_ha",
                       "harvested_km2", "flood_adapted_exclude"]].merge(
    cal_primary[["country", "crop", "plant_month", "harvest_month",
                 "growing_season_months", "growing_season_length", "source"]],
    on=["country", "crop"],
    how="left",
)
panel = panel[panel["harvested_km2"] > 0].copy()

out_panel = os.path.join(OUT_DIR, "harvested_calendar_panel.csv")
panel.to_csv(out_panel, index=False)
print(f"Saved: {out_panel}  ({len(panel):,} rows)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("Phase 1 prep (1.3+1.4) complete. Outputs in outputs/fce/:")
print(f"  harvested_area_by_country_crop.csv")
print(f"  crop_calendar_by_country_crop.csv")
print(f"  harvested_calendar_panel.csv  ← primary output")
print()
print("ISIMIP 3a yield anomalies: run 03_prep_isimip_yield_anomalies.py")
print("Next after GEE: run 03_compute_fce.py")
print("=" * 55)
