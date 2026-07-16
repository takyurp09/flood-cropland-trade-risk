"""
Phase 1 — Dataset 1.5
ISIMIP 3a multi-model ensemble crop yield anomalies

Reads NetCDF files IN PLACE from the existing Crop Yields data directory.
Aggregates gridded yields to country-year means, computes anomaly fraction
relative to 2000–2016 climatology, then takes ensemble mean across models
and irrigation types.

4 crops: rice, wheat, maize, soybeans (matches BACI commodity groups)
Output → outputs/fce/isimip_yield_anomaly_panel.csv

Usage:
    python 03_prep_isimip_yield_anomalies.py

Install if needed:
    pip install xarray netCDF4 numpy pandas geopandas tqdm
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = (
    "./"
    ".shortcut-targets-by-id/1-1FmczG81P-KFYKDBRJd2nYETWMtZ_GU/Crop Yields/data"
)
ISIMIP_DIR = os.path.join(BASE, "isimip3a")

PROJECT = (
    "./"
    "Other computers/My Laptop/UDel/Taky_research/flood_food_security"
)
OUT_DIR  = os.path.join(PROJECT, "outputs", "fce")
NE_SHP   = os.path.join(PROJECT, "data", "country_shapes",
                         "ne_110m_admin_0_countries.shp")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# ISIMIP 3a configuration
# ---------------------------------------------------------------------------
# Crop codes in ISIMIP filenames → canonical crop name
ISIMIP_CROP_MAP = {
    "rice"    : ["ri1", "ri2"],   # two growing seasons
    "wheat"   : ["swh", "wwh"],   # spring + winter wheat
    "maize"   : ["mai"],
    "soybeans": ["soy"],
}

# Models available in ISIMIP 3a agriculture sector
CROP_MODELS = ["epic-iiasa", "lpjml", "pdssat"]

# Irrigation variants
IRR_TYPES = ["firr", "noirr"]

# Forcing protocol string used in ISIMIP 3a filenames
FORCING_STR = "gswp3-w5e5_obsclim_2015soc_default"

YEARS = list(range(2000, 2017))   # ISIMIP3a historical period: 1901–2016

# ---------------------------------------------------------------------------
# Country boundaries
# ---------------------------------------------------------------------------
print("Loading country boundaries …")
world = gpd.read_file(NE_SHP)
world = world[world.geometry.notna()].copy()
world = world.rename(columns={"NAME": "name", "ISO_A3": "iso3"})
world["name"] = world["name"].fillna(world.get("ADMIN", ""))
print(f"  {len(world)} countries loaded.")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_isimip3a_file(fpath, years):
    """
    Load one ISIMIP 3a yield NetCDF and extract the requested years.

    ISIMIP3a time axis convention:  time values are growing-season indices
    where value N corresponds to year 1900 + N (i.e. value 100 → year 2000).

    Returns
    -------
    dict  year → np.ndarray (lat × lon), or empty dict if file unreadable.
    lats  np.ndarray
    lons  np.ndarray
    """
    try:
        ds  = xr.open_dataset(fpath, decode_times=False)
        var = list(ds.data_vars)[0]
        da  = ds[var]
        tv  = ds["time"].values.astype(float)

        lats = da.coords[next(c for c in da.coords if "lat" in c.lower())].values
        lons = da.coords[next(c for c in da.coords if "lon" in c.lower())].values

        year_map = {}
        for yr in years:
            target = float(yr - 1900)           # ISIMIP convention
            idx = np.where(tv == target)[0]
            if len(idx) == 0:
                # Fallback: 0-based index (some files use 0 = year 1901)
                idx2 = yr - 1901
                if 0 <= idx2 < da.shape[0]:
                    idx = [idx2]
            if len(idx) > 0:
                arr = da.isel(time=int(idx[0])).values.astype(float)
                arr[arr < 0] = np.nan
                year_map[yr] = arr

        ds.close()
        return year_map, lats, lons
    except Exception as e:
        print(f"    WARN: could not read {os.path.basename(fpath)}: {e}")
        return {}, None, None


def compute_anomaly(year_map):
    """
    Compute fractional yield anomaly vs. within-sample climatological mean.
    anomaly = (Y_t - Y_mean) / Y_mean
    Negative = below-normal yield.
    """
    if not year_map:
        return {}
    stack    = np.stack(list(year_map.values()), axis=0)   # (T, lat, lon)
    mean_yld = np.nanmean(stack, axis=0)
    out = {}
    for yr, arr in year_map.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            anom = np.where(mean_yld > 0, (arr - mean_yld) / mean_yld, np.nan)
        out[yr] = anom
    return out


def zonal_mean(arr, lats, lons, world_gdf):
    """
    Compute spatial mean of a gridded array within each country bounding box.
    Uses bounding-box approximation for speed (fine at 0.5° resolution).
    """
    results = {}
    for _, row in world_gdf.iterrows():
        try:
            b = row.geometry.bounds    # (minx, miny, maxx, maxy)
            lon_m = (lons >= b[0]) & (lons <= b[2])
            lat_m = (lats >= b[1]) & (lats <= b[3])
            sub   = arr[np.ix_(lat_m, lon_m)]
            vals  = sub[~np.isnan(sub)]
            results[row["name"]] = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
        except Exception:
            results[row["name"]] = np.nan
    return results


# ---------------------------------------------------------------------------
# Main loop: crawl all ISIMIP 3a files
# ---------------------------------------------------------------------------
print(f"\nScanning ISIMIP 3a directory: {ISIMIP_DIR}")

records = []

for crop, codes in ISIMIP_CROP_MAP.items():
    print(f"\nCrop: {crop}")

    for code in codes:
        for model in CROP_MODELS:
            for irr in IRR_TYPES:
                fname = (
                    f"{model}_{FORCING_STR}_"
                    f"yield-{code}-{irr}_global_annual-gs_1901_2016.nc"
                )
                fpath = os.path.join(ISIMIP_DIR, fname)

                if not os.path.exists(fpath):
                    continue

                print(f"  {model} / {code} / {irr} …", end=" ", flush=True)
                year_map, lats, lons = load_isimip3a_file(fpath, YEARS)

                if not year_map:
                    print("skipped.")
                    continue

                anomalies = compute_anomaly(year_map)

                for yr, anom_arr in tqdm(
                    anomalies.items(),
                    desc=f"    anomalies", leave=False, ncols=70
                ):
                    ctry_anom = zonal_mean(anom_arr, lats, lons, world)
                    for country, val in ctry_anom.items():
                        records.append({
                            "country"     : country,
                            "crop"        : crop,
                            "isimip_code" : code,
                            "model"       : model,
                            "irrigation"  : irr,
                            "year"        : yr,
                            "yield_anomaly": round(val, 5) if not np.isnan(val) else np.nan,
                        })

                print("done.")

if not records:
    print("\nWARNING: No ISIMIP 3a files matched. "
          "Check ISIMIP_DIR and filename pattern.")
else:
    raw_df = pd.DataFrame(records)

    # Ensemble mean: average over models, irrigation types, and season codes
    # (e.g. ri1 + ri2 → rice ensemble mean; swh + wwh → wheat ensemble mean)
    panel = (
        raw_df
        .groupby(["country", "crop", "year"])["yield_anomaly"]
        .mean()
        .reset_index()
        .rename(columns={"yield_anomaly": "yield_anomaly_ensemble_mean"})
    )

    out_path = os.path.join(OUT_DIR, "isimip_yield_anomaly_panel.csv")
    panel.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(panel):,} rows)")
    print(f"  Crops: {sorted(panel['crop'].unique())}")
    print(f"  Years: {int(panel['year'].min())}–{int(panel['year'].max())}")
    print(f"  Countries: {panel['country'].nunique()}")

print("\nDone. Run 04_fetch_dfo.py next.")
