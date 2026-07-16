"""
Phase 5 -- Dataset 5.2
GDPCIR ENSO Teleconnection Panel

Uses Planetary Computer CIL-GDPCIR (bias-corrected CMIP6).
Collection: cil-gdpcir-cc-by
Computes monthly regional climate means for ENSO analysis under SSP scenarios.

Output: outputs/projections/gdpcir_enso_teleconnection_panel.csv

Usage:
    python 02_fetch_esgf_gdpcir.py
"""

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

PROJECT = (
    "./"
    "Other computers/My Laptop/UDel/Taky_research/flood_food_security"
)
PROJ_OUT = os.path.join(PROJECT, "outputs", "projections")
OUT_PATH = os.path.join(PROJ_OUT, "gdpcir_enso_teleconnection_panel.csv")
os.makedirs(PROJ_OUT, exist_ok=True)

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

GCMS      = ["GFDL-ESM4", "MPI-ESM1-2-HR", "UKESM1-0-LL"]
SCENARIOS = ["ssp126", "ssp245", "ssp370", "ssp585"]
VARIABLES = ["pr", "tasmax", "tasmin"]

REGIONS = {
    "nino34"     : (-170, -120, -5,  5),  # lon0,lon1,lat0,lat1
    "south_asia" : (  60,  100,  5, 35),
    "ss_africa"  : (  10,   45,-10, 20),
    "mekong"     : (  99,  110, 10, 25),
}

PROJ_START_YEAR = 2020
PROJ_END_YEAR   = 2060

GCMS_MAP = {
    "GFDL-ESM4"    : "NOAA-GFDL",
    "MPI-ESM1-2-HR": "DKRZ",
    "UKESM1-0-LL"  : "MOHC",
}


def open_gdpcir_asset(item, asset_key, pc):
    signed = pc.sign(item.assets[asset_key])
    kwargs = signed.extra_fields.get("xarray:open_kwargs", {}) if signed.extra_fields else {}
    kwargs["chunks"] = {}
    ds = xr.open_dataset(signed.href, **kwargs)
    return ds


def regional_monthly(ds, var, lon0, lon1, lat0, lat1):
    da = ds[var]
    # Spatial subset
    lats = da.lat.values
    if lats[0] > lats[-1]:
        da = da.sel(lat=slice(lat1, lat0))
    else:
        da = da.sel(lat=slice(lat0, lat1))
    da = da.sel(lon=slice(lon0, lon1))
    # Temporal subset — use integer year comparison to avoid cftime mismatch
    years = da.time.dt.year
    da = da.isel(time=(years >= PROJ_START_YEAR) & (years <= PROJ_END_YEAR))
    # Monthly mean (resample works with cftime in xarray >=0.18)
    monthly = da.resample(time="MS").mean()
    spatial_mean = monthly.mean(dim=["lat", "lon"])
    return spatial_mean.compute()


import time

print("=== GDPCIR ENSO Teleconnection Projection ===")

try:
    import planetary_computer as pc
    import pystac_client
except ImportError as e:
    print(f"Missing dependencies: {e}")
    raise SystemExit(1)

def get_catalog():
    """Open a fresh Planetary Computer catalog (re-signs on each call)."""
    return pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)

def fetch_item(catalog, gcm, scenario):
    """Search for and return the first matching STAC item."""
    search = catalog.search(
        collections=["cil-gdpcir-cc-by"],
        query={
            "cmip6:source_id"    : {"in": [gcm]},
            "cmip6:experiment_id": {"in": [scenario]},
        },
    )
    items = list(search.item_collection())
    return items[0] if items else None

# ── Checkpoint: load already-completed combinations ───────────────────────────
if os.path.exists(OUT_PATH):
    try:
        existing = pd.read_csv(OUT_PATH)
        if len(existing) > 0:
            done_keys = set(
                zip(existing["gcm"], existing["scenario"], existing["variable"])
            )
            print(f"Resuming from checkpoint: {len(existing):,} rows already saved")
            print(f"  Completed combinations: {len(done_keys)}")
        else:
            existing = pd.DataFrame()
            done_keys = set()
    except Exception:
        existing = pd.DataFrame()
        done_keys = set()
else:
    existing = pd.DataFrame()
    done_keys = set()

all_new_records = []

catalog = get_catalog()
print("Planetary Computer catalog connected.")

for gcm in GCMS:
    for scenario in SCENARIOS:
        # Check if all variables for this gcm/scenario are done
        gcm_scen_done = all((gcm, scenario, v) in done_keys for v in VARIABLES)
        if gcm_scen_done:
            print(f"\n{gcm} / {scenario}  [SKIP — already in checkpoint]")
            continue

        print(f"\n{gcm} / {scenario}")
        # Fresh catalog fetch = fresh SAS token for each scenario
        try:
            catalog = get_catalog()
            item = fetch_item(catalog, gcm, scenario)
        except Exception as e:
            print(f"  Catalog/search error: {e}")
            continue

        if item is None:
            print(f"  No items found")
            continue

        print(f"  Item: {item.id}, assets: {list(item.assets.keys())}")
        scenario_records = []

        for var in VARIABLES:
            if (gcm, scenario, var) in done_keys:
                print(f"  {var}: [SKIP — already done]")
                continue
            if var not in item.assets:
                print(f"  {var}: not in assets — skip")
                continue

            MAX_RETRIES = 2
            for attempt in range(MAX_RETRIES):
                try:
                    ds = open_gdpcir_asset(item, var, pc)
                    for region_name, (lon0, lon1, lat0, lat1) in REGIONS.items():
                        vals = regional_monthly(ds, var, lon0, lon1, lat0, lat1)
                        for t, v in zip(vals.time.values, vals.values):
                            try:
                                yr = int(t.year)
                                mo = int(t.month)
                            except AttributeError:
                                ts = pd.Timestamp(t)
                                yr = ts.year
                                mo = ts.month
                            scenario_records.append({
                                "gcm"     : gcm,
                                "scenario": scenario,
                                "variable": var,
                                "region"  : region_name,
                                "year"    : yr,
                                "month"   : mo,
                                "value"   : float(v) if not np.isnan(float(v)) else None,
                            })
                    ds.close()
                    print(f"  {var}: done")
                    break  # success — exit retry loop
                except Exception as e:
                    err_str = str(e)
                    if "AuthenticationFailed" in err_str or "Signature not valid" in err_str:
                        print(f"  {var}: SAS token expired on attempt {attempt+1} — refreshing catalog")
                        # Re-sign: get fresh catalog + item
                        time.sleep(2)
                        try:
                            catalog = get_catalog()
                            item = fetch_item(catalog, gcm, scenario)
                            if item is None:
                                print(f"  {var}: item not found after refresh — skip")
                                break
                        except Exception as e2:
                            print(f"  {var}: catalog refresh failed: {e2} — skip")
                            break
                        if attempt == MAX_RETRIES - 1:
                            print(f"  {var}: still failing after {MAX_RETRIES} attempts — skip")
                    else:
                        print(f"  {var}: ERROR - {e}")
                        break

        # Checkpoint: append new scenario records to file immediately
        if scenario_records:
            all_new_records.extend(scenario_records)
            new_df = pd.DataFrame(scenario_records)
            if os.path.exists(OUT_PATH) and os.path.getsize(OUT_PATH) > 10:
                new_df.to_csv(OUT_PATH, mode="a", header=False, index=False)
            else:
                new_df.to_csv(OUT_PATH, index=False)
            print(f"  Checkpoint saved: {len(scenario_records):,} new rows ({gcm}/{scenario})")
            # Update done_keys with newly completed combos
            for rec in scenario_records:
                done_keys.add((rec["gcm"], rec["scenario"], rec["variable"]))

# ── Final summary ─────────────────────────────────────────────────────────────
if os.path.exists(OUT_PATH) and os.path.getsize(OUT_PATH) > 10:
    final = pd.read_csv(OUT_PATH)
    print(f"\nFinal output: {OUT_PATH}  ({len(final):,} rows)")
    print(f"GCMs: {final['gcm'].unique()}")
    print(f"Scenarios: {final['scenario'].unique()}")
    print(f"Years: {final['year'].min()}-{final['year'].max()}")
else:
    print("WARNING: No data collected. Writing scaffold.")
    pd.DataFrame(columns=["gcm","scenario","variable","region","year","month","value"]).to_csv(OUT_PATH, index=False)
