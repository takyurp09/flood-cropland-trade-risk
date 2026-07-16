"""
Phase 5 — ISIMIP 3b qtot processing

Aggregates monthly qtot NetCDF files to country-level annual panels using
regionmask for spatial aggregation.

Unit conversion: qtot (kg m-2 s-1) × 86400 × 365 → mm/year
Anomaly baseline: 2015–2034 mean (early-future reference window)

Input:  data/raw/isimip3b/qtot/*.nc
Output: outputs/projections/isimip3b_qtot_country_panel.csv

Columns: country_iso3, year, scenario, model, gcm, qtot_mm_yr, qtot_anom_mm

Usage:
    python process_isimip3b_qtot.py
"""

import re
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import country_converter as coco
import regionmask
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT  = Path(__file__).resolve().parents[2]
IN_DIR   = PROJECT / "data" / "raw" / "qtot"
PROJ_OUT = PROJECT / "outputs" / "projections"
OUT_PATH = PROJ_OUT / "isimip3b_qtot_country_panel.csv"
PROJ_OUT.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BASELINE_YEARS   = (2015, 2034)     # anomaly reference window
KG_M2_S_TO_MM_YR = 86_400 * 365    # kg m-2 s-1 → mm year-1 (non-leap approx)

# ── Country lookup: regionmask region.number → ISO3 ──────────────────────────
_RM_COUNTRIES  = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
_CC             = coco.CountryConverter()
_NUM_TO_ISO3   = {}
for _r in _RM_COUNTRIES.regions.values():
    _iso = _CC.convert(_r.name, to="ISO3")
    if _iso not in ("not found", ""):
        _NUM_TO_ISO3[_r.number] = str(_iso)

# Parse model/gcm/scenario from filename
# e.g. cwatm_gfdl-esm4_w5e5_ssp126_2015soc-from-histsoc_default_qtot_global_monthly[_YYYY_YYYY].nc
FNAME_RE = re.compile(
    r"^(?P<model>[^_]+)_(?P<gcm>[^_]+(?:-[^_]+)*)_w5e5_"
    r"(?P<scenario>ssp\d+)_.*_qtot_global_monthly",
    re.IGNORECASE,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_filename(fname: str) -> Optional[dict]:
    m = FNAME_RE.match(fname)
    if not m:
        return None
    return {
        "model":    m.group("model").lower(),
        "gcm":      m.group("gcm").lower(),
        "scenario": m.group("scenario").lower(),
    }


def open_dataset_safe(nc_path: Path) -> Optional[xr.Dataset]:
    """Open NetCDF with CF time decoding; fall back to raw numeric time."""
    for kwargs in ({"use_cftime": True}, {"decode_times": False}):
        try:
            ds = xr.open_dataset(nc_path, **kwargs)
            return ds
        except Exception:
            continue
    print(f"  Cannot open {nc_path.name}")
    return None


def extract_years(ds: xr.Dataset) -> np.ndarray:
    """Return integer year array from the 'time' coordinate."""
    tc = ds["time"]
    try:
        return tc.dt.year.values.astype(int)
    except AttributeError:
        # decode_times=False path: numeric offsets
        units = tc.attrs.get("units", "")
        m = re.search(r"since (\d{4})", units)
        ref = int(m.group(1)) if m else 1850
        if units.startswith("months"):
            return (ref + tc.values // 12).astype(int)
        # default: days
        return (ref + tc.values / 365.25).astype(int)


def country_annual_mean(nc_path: Path) -> Optional[pd.DataFrame]:
    """
    Spatial-mean qtot per country per year from one NetCDF file.

    Returns DataFrame with columns: iso3, year, qtot_mm_yr
    Returns None if the file cannot be processed.
    """
    ds = open_dataset_safe(nc_path)
    if ds is None:
        return None

    # ── Identify qtot variable ────────────────────────────────────────────────
    var = next(
        (v for v in ds.data_vars if v.lower() in ("qtot", "dis", "runoff", "ro")),
        None,
    )
    if var is None:
        var = next(iter(ds.data_vars))

    da = ds[var].astype(np.float32)

    # ── Identify and standardise lat/lon dimension names ──────────────────────
    lat_dim = next((d for d in da.dims if "lat" in d.lower()), None)
    lon_dim = next((d for d in da.dims if "lon" in d.lower()), None)
    if lat_dim is None or lon_dim is None:
        print(f"  Cannot identify lat/lon dims in {nc_path.name}: {da.dims}")
        ds.close()
        return None

    rename = {}
    if lat_dim != "lat":
        rename[lat_dim] = "lat"
    if lon_dim != "lon":
        rename[lon_dim] = "lon"
    if rename:
        da = da.rename(rename)

    # ── Replace fill values ───────────────────────────────────────────────────
    da = da.where(da > -1e10)

    # ── Build country mask (works for any resolution ≥ 0.25°) ────────────────
    mask_da = _RM_COUNTRIES.mask(da["lon"].values, da["lat"].values)   # DataArray (lat, lon)
    mask    = mask_da.values   # numpy (lat, lon), int; -1 = no region

    years = extract_years(ds)
    unique_years = np.unique(years)

    records = []
    for yr in unique_years:
        time_idx = np.where(years == yr)[0]
        # Annual mean across months; shape (lat, lon)
        yr_arr = da.isel({"time": time_idx}).mean("time").values
        # kg m-2 s-1 → mm year-1
        yr_arr = yr_arr * KG_M2_S_TO_MM_YR

        for reg_num, iso3 in _NUM_TO_ISO3.items():
            region_px = mask == reg_num
            vals = yr_arr[region_px]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            records.append({
                "iso3":       iso3,
                "year":       int(yr),
                "qtot_mm_yr": float(np.mean(vals)),
            })

    ds.close()
    return pd.DataFrame(records) if records else None


# ── Scan input directory ───────────────────────────────────────────────────────
nc_files = sorted(IN_DIR.rglob("*.nc"))
print(f"Input directory : {IN_DIR}")
print(f"NetCDF files    : {len(nc_files)}")

if not nc_files:
    raise SystemExit(
        f"No NetCDF files found in {IN_DIR}.\n"
        "Run scripts/phase5_projections/download_isimip3b_qtot.py first."
    )

# ── Process each file ─────────────────────────────────────────────────────────
all_records = []
skipped     = []

for nc_path in tqdm(nc_files, desc="Processing"):
    meta = parse_filename(nc_path.name)
    if meta is None:
        print(f"  Skipping unrecognised filename: {nc_path.name}")
        skipped.append(nc_path.name)
        continue

    df = country_annual_mean(nc_path)
    if df is None or df.empty:
        print(f"  No data extracted: {nc_path.name}")
        skipped.append(nc_path.name)
        continue

    df["model"]    = meta["model"]
    df["gcm"]      = meta["gcm"]
    df["scenario"] = meta["scenario"]
    all_records.append(df)

if not all_records:
    raise SystemExit("No data extracted from any file. Check input files.")

if skipped:
    print(f"\nSkipped {len(skipped)} file(s): {skipped}")

panel = pd.concat(all_records, ignore_index=True)
print(f"\nRaw panel : {len(panel):,} rows | "
      f"{panel['iso3'].nunique()} countries | "
      f"years {panel['year'].min()}–{panel['year'].max()}")

# ── Anomaly vs 2015–2034 baseline ─────────────────────────────────────────────
print(f"Computing anomaly vs {BASELINE_YEARS[0]}–{BASELINE_YEARS[1]} baseline …")

baseline = (
    panel[panel["year"].between(*BASELINE_YEARS)]
    .groupby(["iso3", "model", "gcm", "scenario"])["qtot_mm_yr"]
    .mean()
    .rename("qtot_base_mm")
    .reset_index()
)

panel = panel.merge(baseline, on=["iso3", "model", "gcm", "scenario"], how="left")
panel["qtot_anom_mm"] = panel["qtot_mm_yr"] - panel["qtot_base_mm"]
panel = panel.drop(columns=["qtot_base_mm"])

# ── Rename iso3 → country_iso3 for consistency with other panels ──────────────
panel = panel.rename(columns={"iso3": "country_iso3"})

# ── Save ───────────────────────────────────────────────────────────────────────
col_order = ["country_iso3", "year", "scenario", "model", "gcm",
             "qtot_mm_yr", "qtot_anom_mm"]
panel[col_order].to_csv(OUT_PATH, index=False)

n_null_anom = panel["qtot_anom_mm"].isna().sum()
print(f"\nSaved → {OUT_PATH}")
print(f"  Rows       : {len(panel):,}")
print(f"  Countries  : {panel['country_iso3'].nunique()}")
print(f"  Scenarios  : {sorted(panel['scenario'].unique())}")
print(f"  Model×GCM  : {panel[['model','gcm']].drop_duplicates().shape[0]} combinations")
print(f"  qtot_anom null (no baseline coverage) : {n_null_anom:,} "
      f"({100 * n_null_anom / len(panel):.1f}%)")
