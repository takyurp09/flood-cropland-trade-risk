"""
Phase 3 -- Dataset 3.3
ERA5 Country-Level Growing-Season Climate Anomaly

Downloads ERA5 monthly means (2m_temperature, total_precipitation) via CDS API
for 2000-2021 globally. Computes country-year growing-season averages aligned
to crop calendar, then anomalies relative to 2000-2020 baseline.

Output: outputs/outcome/era5_climate_panel.csv

Usage:
    python 03_fetch_era5_climate.py

Requires ~/.cdsapirc with valid CDS API key.
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import cdsapi

PROJECT = (
    "./"
    "Other computers/My Laptop/UDel/Taky_research/flood_food_security"
)
RAW_ERA5 = os.path.join(PROJECT, "data", "raw", "era5")
OUT_DIR  = os.path.join(PROJECT, "outputs", "outcome")
OUT_PATH = os.path.join(OUT_DIR, "era5_climate_panel.csv")
CALENDAR = os.path.join(PROJECT, "outputs", "fce", "crop_calendar_by_country_crop.csv")
NE_SHP   = os.path.join(PROJECT, "data", "country_shapes", "ne_110m_admin_0_countries.shp")

os.makedirs(RAW_ERA5, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

YEARS      = list(range(2000, 2022))
CLIM_START = 2000
CLIM_END   = 2020

# ─── CDS download ────────────────────────────────────────────────────────────

def cds_download(variable, short_name, out_path):
    """Download global monthly means for one ERA5 variable, all years 2000-2021."""
    if os.path.exists(out_path):
        print(f"  {short_name}: cached at {out_path}")
        return out_path

    print(f"  Requesting {variable} from CDS (global, 2000-2021)...")
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type' : 'monthly_averaged_reanalysis',
            'variable'     : variable,
            'year'         : [str(y) for y in YEARS],
            'month'        : [f"{m:02d}" for m in range(1, 13)],
            'time'         : '00:00',
            'format'       : 'netcdf',
        },
        out_path
    )
    print(f"  Saved: {out_path}")
    return out_path


# ─── Country mask ────────────────────────────────────────────────────────────

def build_country_monthly_means(ds_t2m, ds_tp):
    """
    For each country × year × month, compute spatial mean T2M (K) and TP (m).
    Returns DataFrame with columns: country, iso3, year, month, t2m_K, tp_m
    """
    world = gpd.read_file(NE_SHP)
    world = world[world.geometry.notna()].copy()
    world = world.rename(columns={"NAME": "cname", "ISO_A3": "iso3"})

    # Align time coordinates
    t2m_monthly = ds_t2m['t2m']   # (time, lat, lon)
    tp_monthly  = ds_tp['tp']     # (time, lat, lon)

    lats = t2m_monthly.latitude.values
    lons = t2m_monthly.longitude.values

    records = []
    for _, row in world.iterrows():
        geom = row.geometry
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        # Clip with small buffer
        lat_slice = slice(min(bounds[3]+0.5, lats.max()), max(bounds[1]-0.5, lats.min()))
        lon_slice = slice(max(bounds[0]-0.5, lons.min()), min(bounds[2]+0.5, lons.max()))

        sub_t = t2m_monthly.sel(latitude=lat_slice, longitude=lon_slice)
        sub_p = tp_monthly.sel(latitude=lat_slice, longitude=lon_slice)

        if sub_t.latitude.size == 0 or sub_t.longitude.size == 0:
            # Tiny island or edge case — skip spatial filter
            sub_t = t2m_monthly
            sub_p = tp_monthly

        # Compute cos(lat) weights  
        lat_rad = np.deg2rad(sub_t.latitude.values)
        weights = np.cos(lat_rad)
        w2d = xr.DataArray(weights[:, None] * np.ones(sub_t.longitude.size)[None, :],
                           dims=['latitude', 'longitude'])
        t_mean = (sub_t * w2d).sum(dim=['latitude','longitude']) / w2d.values.sum()
        p_mean = (sub_p * w2d).sum(dim=['latitude','longitude']) / w2d.values.sum()

        for i, t in enumerate(t_mean.time.values):
            ts = pd.Timestamp(str(t))
            records.append({
                'country': row['cname'],
                'iso3'   : row['iso3'],
                'year'   : ts.year,
                'month'  : ts.month,
                't2m_K'  : float(t_mean.values[i]),
                'tp_m'   : float(p_mean.values[i]),
            })

    return pd.DataFrame(records)


# ─── Apply crop calendar ─────────────────────────────────────────────────────

def growing_season_mean(df_monthly, cal_df):
    """
    For each country × year, average monthly t2m and tp over the dominant
    growing season from the crop calendar. Default: Apr-Sep if no calendar.
    """
    # One representative season per country (rice>wheat>maize priority)
    priority = {'rice': 0, 'wheat': 1, 'maize': 2, 'soybeans': 3, 'oil_crops': 4}
    if cal_df is not None:
        country_season = (
            cal_df
            .assign(pri=cal_df['crop'].map(priority).fillna(9))
            .sort_values(['country', 'pri'])
            .groupby('country')
            .first()
            .reset_index()
            [['country', 'plant_month', 'harvest_month']]
        )
    else:
        country_season = pd.DataFrame(columns=['country', 'plant_month', 'harvest_month'])

    records = []
    for (country, year), grp in df_monthly.groupby(['country', 'year']):
        cs = country_season[country_season['country'] == country]
        if cs.empty:
            pm, hm = 4, 9
        else:
            pm = int(cs['plant_month'].iloc[0])
            hm = int(cs['harvest_month'].iloc[0])

        if pm <= hm:
            months = list(range(pm, hm + 1))
        else:
            months = list(range(pm, 13)) + list(range(1, hm + 1))

        season = grp[grp['month'].isin(months)]
        if season.empty:
            season = grp   # fallback: all months

        records.append({
            'country'      : country,
            'iso3'         : grp['iso3'].iloc[0],
            'year'         : year,
            'plant_month'  : pm,
            'harvest_month': hm,
            't2m_mean_K'   : season['t2m_K'].mean(),
            'tp_sum_m'     : season['tp_m'].sum(),
        })
    return pd.DataFrame(records)


# ─── Anomaly ─────────────────────────────────────────────────────────────────

def compute_anomalies(panel):
    clim = (
        panel[panel['year'].between(CLIM_START, CLIM_END)]
        .groupby('country')[['t2m_mean_K', 'tp_sum_m']]
        .mean()
        .rename(columns={'t2m_mean_K': 't2m_clim', 'tp_sum_m': 'tp_clim'})
    )
    out = panel.merge(clim.reset_index(), on='country', how='left')
    out['t2m_anom_C']   = out['t2m_mean_K'] - out['t2m_clim']
    out['tp_anom_frac'] = (out['tp_sum_m'] - out['tp_clim']) / out['tp_clim'].abs()
    return out


# ─── Main ────────────────────────────────────────────────────────────────────

print("=== ERA5 country-level growing-season climate anomaly ===\n")

# 1. Download
t2m_path = os.path.join(RAW_ERA5, "era5_t2m_monthly_2000_2021.nc")
tp_path  = os.path.join(RAW_ERA5, "era5_tp_monthly_2000_2021.nc")

print("Downloading ERA5 monthly means via CDS API...")
cds_download('2m_temperature', 't2m', t2m_path)
cds_download('total_precipitation', 'tp', tp_path)

# 2. Load
print("\nLoading NetCDF files...")
ds_t2m = xr.open_dataset(t2m_path)
ds_tp  = xr.open_dataset(tp_path)
# Filter to study years
ds_t2m = ds_t2m.sel(valid_time=ds_t2m['valid_time'].dt.year.isin(YEARS) if 'valid_time' in ds_t2m else ...)
ds_tp  = ds_tp.sel(valid_time=ds_tp['valid_time'].dt.year.isin(YEARS) if 'valid_time' in ds_tp else ...)
# Handle time dimension name (may be 'time' or 'valid_time')
for attr in ('time', 'valid_time'):
    if attr in ds_t2m.dims and attr != 'time':
        ds_t2m = ds_t2m.rename({attr: 'time'})
    if attr in ds_tp.dims and attr != 'time':
        ds_tp  = ds_tp.rename({attr: 'time'})

# Also handle latitude/longitude naming
def norm_coords(ds):
    rn = {}
    for c in ds.coords:
        if c.lower() in ('lat', 'latitude') and c != 'latitude':
            rn[c] = 'latitude'
        if c.lower() in ('lon', 'longitude') and c != 'longitude':
            rn[c] = 'longitude'
    return ds.rename(rn) if rn else ds

ds_t2m = norm_coords(ds_t2m)
ds_tp  = norm_coords(ds_tp)

# Variable name normalization
if 't2m' not in ds_t2m:
    vname = [v for v in ds_t2m.data_vars][0]
    ds_t2m = ds_t2m.rename({vname: 't2m'})
if 'tp' not in ds_tp:
    vname = [v for v in ds_tp.data_vars][0]
    ds_tp = ds_tp.rename({vname: 'tp'})

print(f"T2M dataset: {ds_t2m.dims}, time: {ds_t2m.time.values[0]} to {ds_t2m.time.values[-1]}")
print(f"TP  dataset: {ds_tp.dims}")

# 3. Load crop calendar
print("\nLoading crop calendar...")
cal_df = pd.read_csv(CALENDAR) if os.path.exists(CALENDAR) else None
if cal_df is not None:
    print(f"  {len(cal_df)} rows, {cal_df['country'].nunique()} countries")

# 4. Country monthly means
print("\nComputing country spatial means (may take ~5 minutes)...")
df_monthly = build_country_monthly_means(ds_t2m, ds_tp)
print(f"  Monthly means: {len(df_monthly)} rows")

# 5. Growing season aggregation
print("Applying crop calendar filter...")
panel = growing_season_mean(df_monthly, cal_df)

# 6. Anomalies
print("Computing anomalies...")
panel = compute_anomalies(panel)

# 7. Save
cols = ['country', 'iso3', 'year', 'plant_month', 'harvest_month',
        't2m_mean_K', 'tp_sum_m', 't2m_anom_C', 'tp_anom_frac']
panel[cols].sort_values(['country', 'year']).to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}  ({len(panel):,} rows)")
print(f"Countries: {panel['country'].nunique()}, Years: {YEARS[0]}-{YEARS[-1]}")
print(f"T2M anomaly range: {panel['t2m_anom_C'].min():.2f} to {panel['t2m_anom_C'].max():.2f} C")
print(f"TP anomaly range: {panel['tp_anom_frac'].min():.2f} to {panel['tp_anom_frac'].max():.2f}")

ds_t2m.close()
ds_tp.close()
