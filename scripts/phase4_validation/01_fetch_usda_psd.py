"""
Phase 4 — Dataset 4.1
USDA Production, Supply and Distribution (PSD) Database

Downloads USDA PSD bulk data for key commodities.
Extracts production, imports, exports for validation of FCE estimates.

Output → data/raw/usda_psd_raw.csv
          data/processed/usda_psd_panel.csv

Usage:
    python 01_fetch_usda_psd.py

Install if needed:
    pip install requests pandas
"""

import os
import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = (
    "./"
    "Other computers/My Laptop/UDel/Taky_research/flood_food_security"
)
RAW_DIR   = os.path.join(PROJECT, "data", "raw")
OUT_DIR   = os.path.join(PROJECT, "outputs", "validation")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

PSD_RAW   = os.path.join(RAW_DIR, "usda_psd_raw.csv")
OUT_PATH  = os.path.join(OUT_DIR, "usda_psd_panel.csv")

# USDA PSD bulk download URL (updated annually by USDA FAS)
PSD_URL = (
    "https://apps.fas.usda.gov/psdonline/downloads/psd_alldata.csv"
)

HEADERS = {"User-Agent": "research-replication-bot/1.0 (contact: your-email@example.com)"}

# PSD commodity codes for our 5 crop groups
# (USDA uses commodity names, not HS codes)
COMMODITY_FILTER = [
    "Rice, Milled", "Rice, Paddy (Rough)", "Rice",
    "Wheat", "Wheat and Products",
    "Corn",   # USDA uses "Corn" for maize
    "Soybeans", "Soybean Oil", "Soybean Meal",
    "Palm Oil", "Rapeseed Oil", "Sunflowerseed Oil",
    "Cotton", # exclude — keep only food
]
COMMODITY_KEEP = [c for c in COMMODITY_FILTER if c != "Cotton"]

# Attributes (PSD column: Attribute_Description)
ATTRS_KEEP = [
    "Production", "Imports", "Exports",
    "Beginning Stocks", "Ending Stocks",
    "Dom. Consumption", "Total Supply",
]

YEARS = list(range(2000, 2023))

# ---------------------------------------------------------------------------

def download_psd():
    # Updated URL: USDA FAS now serves zip instead of flat CSV
    urls = [
        "https://apps.fas.usda.gov/psdonline/downloads/psd_alldata_csv.zip",
        PSD_URL,  # original as fallback
    ]
    import io, zipfile
    for url in urls:
        try:
            print(f"  Trying: {url[-60:]} …")
            r = requests.get(url, headers=HEADERS, timeout=300, stream=True)
            r.raise_for_status()
            if url.endswith(".zip"):
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                df = pd.read_csv(zf.open(csv_name), encoding="latin-1", low_memory=False)
                df.to_csv(PSD_RAW, index=False)
                print(f"    Saved: {PSD_RAW}  ({len(df):,} rows)")
                return df
            else:
                with open(PSD_RAW, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                print(f"    Saved: {PSD_RAW}")
                return pd.read_csv(PSD_RAW, encoding="latin-1", low_memory=False)
        except Exception as e:
            print(f"    Failed: {e}")
    raise RuntimeError("All USDA PSD URLs failed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=== USDA PSD download ===\n")

if os.path.exists(PSD_RAW):
    print(f"  Using cached PSD: {PSD_RAW}")
    psd = pd.read_csv(PSD_RAW, encoding="latin-1", low_memory=False)
else:
    psd = download_psd()

print(f"  Loaded: {len(psd):,} rows, columns: {list(psd.columns[:8])} …")

# Normalise columns
psd.columns = [c.strip() for c in psd.columns]

# PSD columns: Country_Code, Country_Name, Commodity_Code, Commodity_Description,
#              Attribute_Description, Unit_Description, Year, Value

col_map = {}
col_lower = {c.lower(): c for c in psd.columns}  # lowercase→original map
for c in psd.columns:
    low = c.lower()
    if low == "country_name":
        col_map[c] = "country"
    elif low == "commodity_description":
        col_map[c] = "commodity"
    elif low == "attribute_description":
        col_map[c] = "attribute"
    elif low == "market_year":
        col_map[c] = "year"
    elif low == "calendar_year" and "market_year" not in col_lower:
        col_map[c] = "year"
    elif low == "value":
        col_map[c] = "value"
    elif low == "unit_description":
        col_map[c] = "unit"
    elif low == "country_code":
        col_map[c] = "country_code"

psd = psd.rename(columns=col_map)

# Filter
if "commodity" not in psd.columns:
    # Try Commodity_Description
    desc_col = next((c for c in psd.columns
                     if "commodity" in c.lower() and "desc" in c.lower()), None)
    if desc_col:
        psd = psd.rename(columns={desc_col: "commodity"})

if "attribute" not in psd.columns:
    attr_col = next((c for c in psd.columns
                     if "attribute" in c.lower()), None)
    if attr_col:
        psd = psd.rename(columns={attr_col: "attribute"})

if "year" not in psd.columns:
    for mc in ["Market_Year", "Calendar_Year"]:
        if mc in psd.columns:
            psd = psd.rename(columns={mc: "year"})
            break
if "year" in psd.columns:
    psd["year"] = pd.to_numeric(psd["year"], errors="coerce")
else:
    import numpy as _np
    psd["year"] = _np.nan

filtered = psd[
    psd["commodity"].isin(COMMODITY_KEEP) &
    psd["attribute"].isin(ATTRS_KEEP) &
    psd["year"].between(2000, 2022)
].copy()

print(f"  After filter: {len(filtered):,} rows")

# Pivot wider: country × commodity × year, attribute as columns
panel = filtered.pivot_table(
    index=["country", "commodity", "year"],
    columns="attribute",
    values="value",
    aggfunc="first",
).reset_index()
panel.columns = [
    c.lower().replace(" ", "_").replace(",", "").replace("-", "_")
    for c in panel.columns
]

panel.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}  ({len(panel):,} rows)")
if not panel.empty:
    print(f"Countries: {panel['country'].nunique()}")
    print(f"Commodities: {sorted(panel['commodity'].unique())}")
    if panel['year'].notna().any():
        print(f"Years: {int(panel['year'].min())}–{int(panel['year'].max())}")
print("\nNext: run 02_fetch_giews.py")
