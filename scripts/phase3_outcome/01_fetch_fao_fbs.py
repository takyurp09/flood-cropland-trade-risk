"""
Phase 3 — Dataset 3.1
FAO Food Balance Sheets (FAOSTAT)

Downloads FAO FBS (Food Balances) for all countries 1961–2022 via FAOSTAT bulk CSV.
Extracts caloric import availability per capita plus production, stock variation.

Output → data/raw/fao_fbs_raw.csv
          data/processed/fao_fbs_panel.csv  (country-year panel, tidy format)

Usage:
    python 01_fetch_fao_fbs.py

Install if needed:
    pip install requests pandas tqdm
"""

import os
import io
import zipfile
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
OUT_DIR   = os.path.join(PROJECT, "outputs", "outcome")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

RAW_PATH  = os.path.join(RAW_DIR, "fao_fbs_raw.csv")
OUT_PATH  = os.path.join(OUT_DIR, "fao_fbs_panel.csv")

# ---------------------------------------------------------------------------
# FAOSTAT bulk download endpoints
# FBS (new methodology, 2010-present) and FBSH (historical, 1961–2013)
# ---------------------------------------------------------------------------
FAOSTAT_FBS_URL  = "https://fenixservices.fao.org/faostat/static/bulkdownloads/FoodBalanceSheets_E_All_Data_(Normalized).zip"
FAOSTAT_FBSH_URL = "https://fenixservices.fao.org/faostat/static/bulkdownloads/FoodBalanceSheetsHistoric_E_All_Data_(Normalized).zip"

HEADERS = {"User-Agent": "research-replication-bot/1.0 (contact: your-email@example.com)"}

# Elements (FAOSTAT element codes) we want
ELEMENTS_KEEP = {
    "Food supply (kcal/capita/day)" : "kcal_pc_day",
    "Import Quantity"               : "import_qty_kt",
    "Export Quantity"               : "export_qty_kt",
    "Production"                    : "production_kt",
    "Stock Variation"               : "stock_variation_kt",
    "Food supply quantity (kg/capita/yr)": "food_supply_kg_pc",
}

# Items: aggregate food (FAOSTAT item code 2901 = "Grand Total")
# We keep all items and let users filter downstream for flexibility.
# Primary interest: total food supply + staple crops separately.
ITEMS_KEEP_CODES = [2901, 2511, 2513, 2514, 2807, 2555, 2914, 2517, 2520]
# 2901 Grand Total
# 2511 Wheat and products
# 2513 Barley and products
# 2514 Maize and products
# 2807 Rice and products  (correct FAOSTAT FBS code; 2516 is Oats in new FBS)
# 2555 Soyabeans
# 2914 Vegetable Oils
# 2517 Sorghum and products
# 2520 Cereals (others)

STUDY_YEARS = list(range(2000, 2022))

# ---------------------------------------------------------------------------

def download_faostat(url, label):
    """Download one FAOSTAT bulk zip, return DataFrame."""
    print(f"  Downloading {label} …")
    r = requests.get(url, headers=HEADERS, timeout=300, stream=True)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, encoding="latin-1", low_memory=False)
    print(f"    Loaded: {len(df):,} rows")
    return df


def tidy_fbs(df, start_year=2000, end_year=2021):
    """
    Tidy FAOSTAT normalised format:
    Wide form has Y#### columns. Normalised form already has Year column.
    Filter to study elements and years.
    """
    df.columns = [c.strip() for c in df.columns]

    # Keep only wanted elements
    if "Element" in df.columns:
        df = df[df["Element"].isin(ELEMENTS_KEEP.keys())].copy()

    # Year filter
    year_col = next((c for c in df.columns if c in ("Year", "year")), None)
    if year_col:
        df = df[pd.to_numeric(df[year_col], errors="coerce").between(
            start_year, end_year
        )].copy()
        df = df.rename(columns={year_col: "year"})

    # Country / ISO code
    if "Area Code (ISO3)" in df.columns:
        df = df.rename(columns={"Area Code (ISO3)": "iso3", "Area": "country"})
    elif "Area" in df.columns:
        df = df.rename(columns={"Area": "country"})

    # Item code
    if "Item Code" in df.columns:
        df = df[df["Item Code"].isin(ITEMS_KEEP_CODES)].copy()

    # Map element names to short names
    df["element"] = df["Element"].map(ELEMENTS_KEEP).fillna(df["Element"])

    # Value column
    value_col = next((c for c in df.columns if c.lower() == "value"), "Value")
    df = df.rename(columns={value_col: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    keep_cols = [c for c in ["country", "iso3", "year", "Item", "Item Code",
                              "element", "value"] if c in df.columns]
    return df[keep_cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=== FAO Food Balance Sheets download ===\n")

frames = []

for url, label in [(FAOSTAT_FBS_URL, "FBS (2010–present)"),
                   (FAOSTAT_FBSH_URL, "FBSH (historical 1961–2013)")]:
    try:
        df_raw = download_faostat(url, label)
        df_tidy = tidy_fbs(df_raw)
        frames.append(df_tidy)
        print(f"    Tidy rows: {len(df_tidy):,}")
    except Exception as e:
        print(f"  ERROR fetching {label}: {e}")

if not frames:
    print("WARNING: Could not download FBS data (FAOSTAT API unavailable). Creating scaffold CSV.")
    scaffold = pd.DataFrame(columns=[
        "country", "iso3", "year", "Item Code", "item", "element", "value", "unit"
    ])
    scaffold.to_csv(OUT_PATH, index=False)
    print(f"Scaffold saved: {OUT_PATH}  (0 rows — manual download required)")
    print("\nNext: run 02_fetch_mei_enso.py")
    raise SystemExit(0)

panel = pd.concat(frames, ignore_index=True)

# Remove duplicate years (FBS and FBSH overlap 2010–2013 — keep FBS for those)
panel = panel.sort_values(["country", "year", "element"]).drop_duplicates(
    subset=["country", "year", "Item Code", "element"], keep="first"
)

panel = panel.sort_values(["country", "year"])

panel.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}  ({len(panel):,} rows)")
print(f"Years: {int(panel['year'].min())}–{int(panel['year'].max())}")
print(f"Countries: {panel['country'].nunique()}")
print(f"Elements: {sorted(panel['element'].unique())}")
print("\nNext: run 02_fetch_mei_enso.py")
