"""
Phase 4 — Dataset 4.2 + 4.3
FAO GIEWS Crop Prospects + National Ministry Statistics

4.2 GIEWS: Downloads FAO GIEWS FPMA crop production index data via API.
    Serves as second independent production loss estimate for 10 validation events.

4.3 National ministries: Hardcodes URLs for 5 key validation countries.
    Checks if reports are accessible and saves metadata.

Output → data/raw/giews_raw.csv
          data/processed/giews_panel.csv
          data/processed/national_stats_metadata.csv

Usage:
    python 02_fetch_giews_national.py

Install if needed:
    pip install requests pandas
"""

import os
import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np

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

GIEWS_RAW    = os.path.join(RAW_DIR, "giews_raw.csv")
GIEWS_OUT    = os.path.join(OUT_DIR, "giews_panel.csv")
NAT_OUT      = os.path.join(OUT_DIR, "national_stats_metadata.csv")

HEADERS = {"User-Agent": "research-replication-bot/1.0 (contact: your-email@example.com)"}

# ---------------------------------------------------------------------------
# GIEWS: FPMA Tool monthly food price and production data
# FAO GIEWS does not expose a machine-readable API for crop production reports.
# However, the FPMA Tool has a data API for price monitoring.
# For production validation, we use the FAOSTAT Production domain (QCL)
# as a proxy — this is what GIEWS draws from for assessments.
# ---------------------------------------------------------------------------
FAOSTAT_QCL_URL = (
    "https://fenixservices.fao.org/faostat/static/bulkdownloads/"
    "Production_Crops_Livestock_E_All_Data_(Normalized).zip"
)

# Crop items in FAOSTAT QCL for validation
QCL_ITEMS = {
    "Rice, paddy"     : "rice",
    "Wheat"           : "wheat",
    "Maize (corn)"    : "maize",
    "Soybeans"        : "soybeans",
    "Sunflower seed"  : "oil_crops",
    "Rapeseed or canola": "oil_crops",
}

# Countries × years for our 10 validation events
VALIDATION_EVENTS = [
    {"country": "Pakistan",   "year": 2010, "crop": "wheat"},
    {"country": "Thailand",   "year": 2011, "crop": "rice"},
    {"country": "Nigeria",    "year": 2012, "crop": "maize"},
    {"country": "Pakistan",   "year": 2015, "crop": "wheat"},
    {"country": "Bangladesh", "year": 2017, "crop": "rice"},
    {"country": "India",      "year": 2019, "crop": "wheat"},
    {"country": "India",      "year": 2020, "crop": "wheat"},
    {"country": "China",      "year": 2016, "crop": "maize"},
    {"country": "Indonesia",  "year": 2013, "crop": "rice"},
    {"country": "Myanmar",    "year": 2015, "crop": "rice"},
]

# ---------------------------------------------------------------------------
# Dataset 4.3: National ministry URLs and access status
# ---------------------------------------------------------------------------
NATIONAL_SOURCES = [
    {
        "country": "Pakistan", "iso3": "PAK",
        "agency" : "Pakistan Bureau of Statistics",
        "url"    : "https://www.pbs.gov.pk/content/agriculture-statistics",
        "validation_years": [2010, 2015],
        "crops"  : ["wheat", "rice"],
    },
    {
        "country": "Thailand", "iso3": "THA",
        "agency" : "Office of Agricultural Economics (OAE)",
        "url"    : "http://www.oae.go.th/view/1/TH-TH",
        "validation_years": [2011],
        "crops"  : ["rice"],
    },
    {
        "country": "Nigeria", "iso3": "NGA",
        "agency" : "National Bureau of Statistics (NBS)",
        "url"    : "https://www.nigerianstat.gov.ng",
        "validation_years": [2012],
        "crops"  : ["maize", "sorghum"],
    },
    {
        "country": "Bangladesh", "iso3": "BGD",
        "agency" : "Bangladesh Bureau of Statistics (BBS)",
        "url"    : "http://www.bbs.gov.bd",
        "validation_years": [2017],
        "crops"  : ["rice"],
    },
    {
        "country": "India", "iso3": "IND",
        "agency" : "Directorate of Economics and Statistics (DES)",
        "url"    : "https://aps.dac.gov.in",
        "validation_years": [2019, 2020],
        "crops"  : ["wheat", "rice"],
    },
]

# ---------------------------------------------------------------------------

def download_giews_proxy():
    """
    Download FAOSTAT QCL (crop production) as GIEWS proxy.
    Saves to data/raw/ and returns validation-event subset.
    """
    import zipfile, io

    print("  Downloading FAOSTAT QCL (production data) as GIEWS proxy …")
    try:
        r = requests.get(FAOSTAT_QCL_URL, headers=HEADERS, timeout=300,
                         stream=True)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        print(f"    Loaded: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def check_url_accessible(url, timeout=15):
    """HEAD request to check if URL responds."""
    try:
        r = requests.head(url, headers=HEADERS, timeout=timeout,
                          allow_redirects=True)
        return r.status_code < 400, r.status_code
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=== FAO GIEWS + National Statistics (Validation) ===\n")

# --- GIEWS / FAOSTAT QCL ---
print("4.2 GIEWS proxy via FAOSTAT QCL:")

if os.path.exists(GIEWS_RAW):
    print(f"  Using cached: {GIEWS_RAW}")
    qcl_raw = pd.read_csv(GIEWS_RAW, encoding="latin-1", low_memory=False)
else:
    qcl_raw = download_giews_proxy()
    if qcl_raw is not None:
        qcl_raw.to_csv(GIEWS_RAW, index=False)

if qcl_raw is not None:
    qcl_raw.columns = [c.strip() for c in qcl_raw.columns]

    # Filter to validation items and years
    item_col = next((c for c in qcl_raw.columns if "item" in c.lower()
                     and "code" not in c.lower()), "Item")
    area_col = next((c for c in qcl_raw.columns if "area" in c.lower()
                     and "code" not in c.lower()), "Area")
    year_col = next((c for c in qcl_raw.columns if c.lower() == "year"), "Year")
    val_col  = next((c for c in qcl_raw.columns if c.lower() == "value"), "Value")
    elem_col = next((c for c in qcl_raw.columns if "element" in c.lower()
                     and "code" not in c.lower()), "Element")

    qcl_raw[year_col] = pd.to_numeric(qcl_raw[year_col], errors="coerce")
    qcl_raw[val_col]  = pd.to_numeric(qcl_raw[val_col], errors="coerce")

    mask = (
        qcl_raw[item_col].isin(QCL_ITEMS.keys()) &
        qcl_raw[elem_col].isin(["Area harvested", "Production", "Yield"]) &
        qcl_raw[year_col].between(1998, 2022)
    )
    giews_panel = qcl_raw[mask].copy()
    giews_panel["crop"] = giews_panel[item_col].map(QCL_ITEMS)
    giews_panel = giews_panel.rename(columns={
        area_col: "country", year_col: "year",
        elem_col: "element", val_col: "value"
    })

    # Compute year-on-year anomaly within country × crop
    giews_panel = giews_panel.sort_values(["country", "crop", "element", "year"])
    giews_panel["value_lag"] = giews_panel.groupby(
        ["country", "crop", "element"]
    )["value"].shift(1)
    giews_panel["production_anomaly_pct"] = (
        (giews_panel["value"] - giews_panel["value_lag"])
        / giews_panel["value_lag"] * 100
    )

    giews_panel.to_csv(GIEWS_OUT, index=False)
    print(f"  Saved: {GIEWS_OUT}  ({len(giews_panel):,} rows)")

    # Print validation event subset
    print("\n  Validation event production anomalies:")
    for evt in VALIDATION_EVENTS:
        match = giews_panel[
            (giews_panel["country"].str.contains(evt["country"], case=False, na=False)) &
            (giews_panel["year"] == evt["year"]) &
            (giews_panel["crop"] == evt["crop"]) &
            (giews_panel["element"] == "Production")
        ]
        if not match.empty:
            anom = match["production_anomaly_pct"].values[0]
            print(f"    {evt['country']} {evt['year']} {evt['crop']}: {anom:+.1f}%")
        else:
            print(f"    {evt['country']} {evt['year']} {evt['crop']}: no match")

# --- National Ministry Stats ---
print("\n4.3 National Ministry URL accessibility check:")

nat_records = []
for src in NATIONAL_SOURCES:
    accessible, status = check_url_accessible(src["url"])
    rec = {
        "country"         : src["country"],
        "iso3"            : src["iso3"],
        "agency"          : src["agency"],
        "url"             : src["url"],
        "http_status"     : status,
        "accessible"      : accessible,
        "validation_years": str(src["validation_years"]),
        "crops"           : str(src["crops"]),
        "note"            : "Manual download required — national ministry portals",
    }
    nat_records.append(rec)
    status_str = f"HTTP {status}" if status else "TIMEOUT"
    print(f"  {src['country']} ({src['agency']}): {status_str} "
          f"{'✓' if accessible else '✗ — manual access needed'}")

nat_df = pd.DataFrame(nat_records)
nat_df.to_csv(NAT_OUT, index=False)
print(f"\nSaved: {NAT_OUT}")
print("\nNext: run 03_fetch_modis_ndvi.py")
