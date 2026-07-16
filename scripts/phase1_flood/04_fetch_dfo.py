"""
Phase 1 — Dataset 1.6
Dartmouth Flood Observatory (DFO) Global Active Archive download

Downloads the event catalogue from:
    https://floodobservatory.colorado.edu/Archives/index.html

The DFO archive is served as a plain HTML page with an embedded table.
This script parses the table and saves it as a clean CSV.

Output → data/raw/dfo_flood_archive.csv

Usage:
    python 04_fetch_dfo.py

Install if needed:
    pip install requests pandas lxml
"""

import os
import time
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
RAW_DIR  = os.path.join(PROJECT, "data", "raw")
OUT_PATH = os.path.join(RAW_DIR, "dfo_flood_archive.csv")
os.makedirs(RAW_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DFO source URLs (try both; the archive format has changed over time)
# ---------------------------------------------------------------------------
# Primary: the main archive index page (HTML table)
DFO_HTML_URL = "https://floodobservatory.colorado.edu/Archives/index.html"

# Fallback: some mirrors expose a raw CSV/Excel
DFO_CSV_MIRRORS = [
    # DFO GloFAS collaboration export (if available)
    "https://floodobservatory.colorado.edu/Archives/ArchiveTabular.csv",
    # Alternative encoding hosted by researchers
    "https://floodobservatory.colorado.edu/Archives/GlobalFloodsRecord.xls",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; academic-research-bot; "
        "+mailto:your-email@example.com)"
    )
}

# ---------------------------------------------------------------------------
# Attempt 1: direct CSV mirror
# ---------------------------------------------------------------------------
def try_csv_mirrors():
    for url in DFO_CSV_MIRRORS:
        ext = os.path.splitext(url)[1].lower()
        try:
            print(f"  Trying: {url}")
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code != 200:
                print(f"    HTTP {r.status_code} — skipping.")
                continue
            if ext == ".csv":
                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                print(f"    CSV loaded: {len(df):,} rows.")
                return df
            elif ext in (".xls", ".xlsx"):
                from io import BytesIO
                df = pd.read_excel(BytesIO(r.content))
                print(f"    Excel loaded: {len(df):,} rows.")
                return df
        except Exception as e:
            print(f"    Error: {e}")
    return None


# ---------------------------------------------------------------------------
# Attempt 2: parse HTML table from index page
# ---------------------------------------------------------------------------
def parse_html_table():
    print(f"\n  Fetching HTML: {DFO_HTML_URL}")
    try:
        r = requests.get(DFO_HTML_URL, headers=HEADERS, timeout=120)
        r.raise_for_status()
    except Exception as e:
        print(f"  ERROR fetching DFO page: {e}")
        return None

    try:
        tables = pd.read_html(r.text)
    except Exception as e:
        print(f"  ERROR parsing HTML tables: {e}")
        return None

    if not tables:
        print("  No tables found in DFO HTML page.")
        return None

    # Pick the largest table (the event catalogue)
    df = max(tables, key=len)
    print(f"  HTML table parsed: {len(df):,} rows × {df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# Standardise column names to common schema
# ---------------------------------------------------------------------------
def standardise_dfo(df):
    """
    DFO column names have varied across releases. Map to a canonical schema.
    Columns we want: ID, Country, OtherCountry, long, lat,
                     Area (km²), Started, Duration (days), Dead, Displaced,
                     Maincause, Severity
    """
    rename_map = {}
    for col in df.columns:
        low = col.lower().strip()
        if "flood" in low and "id" in low:
            rename_map[col] = "flood_id"
        elif low in ("id", "id#", "serial", "number"):
            rename_map[col] = "flood_id"
        elif "country" in low and "other" not in low:
            rename_map[col] = "country"
        elif "othercountry" in low or ("other" in low and "country" in low):
            rename_map[col] = "other_country"
        elif "long" in low or "lon" in low:
            rename_map[col] = "lon"
        elif "lat" in low:
            rename_map[col] = "lat"
        elif "area" in low:
            rename_map[col] = "area_km2"
        elif "began" in low or "started" in low or "begin" in low:
            rename_map[col] = "date_began"
        elif "ended" in low or "end" in low:
            rename_map[col] = "date_ended"
        elif "duration" in low:
            rename_map[col] = "duration_days"
        elif "dead" in low or "death" in low:
            rename_map[col] = "dead"
        elif "displac" in low:
            rename_map[col] = "displaced"
        elif "cause" in low or "maincause" in low:
            rename_map[col] = "main_cause"
        elif "severity" in low or "class" in low:
            rename_map[col] = "severity"

    df = df.rename(columns=rename_map)

    # Parse date columns
    for dcol in ["date_began", "date_ended"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce",
                                      infer_datetime_format=True)

    # Add year column for easy filtering
    if "date_began" in df.columns:
        df["year"] = df["date_began"].dt.year

    # Filter to study period (1985–present)
    if "year" in df.columns:
        n_before = len(df)
        df = df[df["year"].between(1985, 2025, inclusive="both")].copy()
        print(f"  Filtered to 1985–2025: {len(df):,} of {n_before:,} events retained.")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=== DFO Global Flood Archive download ===\n")

df = None

# Try CSV / Excel mirrors first (faster, no parsing)
print("Step 1: Trying direct CSV/Excel mirrors …")
df = try_csv_mirrors()

# Fallback: parse HTML
if df is None:
    print("Step 2: Parsing HTML table …")
    df = parse_html_table()

if df is None:
    print(
        "\nERROR: Could not obtain DFO data from any source.\n"
        "Manual steps:\n"
        "  1. Go to https://floodobservatory.colorado.edu/Archives/index.html\n"
        "  2. Download the full table (CSV or Excel option if available)\n"
        "  3. Save to: " + OUT_PATH
    )
    raise SystemExit(1)

# Standardise
df = standardise_dfo(df)

# Save
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}  ({len(df):,} rows × {df.shape[1]} columns)")
print("Columns:", list(df.columns))
print("\nDone. Run 05_build_adapted_mask.py next.")
