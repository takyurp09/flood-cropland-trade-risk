"""
Phase 3 — Dataset 3.5
IPC Food Security Phase Classification

Downloads IPC country-period phase data from the IPC Global Platform bulk download.
Collapses to annual country-level IPC max phase and % population in IPC ≥ 3.

Coverage: ~80 countries, 2009–present (sparse pre-2015).
Selection bias toward food-insecure countries — documented in output metadata.

Output → data/raw/ipc_raw.csv
          data/processed/ipc_panel.csv  (country-year panel)

Usage:
    python 05_fetch_ipc.py

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
PROC_DIR  = os.path.join(PROJECT, "data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

IPC_RAW   = os.path.join(RAW_DIR, "ipc_raw.csv")
OUT_PATH  = os.path.join(PROC_DIR, "ipc_panel.csv")

# Check for locally cached IPC data (from data/ipc/ directory)
IPC_LOCAL_DIR = os.path.join(
    "./"
    "Other computers/My Laptop/UDel/Taky_research/flood_food_security/data/ipc"
)

# ---------------------------------------------------------------------------
# IPC API / bulk download endpoints
# IPC provides a bulk data download and an API
# ---------------------------------------------------------------------------
IPC_BULK_URL  = "https://www.ipcinfo.org/ipc-country-analysis/ipc-data-download/"
IPC_API_BASE  = "https://api.ipcinfo.org/population"    # requires API key

HEADERS = {"User-Agent": "research-replication-bot/1.0 (contact: your-email@example.com)"}

# ---------------------------------------------------------------------------

def load_local_ipc():
    """Try to load IPC data from local directory."""
    if not os.path.isdir(IPC_LOCAL_DIR):
        return None

    files = [f for f in os.listdir(IPC_LOCAL_DIR)
             if f.endswith((".csv", ".xlsx", ".xls"))]
    if not files:
        return None

    frames = []
    for fname in files:
        fpath = os.path.join(IPC_LOCAL_DIR, fname)
        try:
            if fname.endswith(".csv"):
                df = pd.read_csv(fpath, low_memory=False)
            else:
                df = pd.read_excel(fpath)
            frames.append(df)
            print(f"    Loaded local IPC file: {fname}  ({len(df):,} rows)")
        except Exception as e:
            print(f"    WARNING: could not read {fname}: {e}")

    if frames:
        return pd.concat(frames, ignore_index=True)
    return None


def download_ipc_api():
    """Attempt IPC bulk data download (no API key — public endpoint)."""
    # IPC changed their API in 2023; try the public CSV endpoint
    endpoints = [
        "https://api.ipcinfo.org/population?format=csv&projection=current",
        "https://www.ipcinfo.org/fileadmin/user_upload/ipcinfo/docs/IPC_Population_Tracking_Tool_Data_2009_2023_May23.xlsx",
    ]
    for url in endpoints:
        try:
            print(f"  Trying: {url}")
            r = requests.get(url, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                ext = url.split(".")[-1].lower()
                if ext == "csv":
                    from io import StringIO
                    df = pd.read_csv(StringIO(r.text))
                elif ext in ("xls", "xlsx"):
                    from io import BytesIO
                    df = pd.read_excel(BytesIO(r.content))
                else:
                    from io import StringIO
                    df = pd.read_csv(StringIO(r.text))
                print(f"    Downloaded: {len(df):,} rows")
                return df
            else:
                print(f"    HTTP {r.status_code}")
        except Exception as e:
            print(f"    ERROR: {e}")
    return None


def standardise_ipc(df):
    """
    Map IPC column names to canonical schema and compute annual summary.

    Expected columns (IPC may use different names):
      Country, Year, Period (date range), Phase 1–5 population counts,
      Phase 3plus population, Total analysed population.
    """
    df.columns = [c.strip() for c in df.columns]

    # Column name normalisation
    col_map = {}
    for c in df.columns:
        low = c.lower().replace(" ", "_").replace("-", "_")
        if "country" in low and "iso" not in low:
            col_map[c] = "country"
        elif low in ("iso3", "iso3166", "iso_code"):
            col_map[c] = "iso3"
        elif low in ("year", "reference_year", "analysis_year"):
            col_map[c] = "year"
        elif low in ("phase_3_plus", "phase3plus", "ipc3plus",
                     "phase_3_5", "crisis_or_worse"):
            col_map[c] = "pop_ipc3plus"
        elif low in ("total", "total_pop", "total_analysed", "population_analysed"):
            col_map[c] = "pop_total_analysed"
        elif "phase" in low and "3" in low:
            col_map[c] = "pop_ipc3plus"
        elif "max" in low and "phase" in low:
            col_map[c] = "ipc_max_phase"

    df = df.rename(columns=col_map)

    # Extract year if not present
    if "year" not in df.columns:
        for c in df.columns:
            if "date" in c.lower() or "period" in c.lower():
                df["year"] = pd.to_datetime(df[c], errors="coerce").dt.year
                break

    if "year" not in df.columns:
        print("  WARNING: could not identify year column in IPC data.")
        return pd.DataFrame()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Numeric conversions
    for col in ["pop_ipc3plus", "pop_total_analysed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )

    # % population in IPC ≥ 3
    if "pop_ipc3plus" in df.columns and "pop_total_analysed" in df.columns:
        df["pct_ipc3plus"] = (
            df["pop_ipc3plus"] / df["pop_total_analysed"] * 100
        ).where(df["pop_total_analysed"] > 0)

    # Collapse to annual country level (max phase, sum of IPC3+ pop)
    agg_cols = {"pct_ipc3plus": "mean"}
    if "pop_ipc3plus" in df.columns:
        agg_cols["pop_ipc3plus"] = "sum"
    if "pop_total_analysed" in df.columns:
        agg_cols["pop_total_analysed"] = "sum"
    if "ipc_max_phase" in df.columns:
        agg_cols["ipc_max_phase"] = "max"

    group_cols = [c for c in ["country", "iso3", "year"] if c in df.columns]
    panel = (
        df[df["year"].between(2009, 2022)]
        .groupby(group_cols)
        .agg(agg_cols)
        .reset_index()
    )

    # Recompute pct from aggregated sums
    if "pop_ipc3plus" in panel.columns and "pop_total_analysed" in panel.columns:
        panel["pct_ipc3plus"] = (
            panel["pop_ipc3plus"] / panel["pop_total_analysed"] * 100
        ).where(panel["pop_total_analysed"] > 0)

    # Document selection bias: flag countries with data
    panel["has_ipc_data"] = True

    return panel.sort_values(["country", "year"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=== IPC Food Security Phase Classification ===\n")

df_raw = None

# 1. Check local cache
print("Step 1: Checking local IPC directory …")
df_raw = load_local_ipc()

# 2. Check cached raw file
if df_raw is None and os.path.exists(IPC_RAW):
    print(f"Step 2: Loading cached raw file: {IPC_RAW}")
    df_raw = pd.read_csv(IPC_RAW, low_memory=False)

# 3. Download
if df_raw is None:
    print("Step 3: Attempting download …")
    df_raw = download_ipc_api()

if df_raw is None:
    print(
        "\nWARNING: Could not obtain IPC data (requires auth). Creating scaffold CSV.\n"
        "Manual steps:\n"
        "  1. Go to https://www.ipcinfo.org/ipc-country-analysis/ipc-data-download/\n"
        "  2. Download the 'IPC and CH Country Data' CSV\n"
        "  3. Save to: " + IPC_RAW
    )
    scaffold = pd.DataFrame(columns=[
        "country", "iso3", "year", "ipc_max_phase", "pct_ipc3plus",
        "pop_ipc3plus", "pop_total_analysed", "has_ipc_data"
    ])
    scaffold.to_csv(OUT_PATH, index=False)
    print(f"Scaffold saved: {OUT_PATH}  (0 rows — manual download required)")
    print("\nTier 3 complete.")
    raise SystemExit(0)

# Cache raw
if not os.path.exists(IPC_RAW):
    df_raw.to_csv(IPC_RAW, index=False)
    print(f"Cached raw: {IPC_RAW}")

panel = standardise_ipc(df_raw)

if panel.empty:
    print("WARNING: IPC standardisation returned empty panel. Check column names.")
    print("Raw columns:", list(df_raw.columns))
else:
    panel.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}  ({len(panel):,} rows)")
    print(f"Countries: {panel['country'].nunique()}")
    print(f"Years: {int(panel['year'].min())}–{int(panel['year'].max())}")
    print("\nNOTE: IPC coverage is skewed toward food-insecure countries.")
    print("  Document selection bias in paper methods section.")

print("\nTier 3 complete. Run Tier 4 validation scripts next.")
