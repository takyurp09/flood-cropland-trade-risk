"""
Phase 3 — Dataset 3.2
Multivariate ENSO Index v2 (MEI v2) — NOAA PSL

Downloads MEI v2 bimonthly data from NOAA PSL.
Also downloads ONI (Oceanic Niño Index) as robustness check.
Computes annual mean and DJF/MAM/JJA/SON seasonal means.

Output → data/raw/mei_v2_raw.txt
          data/raw/oni_raw.txt
          data/processed/enso_panel.csv  (country-year doesn't apply;
                                          output is year × season panel)

Usage:
    python 02_fetch_mei_enso.py

Install if needed:
    pip install requests pandas numpy
"""

import os
import warnings
warnings.filterwarnings("ignore")

import requests
import numpy as np
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

MEI_RAW   = os.path.join(RAW_DIR, "mei_v2_raw.txt")
ONI_RAW   = os.path.join(RAW_DIR, "oni_raw.txt")
OUT_PATH  = os.path.join(OUT_DIR, "enso_panel.csv")

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
MEI_URL = "https://psl.noaa.gov/enso/mei/data/meiv2.data"
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

HEADERS = {"User-Agent": "research-replication-bot/1.0 (contact: your-email@example.com)"}

# ---------------------------------------------------------------------------

def fetch_mei_v2():
    """
    MEI v2 file format (space-delimited):
    Year  DJFM JFMA FMAM MAMJ AMJJ MJJA JJAS JASO ASON SOND NODS ONDJ
    (bimonthly seasons, 12 per year)
    Header rows start with non-numeric characters.
    """
    print("  Downloading MEI v2 …")
    r = requests.get(MEI_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()

    with open(MEI_RAW, "w") as f:
        f.write(r.text)
    print(f"    Saved raw: {MEI_RAW}")

    lines = [l for l in r.text.splitlines() if l.strip() and
             l.strip()[0].isdigit()]

    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) < 13:
            continue
        year = int(parts[0])
        vals = [float(v) if v not in ("-999", "-9.99") else np.nan
                for v in parts[1:13]]
        rows.append({"year": year,
                     "DJFM": vals[0], "JFMA": vals[1], "FMAM": vals[2],
                     "MAMJ": vals[3], "AMJJ": vals[4], "MJJA": vals[5],
                     "JJAS": vals[6], "JASO": vals[7], "ASON": vals[8],
                     "SOND": vals[9], "NODS": vals[10], "ONDJ": vals[11]})

    df = pd.DataFrame(rows)
    # Annual mean
    df["mei_annual_mean"] = df[[c for c in df.columns if c != "year"]].mean(axis=1)
    # Approx seasonal means
    df["mei_DJF"] = df[["DJFM", "ONDJ"]].mean(axis=1)
    df["mei_MAM"] = df[["FMAM", "MAMJ"]].mean(axis=1)
    df["mei_JJA"] = df[["MJJA", "JJAS"]].mean(axis=1)
    df["mei_SON"] = df[["ASON", "SOND"]].mean(axis=1)

    return df[["year", "mei_annual_mean", "mei_DJF", "mei_MAM",
               "mei_JJA", "mei_SON"]]


def fetch_oni():
    """
    ONI ASCII text from CPC (www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt).
    Format: SEAS  YR   TOTAL   ANOM  (12 rows per year, one per 3-month season)
    Computes annual mean of ANOM column across all 12 seasons.
    """
    print("  Downloading ONI …")
    try:
        r = requests.get(ONI_URL, headers=HEADERS, timeout=60)
        r.raise_for_status()

        with open(ONI_RAW, "w") as f:
            f.write(r.text)

        rows = []
        for line in r.text.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                yr   = int(parts[1])
                anom = float(parts[3])
                rows.append({"year": yr, "anom": anom})
            except ValueError:
                continue

        df = pd.DataFrame(rows)
        df = df.groupby("year", as_index=False)["anom"].mean()
        df = df.rename(columns={"anom": "oni_annual_mean"})
        return df[["year", "oni_annual_mean"]]

    except Exception as e:
        print(f"    WARNING: ONI download failed: {e}")
        return pd.DataFrame(columns=["year", "oni_annual_mean"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=== ENSO Index (MEI v2 + ONI) download ===\n")

mei_df = fetch_mei_v2()
oni_df = fetch_oni()

panel = mei_df.merge(oni_df, on="year", how="left")
panel = panel[panel["year"].between(2000, 2022)].copy()
panel = panel.sort_values("year")

panel.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}  ({len(panel)} rows)")
print(f"Years: {int(panel['year'].min())}–{int(panel['year'].max())}")
print(f"Columns: {list(panel.columns)}")
print("\nNext: run 03_fetch_era5_climate.py")
