"""
Phase 2 -- Dataset 2.1
BACI Bilateral Trade Flows (CEPII)

Reads BACI HS92 V202601 annual files from local disk.
Source: data/raw/baci/BACI_HS92_VYYYYMM/
Filters to 5 food commodity groups:
  - Wheat:         HS 1001
  - Maize:         HS 1005
  - Rice:          HS 1006
  - Soybeans:      HS 1201
  - Vegetable oils: HS 1507-1515

Output:
  outputs/trade/baci_trade_panel.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm

PROJECT = (
    "./"
    "Other computers/My Laptop/UDel/Taky_research/flood_food_security"
)
BACI_DIR = "data/raw/baci/BACI_HS92_VYYYYMM"
OUT_DIR  = os.path.join(PROJECT, "outputs", "trade")
OUT_PATH = os.path.join(OUT_DIR, "baci_trade_panel.csv")
os.makedirs(OUT_DIR, exist_ok=True)

COUNTRY_CODES_PATH = os.path.join(BACI_DIR, "country_codes_V202601.csv")

COMMODITY_GROUPS = {
    "wheat"    : (100100, 100199),
    "maize"    : (100500, 100599),
    "rice"     : (100600, 100699),
    "soybeans" : (120100, 120199),
    "veg_oils" : (150700, 151599),
}

YEARS = list(range(2000, 2022))


def hs6_mask(hs6_series):
    mask = pd.Series(False, index=hs6_series.index)
    for lo, hi in COMMODITY_GROUPS.values():
        mask |= hs6_series.between(lo, hi)
    return mask


def assign_commodity(hs6):
    for name, (lo, hi) in COMMODITY_GROUPS.items():
        if lo <= hs6 <= hi:
            return name
    return "other"


def read_baci_year(year, country_map):
    fpath = os.path.join(BACI_DIR, f"BACI_HS92_Y{year}_V202601.csv")
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={
        "t": "year", "i": "exporter_code", "j": "importer_code",
        "k": "hs6", "v": "value_kusd", "q": "quantity_t"
    })
    df["hs6"] = pd.to_numeric(df["hs6"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["hs6"])
    df = df[hs6_mask(df["hs6"])].copy()
    if df.empty:
        return None
    df["commodity"] = df["hs6"].apply(assign_commodity)
    df["year"]      = int(year)
    df["exporter"]  = df["exporter_code"].map(country_map).fillna(df["exporter_code"].astype(str))
    df["importer"]  = df["importer_code"].map(country_map).fillna(df["importer_code"].astype(str))
    return df[["year", "exporter", "importer", "hs6", "commodity", "value_kusd", "quantity_t"]]


print("=== BACI trade flow panel (local disk) ===")
print(f"Source: {BACI_DIR}\n")

country_map = {}
if os.path.exists(COUNTRY_CODES_PATH):
    cc = pd.read_csv(COUNTRY_CODES_PATH)
    country_map = dict(zip(cc["country_code"], cc["country_iso3"]))
    print(f"Loaded {len(country_map)} country code mappings")

all_frames = []
for year in tqdm(YEARS, desc="Years"):
    df_yr = read_baci_year(year, country_map)
    if df_yr is None:
        print(f"  {year}: file not found -- skipping")
        continue
    all_frames.append(df_yr)

if not all_frames:
    print("ERROR: No data loaded. Check BACI_DIR.")
    raise SystemExit(1)

panel = pd.concat(all_frames, ignore_index=True)
panel = panel.sort_values(["year", "exporter", "importer", "commodity"])
panel.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}  ({len(panel):,} rows)")
print(f"Years: {int(panel['year'].min())}-{int(panel['year'].max())}")
print(f"Commodities: {sorted(panel['commodity'].unique())}")
print(f"Exporters: {panel['exporter'].nunique()}, Importers: {panel['importer'].nunique()}")
