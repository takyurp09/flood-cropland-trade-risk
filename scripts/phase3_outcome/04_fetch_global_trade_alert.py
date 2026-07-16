"""
Phase 3 — Dataset 3.4
Global Trade Alert — Export Restriction Database

Downloads bulk data from Global Trade Alert.
Filters to food commodity export interventions (bans, quotas, licences).
Supplements pre-2009 years with a hardcoded list of known major events.

Output → data/raw/gta_raw.csv
          data/processed/gta_export_restrictions.csv  (country-year-commodity panel)

Usage:
    python 04_fetch_global_trade_alert.py

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
OUT_DIR   = os.path.join(PROJECT, "outputs", "outcome")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

GTA_RAW   = os.path.join(RAW_DIR, "gta_raw.csv")
OUT_PATH  = os.path.join(OUT_DIR, "gta_export_restrictions.csv")

# ---------------------------------------------------------------------------
# GTA bulk download endpoint
# GTA provides a data extraction API at globaltradealert.org
# ---------------------------------------------------------------------------
GTA_URL = "https://www.globaltradealert.org/data_extraction/download-all"

HEADERS = {"User-Agent": "research-replication-bot/1.0 (contact: your-email@example.com)"}

# Food commodity HS prefixes to filter (same as BACI groups)
FOOD_HS_PREFIXES = [
    "1001",   # Wheat
    "1005",   # Maize
    "1006",   # Rice
    "1201",   # Soybeans
    "1507", "1508", "1509", "1510", "1511", "1512",
    "1513", "1514", "1515",  # Vegetable oils
]

# GTA intervention types that constitute export restrictions
EXPORT_RESTRICTION_TYPES = [
    "Export ban", "Export quota", "Export licensing requirement",
    "Export subsidy", "Export tax", "Export price control",
    "Quantitative export restriction (not otherwise specified)",
]

# ---------------------------------------------------------------------------
# Comprehensive hardcoded export restriction events (manual coding)
# Sources: Headey & Fan (2008), Anderson & Nelgen (2013), Liefert & Westcott
#   (2015), Schmidhuber et al. (2019), IFPRI FAPT, FAO, USDA ERS reports,
#   Glauber & Mamun (2022) "Export Restrictions and Food Security"
#
# Format: one dict per country-year-commodity event.
# Events expanded to annual rows (if multi-year, have one entry per year).
# ---------------------------------------------------------------------------

def _expand(country, iso3, years, commodity, rtype, source):
    """Helper: expand a multi-year restriction to one dict per year."""
    if isinstance(years, int):
        years = [years]
    return [{"country": country, "iso3": iso3, "year": y,
              "commodity": commodity, "restriction_type": rtype,
              "source": source} for y in years]

_E = _expand  # shorthand

HARDCODED_EVENTS = []

# ── WHEAT ────────────────────────────────────────────────────────────────────
# Argentina: export ROEs (withholding taxes) throughout 2002-2021
# Taxes ranged 5-30%; effectively act as export restriction
HARDCODED_EVENTS += _E("Argentina", "ARG", range(2002, 2022), "wheat",
                        "Export tax", "Anderson&Nelgen2013;IFPRI_FAPT")

# Russia: export tax 2003, 2007-08; embargo Aug2010-Jun2011; tax 2014-15, 2021+
HARDCODED_EVENTS += _E("Russia", "RUS", [2003], "wheat", "Export tax", "Anderson&Nelgen2013")
HARDCODED_EVENTS += _E("Russia", "RUS", [2007, 2008], "wheat", "Export tax", "Anderson&Nelgen2013")
HARDCODED_EVENTS += _E("Russia", "RUS", [2010, 2011], "wheat", "Export ban", "Liefert&Westcott2015")
HARDCODED_EVENTS += _E("Russia", "RUS", [2014, 2015], "wheat", "Export tax", "GTA;USDA_ERS")
HARDCODED_EVENTS += _E("Russia", "RUS", [2021], "wheat", "Export quota", "GTA")

# Ukraine: export quotas 2006-2007, 2010-2012, license requirements 2019-2021
HARDCODED_EVENTS += _E("Ukraine", "UKR", [2006, 2007], "wheat", "Export quota", "Anderson&Nelgen2013")
HARDCODED_EVENTS += _E("Ukraine", "UKR", [2008], "wheat", "Export quota", "Anderson&Nelgen2013")
HARDCODED_EVENTS += _E("Ukraine", "UKR", range(2010, 2014), "wheat", "Export quota", "Liefert&Westcott2015")
HARDCODED_EVENTS += _E("Ukraine", "UKR", range(2019, 2022), "wheat", "Export licensing requirement", "GTA")

# Kazakhstan: export bans/quotas 2008, 2012-2013
HARDCODED_EVENTS += _E("Kazakhstan", "KAZ", [2008, 2012, 2013], "wheat", "Export ban", "Anderson&Nelgen2013")

# India: export ban on wheat 2007-2011
HARDCODED_EVENTS += _E("India", "IND", range(2007, 2012), "wheat", "Export ban", "FAO/Headey&Fan2008")

# Pakistan: export ban 2001-2003, 2007-2008
HARDCODED_EVENTS += _E("Pakistan", "PAK", [2001, 2002, 2003, 2007, 2008], "wheat",
                        "Export ban", "Anderson&Nelgen2013")

# Serbia: export ban 2008
HARDCODED_EVENTS += _E("Serbia", "SRB", [2008], "wheat", "Export ban", "GTA")

# Belarus: export ban (aligned with Russia) 2010-2011
HARDCODED_EVENTS += _E("Belarus", "BLR", [2010, 2011], "wheat", "Export ban", "GTA")

# China: export tax 2007-2009
HARDCODED_EVENTS += _E("China", "CHN", [2007, 2008, 2009], "wheat", "Export tax", "FAO_2008")

# Iran: export restrictions various years
HARDCODED_EVENTS += _E("Iran", "IRN", [2008, 2010, 2011, 2012], "wheat", "Export ban", "FAO;USDA")

# Bulgaria/Romania: EU accession transition quotas 2007-2008
HARDCODED_EVENTS += _E("Bulgaria", "BGR", [2007, 2008], "wheat", "Export quota", "EC_reports")
HARDCODED_EVENTS += _E("Romania", "ROM", [2007, 2008], "wheat", "Export quota", "EC_reports")

# Moldova: ban 2008, 2012
HARDCODED_EVENTS += _E("Moldova", "MDA", [2008, 2012], "wheat", "Export ban", "FAO")

# Azerbaijan: ban 2008
HARDCODED_EVENTS += _E("Azerbaijan", "AZE", [2008], "wheat", "Export ban", "FAO")

# Turkey: export restrictions 2007-2009
HARDCODED_EVENTS += _E("Turkey", "TUR", [2007, 2008, 2009], "wheat", "Export licensing requirement", "FAO")

# ── RICE ─────────────────────────────────────────────────────────────────────
# India: MEP on non-basmati rice, full ban 2007-2011, partial restriction 2012-2021
HARDCODED_EVENTS += _E("India", "IND", range(2007, 2012), "rice", "Export ban", "FAO/Headey&Fan2008")
HARDCODED_EVENTS += _E("India", "IND", range(2012, 2022), "rice", "Export licensing requirement",
                        "USDA_ERS;GTA")

# Vietnam: export ban/quota 2008; quantity managed most years
HARDCODED_EVENTS += _E("Vietnam", "VNM", [2008], "rice", "Export ban", "FAO/Headey&Fan2008")
HARDCODED_EVENTS += _E("Vietnam", "VNM", range(2009, 2022), "rice", "Export quota",
                        "Liefert&Westcott2015;GTA")

# Cambodia: export ban 2008
HARDCODED_EVENTS += _E("Cambodia", "KHM", [2008], "rice", "Export ban", "FAO/Headey&Fan2008")

# Egypt: export ban 2008-2010, tax 2009
HARDCODED_EVENTS += _E("Egypt", "EGY", [2008, 2009, 2010], "rice", "Export ban", "FAO/Headey&Fan2008")

# China: export tax 2007-2008
HARDCODED_EVENTS += _E("China", "CHN", [2007, 2008], "rice", "Export tax", "FAO_2008")

# Pakistan: MEP and export restrictions 2007-2009
HARDCODED_EVENTS += _E("Pakistan", "PAK", [2007, 2008, 2009], "rice",
                        "Export licensing requirement", "Anderson&Nelgen2013")

# Myanmar: informal restrictions 2003-2013 (state-controlled exports)
HARDCODED_EVENTS += _E("Myanmar", "MMR", range(2003, 2014), "rice", "Export quota", "FAO;IRRI")

# Tanzania: ban 2005-2006
HARDCODED_EVENTS += _E("Tanzania", "TZA", [2005, 2006], "rice", "Export ban", "FAO_Africa")

# Thailand: export licensing 2000-2013 (state trading enterprise involvement)
HARDCODED_EVENTS += _E("Thailand", "THA", range(2000, 2014), "rice",
                        "Export licensing requirement", "FAO;USDA")

# ── MAIZE ────────────────────────────────────────────────────────────────────
# Ukraine: export quotas 2006-2007, 2010-2013
HARDCODED_EVENTS += _E("Ukraine", "UKR", [2006, 2007], "maize", "Export quota", "Anderson&Nelgen2013")
HARDCODED_EVENTS += _E("Ukraine", "UKR", range(2010, 2014), "maize", "Export quota",
                        "Liefert&Westcott2015")

# India: export ban 2008
HARDCODED_EVENTS += _E("India", "IND", [2008], "maize", "Export ban", "FAO_2008")

# Tanzania: ban 2005-2006, 2009, 2017
HARDCODED_EVENTS += _E("Tanzania", "TZA", [2005, 2006, 2009, 2017], "maize", "Export ban", "FAO_Africa")

# Zambia: sporadic bans 2002, 2005, 2012, 2015, 2016
HARDCODED_EVENTS += _E("Zambia", "ZMB", [2002, 2005, 2012, 2015, 2016], "maize",
                        "Export ban", "FAO_Africa")

# Kenya: restrictions various years
HARDCODED_EVENTS += _E("Kenya", "KEN", [2008, 2009, 2011], "maize", "Export ban", "FAO_Africa")

# Zimbabwe: ban throughout early 2000s
HARDCODED_EVENTS += _E("Zimbabwe", "ZWE", range(2000, 2009), "maize", "Export ban", "FAO_Africa")

# China: export tax 2007-2009
HARDCODED_EVENTS += _E("China", "CHN", [2007, 2008, 2009], "maize", "Export tax", "FAO_2008")

# Argentina: export ROEs on maize 2002-2021
HARDCODED_EVENTS += _E("Argentina", "ARG", range(2002, 2022), "maize",
                        "Export tax", "Anderson&Nelgen2013;IFPRI_FAPT")

# South Africa: informal restrictions 2002-2003 (drought year)
HARDCODED_EVENTS += _E("South Africa", "ZAF", [2002, 2003], "maize", "Export quota", "SADC_reports")

# Serbia: ban 2011-2012
HARDCODED_EVENTS += _E("Serbia", "SRB", [2011, 2012], "maize", "Export ban", "GTA")

# ── SOYBEANS ─────────────────────────────────────────────────────────────────
# Argentina: continuous export tax on soybeans 35% throughout 2002-2021
HARDCODED_EVENTS += _E("Argentina", "ARG", range(2002, 2022), "soybeans",
                        "Export tax", "Anderson&Nelgen2013;IFPRI_FAPT")

# India: export restrictions on oilseeds 2000-2021
HARDCODED_EVENTS += _E("India", "IND", range(2000, 2022), "soybeans",
                        "Export licensing requirement", "USDA_ERS")

# China: various soybean export taxes 2007-2009
HARDCODED_EVENTS += _E("China", "CHN", [2007, 2008, 2009], "soybeans", "Export tax", "FAO_2008")

# ── VEGETABLE OILS ───────────────────────────────────────────────────────────
# Argentina: export tax on soybean oil throughout 2002-2021
HARDCODED_EVENTS += _E("Argentina", "ARG", range(2002, 2022), "veg_oils",
                        "Export tax", "Anderson&Nelgen2013;IFPRI_FAPT")

# Indonesia: CPO export levy (progressive) 2007+; DMO 2018+
HARDCODED_EVENTS += _E("Indonesia", "IDN", range(2007, 2022), "veg_oils",
                        "Export tax", "USDA_ERS;GTA")

# Malaysia: palm oil export duty 2000-2021 (variable rate)
HARDCODED_EVENTS += _E("Malaysia", "MYS", range(2000, 2022), "veg_oils",
                        "Export tax", "USDA_ERS;MPOB_reports")

# Ukraine: sunflower oil export restrictions 2015-2021
HARDCODED_EVENTS += _E("Ukraine", "UKR", range(2015, 2022), "veg_oils",
                        "Export licensing requirement", "GTA")

# India: export restrictions on palm/edible oils 2008, 2012
HARDCODED_EVENTS += _E("India", "IND", [2008, 2012], "veg_oils",
                        "Export licensing requirement", "FAO;USDA")

PRE_GTA_EVENTS = HARDCODED_EVENTS


# ---------------------------------------------------------------------------

def download_gta():
    """Attempt to download GTA bulk dataset."""
    print("  Downloading GTA bulk data …")
    try:
        r = requests.get(GTA_URL, headers=HEADERS, timeout=300, stream=True)
        if r.status_code == 200:
            content_type = r.headers.get("Content-Type", "")
            if "zip" in content_type or "octet" in content_type:
                import zipfile, io
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f, low_memory=False)
            else:
                from io import StringIO
                df = pd.read_csv(StringIO(r.text), low_memory=False)

            df.to_csv(GTA_RAW, index=False)
            print(f"    Saved: {GTA_RAW}  ({len(df):,} rows)")
            return df
        else:
            print(f"    HTTP {r.status_code}")
            return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def filter_gta(df):
    """Filter GTA data to food-commodity export restrictions."""
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names (GTA has changed schema)
    col_map = {}
    for c in df.columns:
        low = c.lower()
        if "implementing" in low or "announcing" in low:
            col_map[c] = "country"
        elif "year" in low and "implement" in low:
            col_map[c] = "year_implemented"
        elif "year" in low and "removal" in low:
            col_map[c] = "year_removed"
        elif "type" in low and "interven" in low:
            col_map[c] = "intervention_type"
        elif "hs" in low and "product" in low:
            col_map[c] = "hs_products"
    df = df.rename(columns=col_map)

    # Filter to export restrictions
    if "intervention_type" in df.columns:
        df = df[df["intervention_type"].isin(EXPORT_RESTRICTION_TYPES)].copy()

    # Filter to food commodities by HS code
    if "hs_products" in df.columns:
        mask = df["hs_products"].astype(str).apply(
            lambda x: any(pref in x for pref in FOOD_HS_PREFIXES)
        )
        df = df[mask].copy()

    return df


def standardise_gta(df):
    """Build a clean country-year-commodity binary indicator panel."""
    records = []

    for _, row in df.iterrows():
        hs = str(row.get("hs_products", ""))
        yr_impl = pd.to_numeric(row.get("year_implemented", np.nan), errors="coerce")
        yr_rem  = pd.to_numeric(row.get("year_removed", np.nan), errors="coerce")

        if pd.isna(yr_impl):
            continue

        years_active = range(
            int(yr_impl),
            int(yr_rem) + 1 if not pd.isna(yr_rem) else 2022
        )

        for pref, commodity in [
            ("1001", "wheat"), ("1005", "maize"), ("1006", "rice"),
            ("1201", "soybeans"), ("1507", "veg_oils"), ("1508", "veg_oils"),
            ("1509", "veg_oils"), ("1510", "veg_oils"), ("1511", "veg_oils"),
            ("1512", "veg_oils"), ("1513", "veg_oils"), ("1514", "veg_oils"),
            ("1515", "veg_oils"),
        ]:
            if pref in hs:
                for yr in years_active:
                    if 2000 <= yr <= 2021:
                        records.append({
                            "country"          : row.get("country", ""),
                            "year"             : yr,
                            "commodity"        : commodity,
                            "intervention_type": row.get("intervention_type", ""),
                            "source"           : "GTA",
                        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=== Global Trade Alert — Export Restriction panel ===\n")

import numpy as np

# ── Try to load raw GTA / download if available ───────────────────────────
gta_df = None

if os.path.exists(GTA_RAW):
    print(f"  Using cached GTA file: {GTA_RAW}")
    gta_df = pd.read_csv(GTA_RAW, low_memory=False)
else:
    gta_df = download_gta()

all_records = []

if gta_df is not None:
    filtered  = filter_gta(gta_df)
    std_panel = standardise_gta(filtered)
    all_records.append(std_panel)
    print(f"  GTA download panel: {len(std_panel):,} country-year-commodity rows")
else:
    print("  GTA bulk download unavailable — using literature-coded events only")

# ── Add comprehensive hardcoded events ────────────────────────────────────
print(f"  Hardcoded events: {len(HARDCODED_EVENTS):,} country-year-commodity entries")
hardcoded_df = pd.DataFrame(PRE_GTA_EVENTS)   # PRE_GTA_EVENTS = HARDCODED_EVENTS
hardcoded_df = hardcoded_df.rename(columns={"restriction_type": "restriction_type",
                                             "source": "source"})
# Keep consistent column set
hardcoded_df = hardcoded_df[["country", "year", "commodity", "restriction_type", "source"]]
all_records.append(hardcoded_df)

# ── Combine & deduplicate ─────────────────────────────────────────────────
panel = pd.concat(all_records, ignore_index=True)
panel["export_restriction"] = 1

panel = (
    panel[panel["year"].between(2000, 2021)]
    .groupby(["country", "year", "commodity"], as_index=False)
    .agg(
        export_restriction=("export_restriction", "max"),
        restriction_types=("restriction_type", lambda x: "|".join(x.dropna().unique())),
        sources=("source", lambda x: "|".join(x.dropna().unique())),
    )
    .sort_values(["country", "year", "commodity"])
    .reset_index(drop=True)
)

panel.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}  ({len(panel):,} rows)")
print(f"  Countries: {panel['country'].nunique()}")
print(f"  Years: {int(panel['year'].min())}–{int(panel['year'].max())}")
print(f"  Commodities: {sorted(panel['commodity'].unique())}")
print(f"  Year-count distribution:")
print(panel.groupby("year")["country"].count().to_string())
print("\nNext: run 05_fetch_ipc.py")
