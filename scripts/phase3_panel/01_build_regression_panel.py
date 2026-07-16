"""
Phase 3 Panel — Script 01
Build main regression panel for analysis

Computes:
  1. FCE (Flood Cropland Exposure) by country × crop × year
     FCE = flooded_cropland_km2 × crop_harvested_share [× damage_weight]
  2. FTE (Flood Trade Exposure) by importer × crop × year
     FTE(i, crop, t) = Σⱼ trade_share(i←j, crop, t) × FCE(j, crop, t)
  3. FTE_total: caloric-share–weighted sum of FTE across crops
  4. Joins all controls: ENSO, ERA5 domestic climate, GTA export restriction
  5. Joins FAO FBS outcome: log food supply kcal/pc/day

Output: outputs/regression/regression_panel.csv

Identification unit: importing country × year
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROJECT = (
    "./"
    "Other computers/My Laptop/UDel/Taky_research/flood_food_security"
)

# ── Paths ────────────────────────────────────────────────────────────────────
P = {
    "flood"    : os.path.join(PROJECT, "outputs/fce/annual_flood_area_by_country.csv"),
    "harvest"  : os.path.join(PROJECT, "outputs/fce/harvested_area_by_country_crop.csv"),
    "calendar" : os.path.join(PROJECT, "outputs/fce/crop_calendar_by_country_crop.csv"),
    "yield_a"  : os.path.join(PROJECT, "outputs/fce/yield_anomaly_by_country_crop_year.csv"),
    "baci"     : os.path.join(PROJECT, "outputs/trade/baci_trade_panel.csv"),
    "fao_fbs"  : os.path.join(PROJECT, "outputs/outcome/fao_fbs_panel.csv"),
    "enso"     : os.path.join(PROJECT, "outputs/outcome/enso_panel.csv"),
    "era5"     : os.path.join(PROJECT, "outputs/outcome/era5_climate_panel.csv"),
    "gta"      : os.path.join(PROJECT, "outputs/outcome/gta_export_restrictions.csv"),
    "wb_inc"   : os.path.join(PROJECT, "data/raw/wb_income_groups.csv"),
}
OUT_DIR  = os.path.join(PROJECT, "outputs", "regression")
OUT_PATH = os.path.join(OUT_DIR, "regression_panel.csv")
os.makedirs(OUT_DIR, exist_ok=True)

STUDY_YEARS = list(range(2000, 2022))

# Caloric density (kcal per kg) for major BACI commodities
KCAL_PER_KG = {
    "rice"    : 3_640,
    "wheat"   : 3_400,
    "maize"   : 3_650,
    "soybeans": 4_470,
    "veg_oils": 8_840,
}

# ── Utility: country name → ISO3 ─────────────────────────────────────────────
print("Building country crosswalk …")
import country_converter as coco

# Manual overrides for names that coco misses
MANUAL_ISO3 = {
    "Bolivia (Plurinational State of)"                : "BOL",
    "Bosnia and Herzegovina"                           : "BIH",
    "Brunei Darussalam"                                : "BRN",
    "Cabo Verde"                                       : "CPV",
    "China, mainland"                                  : "CHN",
    "China, Hong Kong SAR"                             : "HKG",
    "China, Macao SAR"                                 : "MAC",
    "China, Taiwan Province of"                        : "TWN",
    "CÃ´te d'Ivoire"                                  : "CIV",
    "Côte d'Ivoire"                                    : "CIV",
    "Democratic People's Republic of Korea"            : "PRK",
    "Democratic Republic of the Congo"                 : "COD",
    "Eswatini"                                         : "SWZ",
    "Iran (Islamic Republic of)"                       : "IRN",
    "Lao People's Democratic Republic"                 : "LAO",
    "Micronesia (Federated States of)"                 : "FSM",
    "Netherlands (Kingdom of the)"                     : "NLD",
    "Republic of Korea"                                : "KOR",
    "Republic of Moldova"                              : "MDA",
    "Russian Federation"                               : "RUS",
    "Syrian Arab Republic"                             : "SYR",
    "TÃ¼rkiye"                                         : "TUR",
    "Türkiye"                                          : "TUR",
    "United Kingdom of Great Britain and Northern Ireland": "GBR",
    "United Republic of Tanzania"                      : "TZA",
    "Venezuela (Bolivarian Republic of)"               : "VEN",
    "Viet Nam"                                         : "VNM",
    "Brunei Darussalam"                                : "BRN",
    "Dominican Republic"                               : "DOM",
    "Sao Tome and Principe"                            : "STP",
}

def name_to_iso3(names):
    """Convert list/Series of country names to ISO3, using coco with manual fallback."""
    cc = coco.CountryConverter()
    out = {}
    for name in set(names):
        if name in MANUAL_ISO3:
            out[name] = MANUAL_ISO3[name]
        else:
            iso = cc.convert(name, to="ISO3", not_found=None)
            # coco returns a list for ambiguous/aggregate regions — discard those
            if isinstance(iso, list) or iso is None or iso == "not found":
                out[name] = None
            else:
                out[name] = iso
    return out


# ── Load ERA5 as authoritative ISO3 ─────────────────────────────────────────
era5_raw = pd.read_csv(P["era5"])
# era5 has: country, iso3, year, plant_month, harvest_month, t2m_anom_C, tp_anom_frac
# One row per country × year (growing-season averaged)

# Deduplicate to get the iso3 crosswalk
era5_xwalk = era5_raw[["country", "iso3"]].drop_duplicates().dropna()
country_to_iso3 = dict(zip(era5_xwalk["country"], era5_xwalk["iso3"]))

print(f"  ERA5 countries: {len(country_to_iso3)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FLOOD CROPLAND EXPOSURE (FCE)
# FCE(country, crop, year) = flooded_cropland_km2 × crop_share
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Computing FCE …")

flood = pd.read_csv(P["flood"])
flood = flood[flood["year"].isin(STUDY_YEARS)].copy()
flood["iso3"] = flood["country"].map(country_to_iso3)
flood = flood.dropna(subset=["iso3"])

harvest = pd.read_csv(P["harvest"])
# Remove flood-adapted crops from exposure calculation
harvest = harvest[~harvest["flood_adapted_exclude"]].copy()
harvest["iso3"] = harvest["country"].map(country_to_iso3)
harvest = harvest.dropna(subset=["iso3"])
# Harmonise crop name: MapSPAM uses 'oil_crops'; BACI uses 'veg_oils'
harvest["crop"] = harvest["crop"].replace({"oil_crops": "veg_oils"})

# Crop share = crop harvested km² / total harvested km² per country
total_km2 = harvest.groupby("iso3")["harvested_km2"].sum().rename("total_harvested_km2")
harvest = harvest.merge(total_km2, on="iso3")
harvest["crop_share"] = harvest["harvested_km2"] / harvest["total_harvested_km2"]
harvest_share = harvest[["iso3", "crop", "crop_share"]]

# Load yield anomaly for damage-weighting (ISIMIP3a, 2000-2016)
yield_a = pd.read_csv(P["yield_a"])
yield_a["iso3"] = yield_a["country"].map(country_to_iso3)
yield_a = yield_a.dropna(subset=["iso3"])
# Harmonise crop name: ISIMIP3a uses 'soybeans'; MapSPAM may use 'soybean'
yield_a["crop"] = yield_a["crop"].replace({"oil_crops": "veg_oils", "soybean": "soybeans"})

# FCE (raw, no damage weight): country×crop×year
fce_base = (
    flood[["iso3", "year", "flooded_cropland_km2"]]
    .merge(harvest_share[["iso3", "crop", "crop_share"]], on="iso3", how="inner")
)
fce_base["fce_km2"] = fce_base["flooded_cropland_km2"] * fce_base["crop_share"]

# Add damage weight from yield anomaly (negative = damage → weight > 1)
# damage_weight = 1 + max(0, -yield_anomaly)
yield_a_pivot = yield_a[["iso3", "crop", "year", "yield_anomaly_ensemble_mean"]].copy()
yield_a_pivot["damage_weight"] = 1.0 + np.maximum(0, -yield_a_pivot["yield_anomaly_ensemble_mean"])

fce = fce_base.merge(
    yield_a_pivot[["iso3", "crop", "year", "damage_weight"]],
    on=["iso3", "crop", "year"],
    how="left",
)
fce["damage_weight"] = fce["damage_weight"].fillna(1.0)  # no weight if no yield data
fce["fce_weighted_km2"] = fce["fce_km2"] * fce["damage_weight"]

# Normalise FCE by total country cropland km² → exposure fraction
fce = fce.merge(
    flood[["iso3", "year", "country_cropland_km2"]].drop_duplicates(),
    on=["iso3", "year"],
    how="left",
)
fce["fce_frac"]          = fce["fce_km2"]          / fce["country_cropland_km2"].replace(0, np.nan)
fce["fce_weighted_frac"] = fce["fce_weighted_km2"] / fce["country_cropland_km2"].replace(0, np.nan)

print(f"  FCE rows: {len(fce):,}  |  crops: {sorted(fce['crop'].unique())}  |  countries: {fce['iso3'].nunique()}")
print(f"  FCE frac range: {fce['fce_frac'].min():.4f} – {fce['fce_frac'].max():.4f}")

# Save FCE
fce_out = fce[["iso3", "crop", "year", "fce_km2", "fce_frac",
               "fce_weighted_km2", "fce_weighted_frac"]].copy()
fce_out.to_csv(os.path.join(OUT_DIR, "fce_panel.csv"), index=False)
print(f"  Saved: outputs/regression/fce_panel.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FLOOD TRADE EXPOSURE (FTE)
# FTE(importer, crop, year) = Σⱼ trade_share(i←j,crop,t) × FCE(j,crop,t)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Computing FTE …")

baci = pd.read_csv(P["baci"])
baci = baci[baci["year"].isin(STUDY_YEARS)].copy()

# Keep only positive import quantities
baci = baci[baci["quantity_t"] > 0].copy()

# Compute trade share: importer ← exporter for each crop×year
baci_imp = (
    baci.groupby(["year", "importer", "exporter", "commodity"])["quantity_t"]
    .sum()
    .reset_index()
)

total_by_imp = (
    baci_imp.groupby(["year", "importer", "commodity"])["quantity_t"]
    .sum()
    .rename("total_import_t")
    .reset_index()
)
baci_imp = baci_imp.merge(total_by_imp, on=["year", "importer", "commodity"])
baci_imp["trade_share"] = baci_imp["quantity_t"] / baci_imp["total_import_t"]

# Rename for join clarity
baci_imp = baci_imp.rename(columns={"exporter": "iso3", "commodity": "crop"})

# Join FCE for exporter (uses iso3 of exporter)
fce_for_join = fce[["iso3", "crop", "year", "fce_frac", "fce_weighted_frac"]].copy()

fte_long = baci_imp.merge(
    fce_for_join,
    on=["iso3", "crop", "year"],
    how="left",
)
fte_long["fce_frac"]          = fte_long["fce_frac"].fillna(0)
fte_long["fce_weighted_frac"] = fte_long["fce_weighted_frac"].fillna(0)

# FTE = weighted sum of exporter FCE
fte_long["fte_contrib"]          = fte_long["trade_share"] * fte_long["fce_frac"]
fte_long["fte_weighted_contrib"] = fte_long["trade_share"] * fte_long["fce_weighted_frac"]

fte_crop = (
    fte_long.groupby(["year", "importer", "crop"])[
        ["fte_contrib", "fte_weighted_contrib"]
    ]
    .sum()
    .reset_index()
    .rename(columns={"importer": "iso3"})
)

print(f"  FTE crop-level rows: {len(fte_crop):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b — FTE_total: caloric–share weighted FTE across crops
# ─────────────────────────────────────────────────────────────────────────────
# For each importer: compute total caloric imports per crop → share → weight FTE

# Caloric import volume (in kcal) per importer × crop × year
baci_cal = (
    baci.groupby(["year", "importer", "commodity"])["quantity_t"]
    .sum()
    .reset_index()
)
baci_cal["kcal_density"] = baci_cal["commodity"].map(KCAL_PER_KG)
baci_cal["kcal_imports"] = baci_cal["quantity_t"] * 1000 * baci_cal["kcal_density"]

total_cal_imp = (
    baci_cal.groupby(["year", "importer"])["kcal_imports"]
    .sum()
    .rename("total_kcal_imports")
    .reset_index()
)
baci_cal = baci_cal.merge(total_cal_imp, on=["year", "importer"])
baci_cal["crop_kcal_share"] = baci_cal["kcal_imports"] / baci_cal["total_kcal_imports"].replace(0, np.nan)

# Merge crop FTE with caloric share
fte_crop_with_share = fte_crop.merge(
    baci_cal[["year", "importer", "commodity", "crop_kcal_share"]]
    .rename(columns={"importer": "iso3", "commodity": "crop"}),
    on=["year", "iso3", "crop"],
    how="left",
)
fte_crop_with_share["crop_kcal_share"] = fte_crop_with_share["crop_kcal_share"].fillna(0)
fte_crop_with_share["fte_total_contrib"]          = fte_crop_with_share["fte_contrib"]          * fte_crop_with_share["crop_kcal_share"]
fte_crop_with_share["fte_weighted_total_contrib"] = fte_crop_with_share["fte_weighted_contrib"] * fte_crop_with_share["crop_kcal_share"]

fte_total = (
    fte_crop_with_share.groupby(["year", "iso3"])[
        ["fte_total_contrib", "fte_weighted_total_contrib"]
    ]
    .sum()
    .reset_index()
    .rename(columns={
        "fte_total_contrib"         : "fte_total",
        "fte_weighted_total_contrib": "fte_weighted_total",
    })
)

# Save FTE
fte_out = fte_crop.copy()
fte_out.to_csv(os.path.join(OUT_DIR, "fte_by_crop_panel.csv"), index=False)
print(f"  Saved: outputs/regression/fte_by_crop_panel.csv")
print(f"  FTE total rows: {len(fte_total):,}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FAO FBS OUTCOME VARIABLES
# Primary: log_food_supply_kcal_pc_day (Grand Total)
# Secondary: log_cereal_import_qty_kt (wheat + maize)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Building FAO FBS outcomes …")

fbs = pd.read_csv(P["fao_fbs"])

# Build country → iso3 map for FAO names using coco
fao_names = fbs["country"].unique()
fao_iso3_map = name_to_iso3(fao_names)
fbs["iso3"] = fbs["country"].map(fao_iso3_map)

# Remove aggregate regions (iso3 will be None or 'not found')
fbs = fbs[fbs["iso3"].notna() & (fbs["iso3"] != "not found")].copy()

# Primary outcome: Grand Total food supply kcal/pc/day
grand = fbs[(fbs["Item"] == "Grand Total") & (fbs["element"] == "kcal_pc_day")].copy()
grand = grand[grand["year"].isin(STUDY_YEARS)][["iso3", "year", "value"]].rename(
    columns={"value": "food_supply_kcal_pc_day"}
)
grand = grand.dropna(subset=["food_supply_kcal_pc_day"])
grand["log_food_supply_kcal_pc_day"] = np.log(grand["food_supply_kcal_pc_day"].replace(0, np.nan))

# Secondary: staple commodity import quantities from FAO FBS
# (wheat, maize, soybeans, veg oils all have import_qty_kt; rice does not in current FBS)
STAPLE_ITEMS = [
    "Wheat and products", "Maize and products",
    "Rice and products", "Soyabeans", "Vegetable Oils",
]
staple_imp = fbs[
    (fbs["element"] == "import_qty_kt") &
    (fbs["Item"].isin(STAPLE_ITEMS))
].copy()
staple_imp = staple_imp[staple_imp["year"].isin(STUDY_YEARS)].groupby(["iso3", "year"])["value"].sum().reset_index()
staple_imp.columns = ["iso3", "year", "fbs_staple_import_kt"]
# Note: rice import_qty_kt is absent from current FAOSTAT FBS; covered by BACI caloric imports
staple_imp["fbs_staple_import_kt"] = pd.to_numeric(staple_imp["fbs_staple_import_kt"], errors="coerce")
staple_imp["log_fbs_staple_import_kt"] = np.log(staple_imp["fbs_staple_import_kt"].replace(0, np.nan))

# Caloric import volume from BACI (quantity_t × kcal/kg)
baci_cal_country = (
    baci_cal.groupby(["year", "importer"])["kcal_imports"]
    .sum()
    .reset_index()
    .rename(columns={"importer": "iso3", "kcal_imports": "total_kcal_imports_kt_eq"})
)
baci_cal_country["log_kcal_imports"] = np.log(baci_cal_country["total_kcal_imports_kt_eq"].replace(0, np.nan))

print(f"  Grand Total rows: {len(grand):,}  |  countries: {grand['iso3'].nunique()}")
print(f"  Staple import rows: {len(staple_imp):,}")
print(f"  BACI caloric import rows: {len(baci_cal_country):,}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ENSO CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Loading ENSO …")
enso = pd.read_csv(P["enso"])
enso = enso[enso["year"].isin(STUDY_YEARS)].copy()
enso_cols = ["year", "mei_annual_mean", "mei_DJF", "mei_MAM", "mei_JJA", "mei_SON",
             "oni_annual_mean"]
enso = enso[[c for c in enso_cols if c in enso.columns]].copy()
print(f"  ENSO rows: {len(enso):,}  |  cols: {list(enso.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — ERA5 DOMESTIC CLIMATE CONTROLS
# Country-level temperature and precipitation anomalies during growing season
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Loading ERA5 domestic climate …")
era5 = era5_raw[era5_raw["year"].isin(STUDY_YEARS)].copy()
era5_ctrl = era5[["iso3", "year", "t2m_anom_C", "tp_anom_frac"]].dropna(subset=["iso3"]).copy()
print(f"  ERA5 rows: {len(era5_ctrl):,}  |  countries: {era5_ctrl['iso3'].nunique()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — GTA EXPORT RESTRICTION CONTROL
# Trade-weighted average export restriction facing each importer
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Building GTA export restriction control …")
gta = pd.read_csv(P["gta"])
# gta: country, year, commodity, export_restriction (0/1), restriction_types, sources
# Rename 'country' to 'exporter' for clarity
gta = gta.rename(columns={"country": "exporter_name"})
gta["iso3_exporter"] = gta["exporter_name"].apply(
    lambda x: name_to_iso3([x]).get(x)
)

# Map GTA commodity to BACI commodity names
COMMODITY_MAP = {
    "wheat"    : "wheat",
    "rice"     : "rice",
    "maize"    : "maize",
    "soybeans" : "soybeans",
    "soybean"  : "soybeans",
    "palm oil" : "veg_oils",
    "sunflower" : "veg_oils",
}
gta["crop"] = gta["commodity"].str.lower().map(COMMODITY_MAP).fillna(gta["commodity"].str.lower())

# Only keep rows in STUDY_YEARS
gta = gta[gta["year"].isin(STUDY_YEARS)].copy()

# Build trade-weighted GTA exposure for each importer×year
# For each importer, weighted sum of exporter export restrictions
baci_for_gta = baci_imp.rename(columns={"iso3": "iso3_exporter"})

gta_trade = baci_for_gta.merge(
    gta[["year", "iso3_exporter", "crop", "export_restriction"]].dropna(subset=["iso3_exporter"]),
    on=["year", "iso3_exporter", "crop"],
    how="left",
)
gta_trade["export_restriction"] = gta_trade["export_restriction"].fillna(0)
gta_trade["gta_contrib"] = gta_trade["trade_share"] * gta_trade["export_restriction"]

# Aggregate to importer × crop level first, then weight by caloric import share
gta_crop = (
    gta_trade.groupby(["year", "importer", "crop"])["gta_contrib"]
    .sum()
    .reset_index()
)
# Merge caloric import shares to weight across crops
gta_crop = gta_crop.merge(
    baci_cal[["year", "importer", "commodity", "crop_kcal_share"]]
    .rename(columns={"commodity": "crop"}),
    on=["year", "importer", "crop"],
    how="left",
)
gta_crop["crop_kcal_share"] = gta_crop["crop_kcal_share"].fillna(0)
gta_crop["gta_cal_contrib"] = gta_crop["gta_contrib"] * gta_crop["crop_kcal_share"]

gta_control = (
    gta_crop.groupby(["year", "importer"])["gta_cal_contrib"]
    .sum()
    .reset_index()
    .rename(columns={"importer": "iso3", "gta_cal_contrib": "gta_weighted_restriction"})
)
print(f"  GTA control rows: {len(gta_control):,}  |  max weighted restriction: {gta_control['gta_weighted_restriction'].max():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — JOIN ALL INTO REGRESSION PANEL
# Unit: importing country (iso3) × year
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Assembling regression panel …")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6b — WORLD BANK INCOME GROUP
# ─────────────────────────────────────────────────────────────────────────────
wb = pd.read_csv(P["wb_inc"])
wb = wb[wb["income_level_id"].isin(["LIC", "LMC", "UMC", "HIC"])].copy()
wb = wb[["iso3", "income_level_id", "income_level"]].rename(
    columns={"income_level_id": "wb_income_group", "income_level": "wb_income_label"}
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6c — HERFINDAHL INDEX OF IMPORT CONCENTRATION
# HHI(i, crop, t) = Σⱼ trade_share(i←j,crop,t)²
# → measure of how concentrated import sourcing is
# ─────────────────────────────────────────────────────────────────────────────
hhi_crop = (
    baci_imp.groupby(["year", "importer", "crop"])
    .apply(lambda g: (g["trade_share"] ** 2).sum())
    .reset_index(name="hhi")
)
# Aggregate to importer-year (caloric-weighted average HHI across crops)
hhi_crop_cal = hhi_crop.merge(
    baci_cal[["year", "importer", "commodity", "crop_kcal_share"]]
    .rename(columns={"commodity": "crop"}),
    on=["year", "importer", "crop"],
    how="left",
)
hhi_crop_cal["crop_kcal_share"] = hhi_crop_cal["crop_kcal_share"].fillna(0)
hhi_crop_cal["hhi_weighted"] = hhi_crop_cal["hhi"] * hhi_crop_cal["crop_kcal_share"]
hhi_total = (
    hhi_crop_cal.groupby(["year", "importer"])["hhi_weighted"]
    .sum()
    .reset_index()
    .rename(columns={"importer": "iso3", "hhi_weighted": "hhi_import_concentration"})
)

# Start from FTE total (all importers × years with trade data)
panel = fte_total.copy()
panel = panel.rename(columns={"iso3": "iso3"})

# FTE by individual crop (pivot)
fte_pivot = (
    fte_crop[["iso3", "year", "crop", "fte_contrib"]]
    .pivot_table(index=["iso3", "year"], columns="crop", values="fte_contrib", fill_value=0)
    .reset_index()
)
fte_pivot.columns = ["iso3", "year"] + [f"fte_{c}" for c in fte_pivot.columns[2:]]
panel = panel.merge(fte_pivot, on=["iso3", "year"], how="left")

# FAO FBS: food supply
panel = panel.merge(grand[["iso3", "year", "food_supply_kcal_pc_day", "log_food_supply_kcal_pc_day"]],
                    on=["iso3", "year"], how="left")

# BACI caloric imports  
panel = panel.merge(baci_cal_country[["iso3", "year", "total_kcal_imports_kt_eq", "log_kcal_imports"]],
                    on=["iso3", "year"], how="left")

# FAO staple imports
panel = panel.merge(staple_imp[["iso3", "year", "fbs_staple_import_kt", "log_fbs_staple_import_kt"]],
                    on=["iso3", "year"], how="left")

# ENSO (year-level, broadcast across countries)
panel = panel.merge(enso, on="year", how="left")

# ERA5 domestic climate (country × year)
panel = panel.merge(era5_ctrl, on=["iso3", "year"], how="left")

# GTA restriction
panel = panel.merge(gta_control, on=["iso3", "year"], how="left")
panel["gta_weighted_restriction"] = panel["gta_weighted_restriction"].fillna(0)

# World Bank income group
panel = panel.merge(wb, on="iso3", how="left")

# HHI import concentration
panel = panel.merge(hhi_total, on=["iso3", "year"], how="left")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — DERIVED VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
# Lagged log food supply for first-difference / growth outcome
panel = panel.sort_values(["iso3", "year"])
panel["log_food_supply_lag1"] = panel.groupby("iso3")["log_food_supply_kcal_pc_day"].shift(1)
panel["d_log_food_supply"]    = panel["log_food_supply_kcal_pc_day"] - panel["log_food_supply_lag1"]

panel["log_kcal_imports_lag1"] = panel.groupby("iso3")["log_kcal_imports"].shift(1)
panel["d_log_kcal_imports"]    = panel["log_kcal_imports"] - panel["log_kcal_imports_lag1"]

# ENSO regional interactions: define ENSO regions
# These will be created as binary region flags for ENSO interaction terms
REGION_COUNTRIES = {
    "south_asia"    : ["IND", "BGD", "PAK", "LKA", "NPL", "BTN", "MDV", "AFG"],
    "southeast_asia": ["THA", "VNM", "IDN", "PHL", "MYS", "MMR", "KHM", "LAO", "SGP", "BRN"],
    "ssa"           : ["ETH", "KEN", "TZA", "UGA", "RWA", "BDI", "MOZ", "ZMB", "ZWE", "MWI",
                       "NGA", "GHA", "SEN", "MLI", "NER", "BFA", "CIV", "CMR", "TCD", "COD"],
    "central_america": ["MEX", "GTM", "SLV", "HND", "NIC", "CRI", "PAN"],
    "east_africa"   : ["ETH", "KEN", "TZA", "SOM", "ERI", "DJI"],
}
for region, countries in REGION_COUNTRIES.items():
    panel[f"region_{region}"] = panel["iso3"].isin(countries).astype(int)

# ENSO × region interactions (using ONI as the preferred ENSO metric)
enso_var = "oni_annual_mean" if "oni_annual_mean" in panel.columns else "mei_annual_mean"
for region in REGION_COUNTRIES:
    panel[f"enso_x_{region}"] = panel[enso_var] * panel[f"region_{region}"]

# Income group dummies (from WB)
for grp in ["LIC", "LMC", "UMC", "HIC"]:
    panel[f"inc_{grp}"] = (panel["wb_income_group"] == grp).astype(int)

# FTE × income group interactions (heterogeneity, Figure 3 Panel C / Figure 6)
fte_col = "fte_weighted_total"
for grp in ["LIC", "LMC", "UMC", "HIC"]:
    panel[f"fte_x_inc_{grp}"] = panel[fte_col] * panel[f"inc_{grp}"]

# FTE × HHI interaction (supply chain concentration heterogeneity)
panel["fte_x_hhi"] = panel[fte_col] * panel["hhi_import_concentration"]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — SAVE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)

print(f"\n[8] Panel summary:")
print(f"  Rows: {len(panel):,}")
print(f"  Importers: {panel['iso3'].nunique()}")
print(f"  Years: {panel['year'].min()}–{panel['year'].max()}")
print(f"  Columns: {len(panel.columns)}")
print(f"\n  FTE total range: {panel['fte_total'].min():.6f} – {panel['fte_total'].max():.6f}")
print(f"  FTE weighted range: {panel['fte_weighted_total'].min():.6f} – {panel['fte_weighted_total'].max():.6f}")
print(f"  Food supply obs: {panel['food_supply_kcal_pc_day'].notna().sum():,} / {len(panel):,}")
print(f"  ERA5 climate obs: {panel['t2m_anom_C'].notna().sum():,} / {len(panel):,}")
print(f"\n  Missingness by column:")
miss = panel.isnull().mean().sort_values(ascending=False)
for col, pct in miss[miss > 0].items():
    print(f"    {col}: {pct:.1%}")

panel.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")
print(f"Columns: {list(panel.columns)}")
