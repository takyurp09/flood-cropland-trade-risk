"""
02_process_gee_annual_flood.py
Convert GEE JRC annual flood output → annual_flood_area_by_country.csv
used by phase3 regression panel.

Input:  data/takyurp09/annual_flood_area_fixed_2000_2021.csv
        (ADM0_NAME, year, flooded_km2)

Output: outputs/fce/annual_flood_area_by_country.csv
        (country, year, flooded_km2, country_cropland_km2, flooded_cropland_km2, source)

Strategy:
  flooded_cropland_km2 = flooded_km2 × (cropland_km2 / country_area_km2)
  country_area_km2  → from Natural Earth 110m shapefile (projected to Mollweide)
  country_cropland_km2 → from existing DFO-based file (which used MapSPAM)
  country name → ISO3 via country_converter + manual overrides
"""
import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import country_converter as coco

warnings.filterwarnings("ignore")

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GEE_FILE  = os.path.join(PROJECT, "data/takyurp09/annual_flood_area_fixed_2000_2021.csv")
DFO_FILE  = os.path.join(PROJECT, "outputs/fce/annual_flood_area_by_country.csv")
NE_SHP    = os.path.join(PROJECT, "data/country_shapes/ne_110m_admin_0_countries.shp")
OUT_FILE  = os.path.join(PROJECT, "outputs/fce/annual_flood_area_by_country.csv")

# ── Manual overrides for ADM0_NAME → ISO3 ─────────────────────────────────
MANUAL_ISO3 = {
    "Russian Federation"                               : "RUS",
    "Viet Nam"                                         : "VNM",
    "Republic of Korea"                                : "KOR",
    "Dem People's Rep of Korea"                        : "PRK",
    "United Republic of Tanzania"                      : "TZA",
    "Democratic Republic of the Congo"                 : "COD",
    "Syrian Arab Republic"                             : "SYR",
    "Iran  (Islamic Republic of)"                      : "IRN",
    "Iran (Islamic Republic of)"                       : "IRN",
    "Lao People's Democratic Republic"                 : "LAO",
    "Brunei Darussalam"                                : "BRN",
    "Dominican Republic"                               : "DOM",
    "Sao Tome and Principe"                            : "STP",
    "Bosnia and Herzegovina"                           : "BIH",
    "Micronesia (Federated States of)"                 : "FSM",
    "Moldova, Republic of"                             : "MDA",
    "U.K. of Great Britain and Northern Ireland"       : "GBR",
    "Swaziland"                                        : "SWZ",
    "Czech Republic"                                   : "CZE",
    "The former Yugoslav Republic of Macedonia"        : "MKD",
    "Central African Republic"                         : "CAF",
    "Equatorial Guinea"                                : "GNQ",
    "Cape Verde"                                       : "CPV",
    "Antigua and Barbuda"                              : "ATG",
    "Saint Kitts and Nevis"                            : "KNA",
    "Saint Lucia"                                      : "LCA",
    "Saint Vincent and the Grenadines"                 : "VCT",
    "Trinidad and Tobago"                              : "TTO",
    "Falkland Islands (Malvinas)"                      : "FLK",
    "South Sudan"                                      : "SSD",
}

# ── Load GEE data ──────────────────────────────────────────────────────────
print("Loading GEE annual flood data …")
gee = pd.read_csv(GEE_FILE)
print(f"  Shape: {gee.shape}, countries: {gee['ADM0_NAME'].nunique()}, years: {gee['year'].nunique()}")

# ── ISO3 crosswalk ─────────────────────────────────────────────────────────
print("Converting country names to ISO3 …")
cc = coco.CountryConverter()
all_names = gee["ADM0_NAME"].unique()
iso3_map = {}
for name in all_names:
    if name in MANUAL_ISO3:
        iso3_map[name] = MANUAL_ISO3[name]
    else:
        iso3 = cc.convert(name, to="ISO3", not_found=None)
        if isinstance(iso3, list) or iso3 is None or iso3 == "not found":
            iso3_map[name] = None
        else:
            iso3_map[name] = iso3

gee["iso3"] = gee["ADM0_NAME"].map(iso3_map)
n_unmatched = gee["iso3"].isna().sum()
unmatched_names = gee.loc[gee["iso3"].isna(), "ADM0_NAME"].unique()
print(f"  Unmatched rows: {n_unmatched} ({len(unmatched_names)} unique names)")
if len(unmatched_names) > 0 and len(unmatched_names) <= 30:
    for nm in sorted(unmatched_names):
        print(f"    {nm}")

gee = gee.dropna(subset=["iso3"]).copy()
print(f"  After drop: {gee.shape}, {gee['iso3'].nunique()} ISO3 countries")

# ── Aggregate countries with multiple ADM0 entries to one ISO3 ─────────────
gee_agg = (
    gee.groupby(["iso3", "year"], as_index=False)["flooded_km2"].sum()
)
print(f"  After ISO3 aggregation: {gee_agg.shape}")

# ── Country total area from Natural Earth ──────────────────────────────────
print("Computing country areas from Natural Earth shapefile …")
shp = gpd.read_file(NE_SHP)
shp_ea = shp.to_crs("+proj=moll +datum=WGS84")
shp["area_km2"] = shp_ea.geometry.area / 1e6

# Map NE ISO_A3 → area; NE uses -99 for some territories
ne_area = shp[shp["ISO_A3"] != "-99"][["ISO_A3", "area_km2"]].copy()
ne_area.columns = ["iso3", "country_area_km2"]
ne_area = ne_area.groupby("iso3", as_index=False)["country_area_km2"].sum()

# ── Country cropland from existing DFO-based FCE file ─────────────────────
print("Loading MapSPAM cropland areas from existing FCE file …")
dfo = pd.read_csv(DFO_FILE)
# Map country name → iso3 using same logic
existing_iso3_map = {}
for cname in dfo["country"].unique():
    if cname in MANUAL_ISO3:
        existing_iso3_map[cname] = MANUAL_ISO3[cname]
    else:
        iso3 = cc.convert(cname, to="ISO3", not_found=None)
        if isinstance(iso3, list) or iso3 is None or iso3 == "not found":
            existing_iso3_map[cname] = None
        else:
            existing_iso3_map[cname] = iso3

dfo["iso3"] = dfo["country"].map(existing_iso3_map)
cropland = (
    dfo[["iso3", "country", "country_cropland_km2"]]
    .dropna(subset=["iso3"])
    .drop_duplicates(subset=["iso3"])
)
print(f"  Cropland data for {len(cropland)} countries")

# ── Join and compute flooded_cropland_km2 ─────────────────────────────────
print("Computing flooded cropland …")
merged = gee_agg.merge(ne_area, on="iso3", how="left")
merged = merged.merge(cropland[["iso3", "country", "country_cropland_km2"]], on="iso3", how="left")

# Countries missing from NE 110m (small islands, territories) → use cropland data only
# For those, we still need a country area — fill with median cropland/area ratio
median_ratio = (
    merged.dropna(subset=["country_area_km2", "country_cropland_km2"])
    .assign(ratio=lambda df: df["country_cropland_km2"] / df["country_area_km2"])
    ["ratio"].median()
)
print(f"  Median cropland/area ratio: {median_ratio:.4f}")

# For countries where we have cropland but no NE area, estimate area from cropland
no_area = merged["country_area_km2"].isna() & merged["country_cropland_km2"].notna()
merged.loc[no_area, "country_area_km2"] = merged.loc[no_area, "country_cropland_km2"] / median_ratio
n_filled = no_area.sum()
if n_filled > 0:
    print(f"  Filled {n_filled} rows with estimated country area")

# flooded_cropland_km2 = flooded_km2 × min(1.0, cropland_km2 / area_km2)
merged["cropland_frac"] = (
    merged["country_cropland_km2"] / merged["country_area_km2"]
).clip(upper=1.0)

# For rows still missing cropland/area: set fraction to global median
still_missing = merged["cropland_frac"].isna().sum()
if still_missing > 0:
    print(f"  {still_missing} rows still missing cropland_frac → using median {median_ratio:.4f}")
    merged["cropland_frac"] = merged["cropland_frac"].fillna(median_ratio)

merged["flooded_cropland_km2"] = merged["flooded_km2"] * merged["cropland_frac"]

# ── Fill country name where missing ───────────────────────────────────────
iso3_to_name = cropland.set_index("iso3")["country"].to_dict()
ne_name = shp.set_index("ISO_A3")["NAME"].to_dict()
merged["country"] = merged["country"].fillna(merged["iso3"].map(ne_name))
merged["country"] = merged["country"].fillna(merged["iso3"].map(iso3_to_name))
merged["country"] = merged["country"].fillna(merged["iso3"])  # fallback to ISO3

# ── Add source tag and write ───────────────────────────────────────────────
merged["source"] = "GEE_JRC"

out_cols = ["country", "year", "flooded_km2", "country_cropland_km2", "flooded_cropland_km2", "source"]
# Retain iso3 for later merge convenience (panel builder maps country again anyway)
out = merged[out_cols + ["iso3"]].copy()
out = out.sort_values(["country", "year"]).reset_index(drop=True)

print(f"\nFinal output: {out.shape}")
print(f"  Countries: {out['iso3'].nunique()}, Years: {sorted(out['year'].unique())}")
print(f"  flooded_cropland_km2 range: {out['flooded_cropland_km2'].min():.1f} – {out['flooded_cropland_km2'].max():.1f}")
print(f"  Nonzero flooded cropland rows: {(out['flooded_cropland_km2'] > 0).sum()}/{len(out)}")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
out.to_csv(OUT_FILE, index=False)
print(f"\nSaved → {OUT_FILE}")

# ── Sanity check: year-on-year variation ──────────────────────────────────
print("\nTop 10 flood events (flooded_cropland_km2):")
print(out.nlargest(10, "flooded_cropland_km2")[["country","year","flooded_km2","flooded_cropland_km2"]].to_string(index=False))

print("\nMean annual flooded cropland by year (global sum):")
print(out.groupby("year")["flooded_cropland_km2"].sum().to_string())
