"""
Phase 1 — Dataset 1.7
Flood-adapted agriculture exclusion mask

Hardcodes three regions where flooding is agronomically normal or beneficial:
  1. Bangladesh deepwater rice zone
  2. Inner Niger Delta (Niger / Mali flood-recession farming)
  3. Mekong Delta (Vietnam scheduled inundation rice)

Output → data/processed/flood_adapted_mask.geojson

Polygons are hand-coded from published literature:
  - Mainuddin et al. (2019): Bangladesh deepwater rice belt
  - Zwarts et al. (2005): Inner Niger Delta boundaries
  - Mekong River Commission reports: Mekong Delta extent

Usage:
    python 05_build_adapted_mask.py

Install if needed:
    pip install geopandas shapely
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
from shapely.geometry import Polygon, mapping

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = (
    "./"
    "Other computers/My Laptop/UDel/Taky_research/flood_food_security"
)
PROC_DIR = os.path.join(PROJECT, "data", "processed")
OUT_PATH = os.path.join(PROC_DIR, "flood_adapted_mask.geojson")
os.makedirs(PROC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Region definitions  (lon_min, lat_min, lon_max, lat_max → Polygon)
# Each polygon is a bounding box of the published agro-ecological zone.
# ---------------------------------------------------------------------------

# 1. Bangladesh deepwater rice belt
#    Haor basin (Sylhet–Netrokona–Sunamganj): ~22.5–25.0°N, 90.5–92.5°E
#    Baro-Baor-Bill lowlands (Dhaka–Rajshahi chars): ~23.0–24.5°N, 89.0–91.0°E
#    Source: Mainuddin et al. (2019), Fig 1; IRRI water depth maps

BANGLADESH_HAOR = Polygon([
    (90.5, 22.5), (92.5, 22.5), (92.5, 25.0), (90.5, 25.0), (90.5, 22.5)
])

BANGLADESH_BARO = Polygon([
    (89.0, 23.0), (91.0, 23.0), (91.0, 24.5), (89.0, 24.5), (89.0, 23.0)
])

# 2. Inner Niger Delta (Mali / Niger border region)
#    Central Niger floodplain: ~13.0–16.5°N, 3.0–6.0°W
#    Source: Zwarts et al. (2005), Wetlands International Inner Niger Delta atlas

INNER_NIGER_DELTA = Polygon([
    (-6.0, 13.0), (-3.0, 13.0), (-3.0, 16.5), (-6.0, 16.5), (-6.0, 13.0)
])

# 3. Mekong Delta (Vietnam)
#    Mekong Delta proper: ~9.0–11.5°N, 104.5–106.8°E
#    Source: Mekong River Commission (2020) State of the Basin report

MEKONG_DELTA = Polygon([
    (104.5, 9.0), (106.8, 9.0), (106.8, 11.5), (104.5, 11.5), (104.5, 9.0)
])

# ---------------------------------------------------------------------------
# Assemble GeoDataFrame
# ---------------------------------------------------------------------------
features = [
    {
        "geometry" : BANGLADESH_HAOR,
        "region_id": "BGD_haor",
        "country"  : "Bangladesh",
        "system"   : "deepwater_rice",
        "crops"    : "rice",
        "description": (
            "Haor basin deepwater rice zone (Sylhet–Sunamganj–Netrokona). "
            "Naturally inundated June–November; Boro rice grown Nov–May. "
            "Source: Mainuddin et al. (2019), IRRI."
        ),
    },
    {
        "geometry" : BANGLADESH_BARO,
        "region_id": "BGD_baro",
        "country"  : "Bangladesh",
        "system"   : "deepwater_rice",
        "crops"    : "rice",
        "description": (
            "Baro-Baor-Bill chars (Dhaka–Rajshahi division). "
            "Flood-tolerant Aman rice, seasonally inundated. "
            "Source: IRRI water depth maps."
        ),
    },
    {
        "geometry" : INNER_NIGER_DELTA,
        "region_id": "MLI_NER_inner_niger_delta",
        "country"  : "Mali / Niger",
        "system"   : "flood_recession_farming",
        "crops"    : "sorghum,millet,rice",
        "description": (
            "Inner Niger Delta flood-recession farming zone. "
            "Crops planted on receding floodwaters July–October. "
            "Flooding is relied upon, not damaging. "
            "Source: Zwarts et al. (2005) Wetlands International."
        ),
    },
    {
        "geometry" : MEKONG_DELTA,
        "region_id": "VNM_mekong_delta",
        "country"  : "Vietnam",
        "system"   : "scheduled_inundation_rice",
        "crops"    : "rice",
        "description": (
            "Mekong Delta seasonal inundation zone (An Giang, Dong Thap, "
            "Kien Giang provinces). Annual August–November flooding is the "
            "primary water source for triple-cropped rice. "
            "Source: Mekong River Commission State of the Basin (2020)."
        ),
    },
]

gdf = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")

# ---------------------------------------------------------------------------
# Validate and save
# ---------------------------------------------------------------------------
print("Flood-adapted agriculture mask regions:")
for _, row in gdf.iterrows():
    area_km2 = gdf[gdf["region_id"] == row["region_id"]].to_crs(
        "EPSG:6933"   # equal-area projection
    ).geometry.area.values[0] / 1e6
    print(f"  {row['region_id']}: {row['country']} | {row['system']} "
          f"| ~{area_km2:,.0f} km²")

gdf.to_file(OUT_PATH, driver="GeoJSON")
print(f"\nSaved: {OUT_PATH}  ({len(gdf)} regions)")

# Also print GeoJSON for quick inspection
with open(OUT_PATH) as f:
    geojson = json.load(f)
print(f"GeoJSON type: {geojson['type']}, "
      f"features: {len(geojson['features'])}")

print("\nDone. Import this mask in 03_compute_fce.py to exclude adapted areas.")
