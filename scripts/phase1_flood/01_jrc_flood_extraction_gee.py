"""
Phase 1 — Dataset 1.1
JRC Monthly Water History extraction via Google Earth Engine
FIXED 1: .unmask(0) before .lt(80) in PERM mask
FIXED 2: rename sum → flooded_km2 after reduceRegions
FIXED 3: removed getInfo() sanity check (times out on global compute)
"""
import ee, json, os
from datetime import datetime

GEE_PROJECT  = "t2m-precip-monthly-country"
DRIVE_FOLDER = "takyurp09"
YEARS        = list(range(2000, 2022))
SCALE        = 500

ee.Initialize(project=GEE_PROJECT)
print(f"GEE initialised | {datetime.now():%Y-%m-%d %H:%M:%S}")

COUNTRIES = ee.FeatureCollection("FAO/GAUL/2015/level0")

PERM = (
    ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    .select("occurrence").unmask(0).lt(80)
)
JRC = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")

def flood_image(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, "month")
    img   = JRC.filterDate(start, end).first()
    img   = ee.Algorithms.If(
        img,
        ee.Image(img).select("water").eq(2).updateMask(PERM).unmask(0),
        ee.Image.constant(0)
    )
    return ee.Image(img).rename("flood").set("year", year, "month", month)

def annual_stats(year):
    months     = ee.ImageCollection([flood_image(year, m) for m in range(1,13)])
    annual     = months.max().rename("flood")
    pixel_area = ee.Image.pixelArea().divide(1e6)
    flooded    = annual.multiply(pixel_area)
    stats      = flooded.reduceRegions(
        collection=COUNTRIES, reducer=ee.Reducer.sum(),
        scale=SCALE, crs="EPSG:4326")
    return stats.map(lambda f: f.set("flooded_km2", f.get("sum"), "year", year))

def monthly_stats(year, month):
    img        = flood_image(year, month)
    pixel_area = ee.Image.pixelArea().divide(1e6)
    flooded    = ee.Image(img).multiply(pixel_area)
    stats      = flooded.reduceRegions(
        collection=COUNTRIES, reducer=ee.Reducer.sum(),
        scale=SCALE, crs="EPSG:4326")
    return stats.map(lambda f: f.set(
        "flooded_km2", f.get("sum"), "year", year, "month", month))

print("Building collections...")
annual_fc  = ee.FeatureCollection([annual_stats(y) for y in YEARS]).flatten()
monthly_fc = ee.FeatureCollection(
    [monthly_stats(y, m) for y in YEARS for m in range(1,13)]).flatten()

task_a = ee.batch.Export.table.toDrive(
    collection     = annual_fc,
    description    = "annual_flood_area_fixed_2000_2021",
    folder         = DRIVE_FOLDER,
    fileNamePrefix = "annual_flood_area_fixed_2000_2021",
    fileFormat     = "CSV",
    selectors      = ["ADM0_NAME", "year", "flooded_km2"],
)
task_a.start()
print(f"Annual  task: {task_a.id}")

task_m = ee.batch.Export.table.toDrive(
    collection     = monthly_fc,
    description    = "monthly_flood_area_fixed_2000_2021",
    folder         = DRIVE_FOLDER,
    fileNamePrefix = "monthly_flood_area_fixed_2000_2021",
    fileFormat     = "CSV",
    selectors      = ["ADM0_NAME", "year", "month", "flooded_km2"],
)
task_m.start()
print(f"Monthly task: {task_m.id}")
print(f"Monitor: https://code.earthengine.google.com/tasks")

os.makedirs("logs", exist_ok=True)
log = {"submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
       "task_annual": task_a.id, "task_monthly": task_m.id}
with open("logs/gee_task_fixed.json", "w") as f:
    json.dump(log, f, indent=2)
print(f"Log saved → logs/gee_task_fixed.json")
