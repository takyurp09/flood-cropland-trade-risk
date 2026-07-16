"""
03b_extend_isimip_yield_anomaly.py
Extend ISIMIP 3a yield anomaly panel from 2016 to 2021.

Strategy:
  - 2017-2021: Fill with 0 (no anomaly) and add missing_yield_flag=1 dummy
    Rationale: ISIMIP 3b uses GCM-forced (not observation-driven) protocol;
    introducing zero assumption is conservative and avoids spurious trend
    attribution. Damage weight = 1 + max(0,-anomaly) = 1 for zero-fill years.
  - 157 null values: Impute with crop×country mean anomaly over available years.

Input:  outputs/fce/yield_anomaly_by_country_crop_year.csv
Output: outputs/fce/yield_anomaly_by_country_crop_year.csv  (overwrite)
"""
import os
import numpy as np
import pandas as pd

PROJECT = (
    "./"
    "Other computers/My Laptop/UDel/Taky_research/flood_food_security"
)
FP = os.path.join(PROJECT, "outputs/fce/yield_anomaly_by_country_crop_year.csv")

print("Loading ISIMIP 3a yield anomaly panel ...")
df = pd.read_csv(FP)
print(f"  Shape: {df.shape}  |  Years: {df['year'].min()}-{df['year'].max()}")
print(f"  Null anomalies: {df['yield_anomaly_ensemble_mean'].isna().sum()}")

# ── Step 1: Impute 157 null values with crop×country mean ────────────────────
print("\n[1] Imputing null anomalies with crop×country mean ...")
means = (
    df.groupby(["country", "crop"])["yield_anomaly_ensemble_mean"]
    .mean()
    .rename("mean_anom")
)
df = df.join(means, on=["country", "crop"])
null_mask = df["yield_anomaly_ensemble_mean"].isna()
print(f"  Null rows to impute: {null_mask.sum()}")
df.loc[null_mask, "yield_anomaly_ensemble_mean"] = df.loc[null_mask, "mean_anom"]
# For any country-crop with ALL values null → fill with 0
still_null = df["yield_anomaly_ensemble_mean"].isna()
if still_null.sum() > 0:
    print(f"  Still null after crop×country mean: {still_null.sum()} → filled with 0")
    df.loc[still_null, "yield_anomaly_ensemble_mean"] = 0.0
df = df.drop(columns=["mean_anom"])
print(f"  Nulls remaining: {df['yield_anomaly_ensemble_mean'].isna().sum()}")

# ── Step 2: Add missing_yield_flag column (0 for 3a years, will be 1 for ext) ─
df["missing_yield_flag"] = 0

# ── Step 3: Extend to 2017-2021 with 0 + flag=1 ─────────────────────────────
print("\n[2] Extending panel from 2017 to 2021 ...")
extension_years = [2017, 2018, 2019, 2020, 2021]
countries = df["country"].unique()
crops = df["crop"].unique()
print(f"  Extending {len(countries)} countries × {len(crops)} crops × {len(extension_years)} years")

ext_records = []
for yr in extension_years:
    for country in countries:
        for crop in crops:
            ext_records.append({
                "country": country,
                "crop": crop,
                "year": yr,
                "yield_anomaly_ensemble_mean": 0.0,
                "missing_yield_flag": 1,
            })
ext_df = pd.DataFrame(ext_records)
print(f"  Extension rows: {len(ext_df):,}")

# ── Step 4: Combine and save ──────────────────────────────────────────────────
combined = pd.concat([df, ext_df], ignore_index=True)
combined = combined.sort_values(["country", "crop", "year"]).reset_index(drop=True)

print(f"\nFinal panel: {combined.shape}")
print(f"  Years: {combined['year'].min()}-{combined['year'].max()}")
print(f"  Nulls: {combined['yield_anomaly_ensemble_mean'].isna().sum()}")
print(f"  Flag=1 (zero-filled extension) rows: {(combined['missing_yield_flag']==1).sum():,}")
print(f"  Countries: {combined['country'].nunique()}, Crops: {combined['crop'].nunique()}")

combined.to_csv(FP, index=False)
print(f"\nSaved → {FP}")
