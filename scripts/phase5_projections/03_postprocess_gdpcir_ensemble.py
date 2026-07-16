"""
03_postprocess_gdpcir_ensemble.py
Post-process raw per-GCM GDPCIR output into final ensemble mean/std panel.

Run AFTER 02_fetch_esgf_gdpcir.py completes all GCMs (GFDL-ESM4, MPI-ESM1-2-HR, UKESM1-0-LL).
Can also run mid-run to get partial ensemble with however many GCMs are done.

Input:  outputs/projections/gdpcir_enso_teleconnection_panel.csv
  Columns (possibly mixed from different script versions):
    gcm, scenario, variable, region, year, month, ensemble_mean [or value], [ensemble_std]

Output: outputs/projections/gdpcir_enso_teleconnection_panel.csv
  Columns: scenario, variable, region, year, month, n_gcms, ensemble_mean, ensemble_std, gcms_used
"""
import os
import numpy as np
import pandas as pd

PROJECT = (
    "./"
    "Other computers/My Laptop/UDel/Taky_research/flood_food_security"
)
RAW_PATH   = os.path.join(PROJECT, "outputs/projections/gdpcir_enso_teleconnection_panel.csv")
FINAL_PATH = os.path.join(PROJECT, "outputs/projections/gdpcir_ensemble_final.csv")

if not os.path.exists(RAW_PATH):
    print(f"ERROR: {RAW_PATH} not found. Run 02_fetch_esgf_gdpcir.py first.")
    raise SystemExit(1)

print(f"Loading raw GDPCIR panel: {RAW_PATH}")
raw = pd.read_csv(RAW_PATH, low_memory=False)
print(f"  Shape: {raw.shape}")
print(f"  Columns: {list(raw.columns)}")
print(f"  GCMs: {raw['gcm'].unique()}")
print(f"  Scenarios: {raw['scenario'].unique()}")

# ── Normalize: use whichever value column is present ─────────────────────────
if "ensemble_mean" in raw.columns and "value" not in raw.columns:
    raw = raw.rename(columns={"ensemble_mean": "value"})
elif "value" in raw.columns and "ensemble_mean" in raw.columns:
    # Prefer 'value'; fill missing from 'ensemble_mean'
    raw["value"] = raw["value"].fillna(raw["ensemble_mean"])

# Drop old ensemble_std (will recompute)
if "ensemble_std" in raw.columns:
    raw = raw.drop(columns=["ensemble_std"])

# Keep only relevant columns
raw = raw[["gcm", "scenario", "variable", "region", "year", "month", "value"]].copy()
raw = raw.dropna(subset=["value"])

print(f"\nAfter normalization: {raw.shape}")
print(f"  GCMs in file: {sorted(raw['gcm'].unique())}")
print(f"  Rows per GCM:")
for gcm, grp in raw.groupby("gcm"):
    print(f"    {gcm}: {len(grp):,} rows, scenarios: {sorted(grp['scenario'].unique())}")

# ── Compute ensemble mean and std across GCMs ─────────────────────────────────
print("\nComputing ensemble statistics across GCMs ...")
ens = (
    raw.groupby(["scenario", "variable", "region", "year", "month"])
    .agg(
        ensemble_mean=("value", "mean"),
        ensemble_std=("value", "std"),
        n_gcms=("gcm", "nunique"),
        gcms_used=("gcm", lambda x: "|".join(sorted(x.unique()))),
    )
    .reset_index()
)

# For single-GCM combos, std will be NaN — fill with 0 but flag
n_single = (ens["n_gcms"] == 1).sum()
if n_single > 0:
    print(f"  {n_single:,} rows have only 1 GCM (std=NaN → 0 for those)")
    ens["ensemble_std"] = ens["ensemble_std"].fillna(0.0)

print(f"\nEnsemble panel: {ens.shape}")
print(f"  Scenarios: {sorted(ens['scenario'].unique())}")
print(f"  Variables: {sorted(ens['variable'].unique())}")
print(f"  Regions: {sorted(ens['region'].unique())}")
print(f"  Years: {ens['year'].min()}–{ens['year'].max()}")
print(f"  GCM combos (n_gcms): {ens['n_gcms'].value_counts().to_dict()}")
print(f"  ensemble_std null: {ens['ensemble_std'].isna().sum()}")

# Sample output
print("\nSample rows:")
print(ens.head(5).to_string(index=False))

ens.to_csv(FINAL_PATH, index=False)
print(f"\nSaved ensemble panel → {FINAL_PATH}")
print(f"  ({len(ens):,} rows)")
