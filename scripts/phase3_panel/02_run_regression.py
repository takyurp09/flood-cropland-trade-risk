"""
Phase 3 Panel — Script 02
Main Regression Analysis

Estimates the pass-through coefficient β₁ from Flood Trade Exposure (FTE)
to log caloric import availability, using two-way fixed effects (country + year)
with two-way clustered standard errors.

Specification (paper_plan_optionA_v4.md §5.4):
  ΔLog_caloric_imports(i,t) = α(i) + γ(t)
                               + β₁ FTE(i,t)
                               + β₂ MEI(t)
                               + β₃ [MEI(t) × Region(i)]
                               + β₄ DomClimate(i,t)
                               + β₅ GTA_restriction(j,t)
                               + ε(i,t)

Standard errors: two-way clustered by (importing country, year)

Output: outputs/regression/
  results_main.csv          — coefficient table across 4 specs
  results_by_commodity.csv  — β₁ per crop
  results_by_income.csv     — β₁ by importer income group (placeholder)
  regression_summary.txt    — human-readable summary

Usage:
    python 02_run_regression.py
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
PANEL_PATH = os.path.join(PROJECT, "outputs", "regression", "regression_panel.csv")
OUT_DIR    = os.path.join(PROJECT, "outputs", "regression")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load panel ────────────────────────────────────────────────────────────────
print("Loading regression panel …")
panel = pd.read_csv(PANEL_PATH)

# Primary outcome: log of BACI-based caloric import volume (most complete, all 5 crops)
# Secondary: log of total food supply kcal/pc/day from FAO FBS (robustness)

# Use level of log_kcal_imports as primary outcome (not first-difference yet)
# Country + year FE will absorb structural trends
panel["y_main"] = panel["log_kcal_imports"]        # BACI caloric imports
panel["y_fbs"]  = panel["log_food_supply_kcal_pc_day"]  # FAO FBS food supply

# FTE variable: caloric-share–weighted average across crops
panel["fte"] = panel["fte_weighted_total"]

# ENSO control: use ONI (preferred) or MEI
ENSO_VAR = "oni_annual_mean"  # can switch to mei_annual_mean

# Region dummies for ENSO interactions (already in panel)
REGION_COLS = ["region_south_asia", "region_southeast_asia", "region_ssa",
               "region_central_america", "region_east_africa"]
ENSO_INTERACT_COLS = ["enso_x_south_asia", "enso_x_southeast_asia", "enso_x_ssa",
                      "enso_x_central_america", "enso_x_east_africa"]

# Domestic climate controls
CLIMATE_COLS = ["t2m_anom_C", "tp_anom_frac"]

# GTA control
GTA_COL = "gta_weighted_restriction"

# ── Sample selection ──────────────────────────────────────────────────────────
# Keep only observations with FTE and the primary outcome
required = ["y_main", "fte", "year", "iso3"]
df = panel.dropna(subset=required).copy()
df = df[df["year"].between(2000, 2021)].copy()

print(f"Full sample: {len(df):,} country-year obs | {df['iso3'].nunique()} countries")

# ── Helper: run TWFE regression ───────────────────────────────────────────────
try:
    from linearmodels.panel import PanelOLS
    USE_LINEARMODELS = True
    print("Using linearmodels.panel.PanelOLS (two-way clustered SE)")
except ImportError:
    USE_LINEARMODELS = False
    print("WARNING: linearmodels not found — using statsmodels OLS as fallback")

import statsmodels.formula.api as smf


def run_twfe(df_fit, y_col, x_cols, label=""):
    """
    Two-way FE regression with linearmodels or statsmodels fallback.
    Returns dict with coefficient table.
    """
    fit_df = df_fit.dropna(subset=[y_col] + x_cols).copy()
    fit_df = fit_df.sort_values(["iso3", "year"])

    if USE_LINEARMODELS:
        fit_df = fit_df.set_index(["iso3", "year"])
        x = fit_df[x_cols]
        y = fit_df[[y_col]]

        try:
            mod = PanelOLS(
                y, x,
                entity_effects=True,
                time_effects=True,
                drop_absorbed=True,
            )
            res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
            coefs = pd.DataFrame({
                "variable"  : res.params.index,
                "coef"      : res.params.values,
                "se"        : res.std_errors.values,
                "tstat"     : res.tstats.values,
                "pvalue"    : res.pvalues.values,
                "ci_lo"     : res.params.values - 1.96 * res.std_errors.values,
                "ci_hi"     : res.params.values + 1.96 * res.std_errors.values,
                "nobs"      : res.nobs,
                "r2_within" : res.rsquared,
                "specification": label,
            })
            return coefs, res
        except Exception as e:
            print(f"  linearmodels error ({label}): {e}. Falling back to statsmodels.")

    # statsmodels fallback with absorbed fixed effects (within transformation)
    fit_df = fit_df.reset_index() if "iso3" not in fit_df.columns else fit_df

    # Within transform (demean by entity and time, add overall mean back)
    for col in [y_col] + x_cols:
        if col not in fit_df.columns:
            continue
        grand_mean = fit_df[col].mean()
        entity_mean = fit_df.groupby("iso3")[col].transform("mean")
        time_mean   = fit_df.groupby("year")[col].transform("mean")
        fit_df[f"_w_{col}"] = fit_df[col] - entity_mean - time_mean + grand_mean

    w_y  = f"_w_{y_col}"
    w_xs = [f"_w_{c}" for c in x_cols if f"_w_{c}" in fit_df.columns]
    formula = f"{w_y} ~ {' + '.join(w_xs)} - 1"
    res_sm = smf.ols(formula, data=fit_df).fit(
        cov_type="cluster", cov_kwds={"groups": fit_df["iso3"]}
    )
    # Extract results
    params = res_sm.params.rename(
        lambda x: x.replace("_w_", "") if x.startswith("_w_") else x
    )
    se = res_sm.bse.rename(
        lambda x: x.replace("_w_", "") if x.startswith("_w_") else x
    )
    pvals = res_sm.pvalues.rename(
        lambda x: x.replace("_w_", "") if x.startswith("_w_") else x
    )
    coefs = pd.DataFrame({
        "variable"  : params.index,
        "coef"      : params.values,
        "se"        : se.values,
        "tstat"     : (params / se).values,
        "pvalue"    : pvals.values,
        "ci_lo"     : params.values - 1.96 * se.values,
        "ci_hi"     : params.values + 1.96 * se.values,
        "nobs"      : res_sm.nobs,
        "r2_within" : res_sm.rsquared,
        "specification": label,
    })
    return coefs, res_sm


# ── SPECIFICATION LOOP (Figure 3 Panel A) ─────────────────────────────────────
print("\n=== Estimating main specifications ===")

specs = {
    "S1_naive": {
        "label": "1. Naive (no FE, no controls)",
        "fe": False,
        # Include ONI directly (no year FE to absorb it)
        "controls": [ENSO_VAR],
    },
    "S2_year_fe": {
        "label": "2. Year FE only",
        "fe": "year_only",
        # Year FEs absorb global ONI — drop it, keep only regional interactions
        "controls": ENSO_INTERACT_COLS,
    },
    "S3_twfe_enso": {
        "label": "3. TWFE + ENSO regional interactions",
        "fe": "both",
        # Year FEs absorb global ONI — keep only cross-sectional regional ENSO interactions
        "controls": ENSO_INTERACT_COLS,
    },
    "S4_full": {
        "label": "4. TWFE + ENSO + DomClimate + GTA",
        "fe": "both",
        "controls": ENSO_INTERACT_COLS + CLIMATE_COLS + [GTA_COL],
    },
}

all_coefs = []

for spec_key, spec in specs.items():
    x_cols = ["fte"] + spec["controls"]
    # Filter to rows with all required variables
    required_cols = ["y_main"] + x_cols
    df_spec = df.dropna(subset=required_cols).copy()

    if spec["fe"] == False:
        # Pure OLS, no FE
        formula = f"y_main ~ {' + '.join(x_cols)}"
        try:
            res = smf.ols(formula, data=df_spec).fit(
                cov_type="cluster", cov_kwds={"groups": df_spec["iso3"]}
            )
            coefs = pd.DataFrame({
                "variable"  : res.params.index,
                "coef"      : res.params.values,
                "se"        : res.bse.values,
                "tstat"     : res.tvalues.values,
                "pvalue"    : res.pvalues.values,
                "ci_lo"     : res.params.values - 1.96 * res.bse.values,
                "ci_hi"     : res.params.values + 1.96 * res.bse.values,
                "nobs"      : res.nobs,
                "r2_within" : res.rsquared,
                "specification": spec["label"],
            })
            all_coefs.append(coefs[coefs["variable"] == "fte"])
        except Exception as e:
            print(f"  {spec_key} ERROR: {e}")
        continue

    elif spec["fe"] == "year_only":
        # Add year dummies to control list, no entity FE
        df_spec = pd.get_dummies(df_spec, columns=["year"], prefix="yr", drop_first=True)
        yr_dummies = [c for c in df_spec.columns if c.startswith("yr_")]
        x_full = x_cols + yr_dummies
        formula = f"y_main ~ {' + '.join(x_full)}"
        try:
            res = smf.ols(formula, data=df_spec).fit(
                cov_type="cluster", cov_kwds={"groups": df_spec["iso3"]}
            )
            coefs = pd.DataFrame({
                "variable"  : res.params.index,
                "coef"      : res.params.values,
                "se"        : res.bse.values,
                "tstat"     : res.tvalues.values,
                "pvalue"    : res.pvalues.values,
                "ci_lo"     : res.params.values - 1.96 * res.bse.values,
                "ci_hi"     : res.params.values + 1.96 * res.bse.values,
                "nobs"      : res.nobs,
                "r2_within" : res.rsquared,
                "specification": spec["label"],
            })
            all_coefs.append(coefs[coefs["variable"] == "fte"])
        except Exception as e:
            print(f"  {spec_key} ERROR: {e}")
        continue

    else:
        # Two-way FE with linearmodels
        coefs, res = run_twfe(df_spec, "y_main", x_cols, spec["label"])
        fte_row = coefs[coefs["variable"] == "fte"]
        all_coefs.append(fte_row)

    # Print summary
    fte_info = coefs[coefs["variable"] == "fte"]
    if not fte_info.empty:
        r = fte_info.iloc[0]
        stars = "***" if r["pvalue"] < 0.01 else ("**" if r["pvalue"] < 0.05 else ("*" if r["pvalue"] < 0.1 else ""))
        nobs = int(r["nobs"])
        r2 = r["r2_within"]
        print(f"  {spec['label']}")
        print(f"    β₁(FTE) = {r['coef']:.4f} ({r['se']:.4f}) {stars}  [p={r['pvalue']:.3f}, N={nobs:,}, R²={r2:.4f}]")

# Save main specification table
main_table = pd.concat(all_coefs, ignore_index=True)
main_table.to_csv(os.path.join(OUT_DIR, "results_main.csv"), index=False)
print(f"\nSaved: outputs/regression/results_main.csv")

# ── BY COMMODITY (Figure 3 Panel B) ──────────────────────────────────────────
print("\n=== Estimating by commodity (full specification) ===")

CROPS = ["maize", "rice", "soybeans", "veg_oils", "wheat"]
# TWFE absorbs global ONI — use only regional interaction terms for ENSO
CTRL = ENSO_INTERACT_COLS + CLIMATE_COLS + [GTA_COL]
crop_coefs = []

for crop in CROPS:
    crop_fte_col = f"fte_{crop}"
    if crop_fte_col not in panel.columns:
        print(f"  {crop}: column {crop_fte_col} missing")
        continue
    panel[f"_fte_crop"] = panel[crop_fte_col]
    x_cols = ["_fte_crop"] + CTRL
    df_crop = panel.dropna(subset=["y_main"] + x_cols).copy()
    if len(df_crop) < 200:
        print(f"  {crop}: too few obs ({len(df_crop)})")
        continue
    coefs, _ = run_twfe(df_crop, "y_main", x_cols, label=crop)
    row = coefs[coefs["variable"] == "_fte_crop"].copy()
    if not row.empty:
        row["variable"] = crop
        crop_coefs.append(row)
        r = row.iloc[0]
        stars = "***" if r["pvalue"] < 0.01 else ("**" if r["pvalue"] < 0.05 else ("*" if r["pvalue"] < 0.1 else ""))
        print(f"  {crop:12s}: β₁ = {r['coef']:.4f} ({r['se']:.4f}) {stars}  [p={r['pvalue']:.3f}, N={int(r['nobs']):,}]")

if crop_coefs:
    crop_table = pd.concat(crop_coefs, ignore_index=True)
    crop_table.to_csv(os.path.join(OUT_DIR, "results_by_commodity.csv"), index=False)
    print(f"Saved: outputs/regression/results_by_commodity.csv")

# ── BY INCOME GROUP (Figure 3 Panel C / Figure 6) ────────────────────────────
print("\n=== Estimating by importer income group ===")
INCOME_GROUPS = ["LIC", "LMC", "UMC", "HIC"]
inc_coefs = []

for grp in INCOME_GROUPS:
    grp_col = f"inc_{grp}"
    if grp_col not in panel.columns:
        continue
    df_grp = panel[panel[grp_col] == 1].dropna(subset=["y_main", "fte"] + CTRL).copy()
    if len(df_grp) < 100 or df_grp["iso3"].nunique() < 5:
        print(f"  {grp}: too few obs ({len(df_grp)}) — skip")
        continue
    coefs_grp, _ = run_twfe(df_grp, "y_main", ["fte"] + CTRL, label=grp)
    row = coefs_grp[coefs_grp["variable"] == "fte"].copy()
    if not row.empty:
        row["income_group"] = grp
        row["n_countries"] = df_grp["iso3"].nunique()
        inc_coefs.append(row)
        r = row.iloc[0]
        stars = "***" if r["pvalue"] < 0.01 else ("**" if r["pvalue"] < 0.05 else ("*" if r["pvalue"] < 0.1 else ""))
        print(f"  {grp} (N_ctry={df_grp['iso3'].nunique()!r}) β₁={r['coef']:.4f} ({r['se']:.4f}) {stars}  [p={r['pvalue']:.3f}]")

if inc_coefs:
    inc_table = pd.concat(inc_coefs, ignore_index=True)
    inc_table.to_csv(os.path.join(OUT_DIR, "results_by_income.csv"), index=False)
    print(f"Saved: outputs/regression/results_by_income.csv")

# ── BY HHI TERCILE (trade concentration heterogeneity) ───────────────────────
print("\n=== Estimating by import concentration tercile ===")
hhi_coefs = []

panel_hhi = panel.dropna(subset=["hhi_import_concentration"]).copy()
q33 = panel_hhi["hhi_import_concentration"].quantile(0.33)
q67 = panel_hhi["hhi_import_concentration"].quantile(0.67)
panel_hhi["hhi_tercile"] = pd.cut(
    panel_hhi["hhi_import_concentration"],
    bins=[-np.inf, q33, q67, np.inf],
    labels=["Low_HHI", "Mid_HHI", "High_HHI"]
)
for tercile in ["Low_HHI", "Mid_HHI", "High_HHI"]:
    df_t = panel_hhi[panel_hhi["hhi_tercile"] == tercile].dropna(subset=["y_main", "fte"] + CTRL).copy()
    if len(df_t) < 100:
        continue
    coefs_t, _ = run_twfe(df_t, "y_main", ["fte"] + CTRL, label=tercile)
    row = coefs_t[coefs_t["variable"] == "fte"].copy()
    if not row.empty:
        row["hhi_group"] = tercile
        hhi_coefs.append(row)
        r = row.iloc[0]
        stars = "***" if r["pvalue"] < 0.01 else ("**" if r["pvalue"] < 0.05 else ("*" if r["pvalue"] < 0.1 else ""))
        print(f"  {tercile} (median HHI ≈ {df_t['hhi_import_concentration'].median():.3f}): β₁={r['coef']:.4f} ({r['se']:.4f}) {stars}  [p={r['pvalue']:.3f}]")

if hhi_coefs:
    hhi_table = pd.concat(hhi_coefs, ignore_index=True)
    hhi_table.to_csv(os.path.join(OUT_DIR, "results_by_hhi.csv"), index=False)
    print(f"Saved: outputs/regression/results_by_hhi.csv")

# ── ROBUSTNESS: FAO FBS food supply as outcome ──────────────────────────────
print("\n=== Robustness: FAO FBS food supply outcome ===")
df_fbs = panel.dropna(subset=["y_fbs", "fte"] + CTRL).copy()
print(f"  FAO FBS sample: {len(df_fbs):,} obs | {df_fbs['iso3'].nunique()} countries")
if len(df_fbs) > 200:
    x_cols_full = ["fte"] + ENSO_INTERACT_COLS + CLIMATE_COLS + [GTA_COL]
    coefs_fbs, _ = run_twfe(df_fbs, "y_fbs", x_cols_full, label="FAO_FBS_food_supply")
    fte_fbs = coefs_fbs[coefs_fbs["variable"] == "fte"]
    if not fte_fbs.empty:
        r = fte_fbs.iloc[0]
        stars = "***" if r["pvalue"] < 0.01 else ("**" if r["pvalue"] < 0.05 else ("*" if r["pvalue"] < 0.1 else ""))
        print(f"  β₁(FTE→FoodSupply) = {r['coef']:.4f} ({r['se']:.4f}) {stars}  [p={r['pvalue']:.3f}]")
    coefs_fbs.to_csv(os.path.join(OUT_DIR, "results_fbs_robustness.csv"), index=False)
    print(f"  Saved: outputs/regression/results_fbs_robustness.csv")

# ── HUMAN-READABLE SUMMARY ───────────────────────────────────────────────────
summary_path = os.path.join(OUT_DIR, "regression_summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 72 + "\n")
    f.write("Flood Trade Exposure → Caloric Import Availability\n")
    f.write("Main Regression Results\n")
    f.write("=" * 72 + "\n\n")
    f.write("Dependent variable: Log(caloric import volume, kt equivalent) [BACI]\n")
    f.write("Identification: Two-way FE (country + year), two-way clustered SE\n")
    f.write("Sample: 2000–2021, 229 importers (varies by specification)\n\n")
    f.write("β₁ Coefficient on FTE (Flood Trade Exposure) across specifications:\n")
    f.write("-" * 72 + "\n")
    if not main_table.empty:
        for _, row in main_table.iterrows():
            stars = "***" if row["pvalue"] < 0.01 else ("**" if row["pvalue"] < 0.05 else ("*" if row["pvalue"] < 0.1 else "   "))
            f.write(f"  {row['specification'][:50]:<50} β₁={row['coef']:+.4f} ({row['se']:.4f}){stars}\n")
    f.write("\n*** p<0.01  ** p<0.05  * p<0.1\n\n")
    f.write("Interpretation:\n")
    f.write("  A 1-unit increase in FTE (i.e., if 100% of food imports come from\n")
    f.write("  a country with 100% of cropland flooded) is associated with a\n")
    f.write("  β₁-unit change in log caloric import volume.\n\n")
    f.write("FCE construction: DFO proxy flood × MapSPAM crop area share\n")
    f.write("FTE: Σⱼ BACI trade share(i←j,crop,t) × FCE(j,crop,t), caloric-weighted\n")
    f.write("ENSO control: ONI annual mean × 5 regional indicators\n")
    f.write("Domestic climate: ERA5 growing-season T2M and TP anomalies\n")
    f.write("GTA: Trade-weighted caloric-share export restriction index\n")

print(f"\nSaved: outputs/regression/regression_summary.txt")
print("\n=== Done ===")
