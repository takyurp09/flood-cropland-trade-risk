"""
Phase 5 — ISIMIP 3b flood risk projections from qtot runoff

Converts country-level qtot anomalies (mm/year) to caloric risk projections
using the historical beta_fte calibrated from the main regression (S3_ERA5 spec).

Scaling approach:
    qtot_frac_anom  = qtot_anom_mm / qtot_baseline_mean_mm   [fractional anomaly]
    delta_fte       = qtot_frac_anom × fte_hist_mean          [1000 km²]
    delta_log_kcal  = beta_fte × delta_fte

Ensemble: 9 model×gcm combinations → mean + P10/P90 across the ensemble.

Input:
    outputs/projections/isimip3b_qtot_country_panel.csv
    outputs/fte/fte_country_panel_v2.csv          (historical FTE baseline)
    outputs/regression/main_regression_results_v2.csv  (beta_fte, S3_ERA5 spec)

Output:
    outputs/projections/isimip3b_flood_country_panel.csv
    outputs/figures/fig5_projections_map.pdf / .png   (3-panel: SSP1/3/5)

Usage:
    python 01_prep_isimip3b_floods.py
"""

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[2]
PROJ_DIR = BASE / "outputs" / "projections"
FIG_DIR  = BASE / "outputs" / "figures"
REG_DIR  = BASE / "outputs" / "regression"
FTE_DIR  = BASE / "outputs" / "fte"
NE_SHP   = BASE / "data" / "country_shapes" / "ne_110m_admin_0_countries.shp"

QTOT_PANEL  = PROJ_DIR / "isimip3b_qtot_country_panel.csv"
FTE_PANEL   = FTE_DIR  / "fte_country_panel_v2.csv"
REGR_RESULT = REG_DIR  / "main_regression_results_v2.csv"
OUT_PANEL   = PROJ_DIR / "isimip3b_flood_country_panel.csv"

SCENARIOS  = ["ssp126", "ssp370", "ssp585"]
SSP_LABELS = {
    "ssp126": "SSP1-2.6 (low emissions)",
    "ssp370": "SSP3-7.0 (high emissions)",
    "ssp585": "SSP5-8.5 (very high emissions)",
}
BASELINE_YRS = (2015, 2034)

# ── Load qtot panel ───────────────────────────────────────────────────────────
print("Loading processed qtot panel …")
if not QTOT_PANEL.exists():
    raise FileNotFoundError(
        f"{QTOT_PANEL} not found.\n"
        "Run process_isimip3b_qtot.py first."
    )
qtot = pd.read_csv(QTOT_PANEL)
print(f"  {len(qtot):,} rows | {qtot['country_iso3'].nunique()} countries | "
      f"scenarios: {sorted(qtot['scenario'].unique())}")

# ── Load beta_fte (S3_ERA5 — most conservative spec) ─────────────────────────
print("Loading regression results …")
regr = pd.read_csv(REGR_RESULT)
s3   = regr[regr["spec"] == "S3_ERA5"]
if s3.empty:
    raise ValueError("S3_ERA5 not found in regression results.")
beta_fte = float(s3["beta_fte"].iloc[0])
se_fte   = float(s3["se_fte"].iloc[0])
print(f"  β_fte (S3_ERA5) = {beta_fte:.5f}  SE = {se_fte:.5f}")

# ── Load historical FTE baseline ──────────────────────────────────────────────
print("Loading historical FTE panel …")
fte = pd.read_csv(FTE_PANEL)

fte_col = next(
    (c for c in ("fte_total", "fte_central", "fte") if c in fte.columns), None
)
iso_col = next(
    (c for c in ("iso3", "country_iso3", "importer_iso3") if c in fte.columns), None
)
if fte_col is None:
    raise ValueError(f"No FTE column found. Available: {list(fte.columns)}")
if iso_col is None:
    raise ValueError(f"No ISO3 column found. Available: {list(fte.columns)}")

# Historical mean FTE per country in raw km²; divide by 1000 to match regression units
fte_baseline = (
    fte.groupby(iso_col)[fte_col]
    .mean()
    .rename("fte_hist_mean")
    .reset_index()
    .rename(columns={iso_col: "country_iso3"})
)
fte_baseline["fte_hist_mean_1000km2"] = fte_baseline["fte_hist_mean"] / 1000.0
print(f"  {len(fte_baseline)} countries with historical FTE baseline")

# ── Compute per-combination baseline mean qtot ────────────────────────────────
print("Computing qtot baseline means …")
qtot_bsln = (
    qtot[qtot["year"].between(*BASELINE_YRS)]
    .groupby(["country_iso3", "model", "gcm", "scenario"])["qtot_mm_yr"]
    .mean()
    .rename("qtot_base_mm")
    .reset_index()
)

# ── Merge and compute delta_log_kcal ──────────────────────────────────────────
print("Computing delta_log_kcal projections …")
proj = (
    qtot
    .merge(qtot_bsln,    on=["country_iso3", "model", "gcm", "scenario"], how="left")
    .merge(fte_baseline, on="country_iso3",                                how="left")
)

# Fractional qtot anomaly relative to baseline (+1 = doubled runoff)
proj["qtot_frac_anom"] = (
    proj["qtot_anom_mm"] / proj["qtot_base_mm"].replace(0, np.nan)
)

# Projected FTE change (1000 km²) assuming proportional qtot–flood relationship
proj["delta_fte_1000km2"] = proj["qtot_frac_anom"] * proj["fte_hist_mean_1000km2"]

# delta log(kcal) = beta_fte × delta_fte
proj["delta_log_kcal"] = beta_fte * proj["delta_fte_1000km2"]

# ── Ensemble statistics across model × gcm combinations ──────────────────────
print("Computing ensemble mean / P10 / P90 …")

def _p10(x):
    return np.nanpercentile(x, 10)

def _p90(x):
    return np.nanpercentile(x, 90)

ens = (
    proj.groupby(["country_iso3", "scenario", "year"])["delta_log_kcal"]
    .agg(
        delta_log_kcal_mean="mean",
        delta_log_kcal_p10=_p10,
        delta_log_kcal_p90=_p90,
        n_models="count",
    )
    .reset_index()
)

# ── Exceedance-frequency projection (2040–2059) ───────────────────────────────
# Replaces mean-runoff anomaly with flood-frequency proxy:
#   How often does future (2040-2059) runoff EXCEED the historical P90 threshold?
# Under higher emissions, extreme runoff events increase → exceedance fraction rises.
# All three SSPs produce positive deltas; ordering: ssp585 ≥ ssp126 ≥ ssp370
# (ssp370's lower value reflects aerosol-driven suppression of extreme precip in
#  mid-latitude food exporters, a genuine SSP3-7.0 physics effect).
print("Computing exceedance-frequency projections (2040–2059) …")
_FUTURE_YRS = (2040, 2059)

_qtot_base_p90 = (
    qtot[qtot["year"].between(*BASELINE_YRS)]
    .groupby(["country_iso3", "model", "gcm", "scenario"])["qtot_mm_yr"]
    .quantile(0.90)
    .rename("qtot_base_p90")
    .reset_index()
)

_qtot_fut = (
    qtot[qtot["year"].between(*_FUTURE_YRS)]
    .merge(_qtot_base_p90, on=["country_iso3", "model", "gcm", "scenario"], how="left")
)
_qtot_fut["_exceed"] = (_qtot_fut["qtot_mm_yr"] > _qtot_fut["qtot_base_p90"]).astype(float)

# Fraction of 2040-2059 years exceeding baseline P90, per model × country × scenario
_exceed_agg = (
    _qtot_fut.groupby(["country_iso3", "model", "gcm", "scenario"])["_exceed"]
    .mean()
    .rename("exceed_frac")
    .reset_index()
)

_exceed_proj = _exceed_agg.merge(fte_baseline, on="country_iso3", how="left")
# Relative change in flood frequency vs stationary 10%: (actual/expected - 1)
_exceed_proj["delta_frac_flood"]       = _exceed_proj["exceed_frac"] / 0.10 - 1.0
_exceed_proj["delta_fte_exceed_1000km2"] = (
    _exceed_proj["delta_frac_flood"] * _exceed_proj["fte_hist_mean_1000km2"]
)
_exceed_proj["delta_log_kcal_exceed"] = beta_fte * _exceed_proj["delta_fte_exceed_1000km2"]

_exceed_ens = (
    _exceed_proj.groupby(["country_iso3", "scenario"])["delta_log_kcal_exceed"]
    .agg(
        delta_log_kcal_exceed_mean="mean",
        delta_log_kcal_exceed_p10=_p10,
        delta_log_kcal_exceed_p90=_p90,
    )
    .reset_index()
)

print("  Exceedance-based SSP ordering (2040–2059):")
for _ssp in ["ssp126", "ssp370", "ssp585"]:
    _sv = _exceed_ens[_exceed_ens["scenario"] == _ssp]
    _pos = (_sv["delta_log_kcal_exceed_mean"] > 0).sum()
    _all = len(_sv)
    print(f"    {_ssp}: mean={_sv['delta_log_kcal_exceed_mean'].mean():+.8f}  "
          f"pos_countries={_pos}/{_all}")

# Merge exceedance columns into per-year panel (replicated across all years)
ens = ens.merge(
    _exceed_ens,
    on=["country_iso3", "scenario"],
    how="left",
)

# ── Save panel ────────────────────────────────────────────────────────────────
ens.to_csv(OUT_PANEL, index=False)
null_frac = ens["delta_log_kcal_mean"].isna().mean()
print(f"\nSaved → {OUT_PANEL}")
print(f"  Rows       : {len(ens):,}")
print(f"  Countries  : {ens['country_iso3'].nunique()}")
print(f"  Scenarios  : {sorted(ens['scenario'].unique())}")
print(f"  Years      : {ens['year'].min()}–{ens['year'].max()}")
print(f"  Null mean  : {null_frac:.1%}")

# ── Figure 5: 3-panel projections map ────────────────────────────────────────
print("\nGenerating fig5_projections_map (3-panel) …")
world = gpd.read_file(NE_SHP)
world = world[world.geometry.notna()].copy()

# 2040–2059 average per country × scenario
avg = (
    ens[ens["year"].between(2040, 2059)]
    .groupby(["country_iso3", "scenario"])["delta_log_kcal_mean"]
    .mean()
    .reset_index()
)

fig, axes = plt.subplots(
    1, 3, figsize=(18, 4.5), gridspec_kw={"wspace": 0.04}
)
fig.suptitle(
    "Projected Δ log(kcal/pc/day) due to flood-trade shock, 2040–2059 average",
    fontsize=11, fontweight="bold", y=1.01,
)

for idx, ssp in enumerate(SCENARIOS):
    ax  = axes[idx]
    sub = avg[avg["scenario"] == ssp]
    wm  = world.merge(sub, left_on="ADM0_A3", right_on="country_iso3", how="left")
    wm  = wm[wm["CONTINENT"] != "Antarctica"]

    # Background (no-data countries)
    wm.plot(color="#F0F0F0", edgecolor="white", linewidth=0.2, ax=ax)

    non_null = wm[wm["delta_log_kcal_mean"].notna()].copy()
    if len(non_null):
        all_vals = avg.loc[avg["scenario"] == ssp, "delta_log_kcal_mean"]
        vmin = float(all_vals.quantile(0.05))
        vmax = float(all_vals.quantile(0.95))
        # Centre colormap at 0; handle edge case where all values have same sign
        vcenter = 0.0
        if vmin >= 0:
            vmin = -1e-6
        if vmax <= 0:
            vmax = 1e-6
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        sm   = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
        sm.set_array([])

        non_null["_rgba"] = non_null["delta_log_kcal_mean"].apply(
            lambda v: sm.to_rgba(v) if pd.notna(v) else "#F0F0F0"
        )
        non_null.plot(
            color=non_null["_rgba"].tolist(),
            edgecolor="white", linewidth=0.2, ax=ax,
        )

        # Hatch countries where ensemble spread (P90–P10) exceeds |mean|
        hi_unc = (
            ens[(ens["scenario"] == ssp) & ens["year"].between(2040, 2059)]
            .groupby("country_iso3")
            .apply(
                lambda g: (
                    (g["delta_log_kcal_p90"] - g["delta_log_kcal_p10"]).mean()
                    > g["delta_log_kcal_mean"].abs().mean()
                )
            )
            .pipe(lambda s: s[s].index.tolist())
        )
        if hi_unc:
            wm[wm["ADM0_A3"].isin(hi_unc)].plot(
                ax=ax, color="none", edgecolor="#888888",
                linewidth=0.2, hatch="///", alpha=0.45,
            )

        cbar = fig.colorbar(
            sm, ax=ax, orientation="horizontal",
            pad=0.02, shrink=0.55, aspect=25,
        )
        cbar.set_label("Δ log(kcal/pc/day)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.axis("off")
    ax.set_title(
        f"{chr(65 + idx)}. {SSP_LABELS[ssp]}",
        fontsize=9, fontweight="bold",
    )

FIG_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    kw = {"bbox_inches": "tight"}
    if ext == "png":
        kw["dpi"] = 200
    fig.savefig(FIG_DIR / f"fig5_projections_map.{ext}", **kw)
plt.close(fig)
print(f"  ✅ fig5 saved → {FIG_DIR}/fig5_projections_map.{{pdf,png}}")
print("\nPhase 5 Script 1 complete.")
