"""
Phase 5 — Caloric Risk Projections (2030-2080)
1. Apply historical FCE damage weights to ISIMIP projected yield anomalies
2. Compute projected FTE using 2018-2021 average trade shares (baseline)
3. Apply regression β (S3) to projected ΔFTE to estimate caloric risk
4. Aggregate by SSP, decade, and income group
"""

import os, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
BASE     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJ_DIR = os.path.join(BASE, 'outputs', 'projections')
FTE_DIR  = os.path.join(BASE, 'outputs', 'fte')
REG_DIR  = os.path.join(BASE, 'outputs', 'regression')
FCE_DIR  = os.path.join(BASE, 'outputs', 'fce')

# ── 1. Load data (prefer v2 files, fall back to v1) ───────────────────────────
def _load(dir_, v2name, v1name):
    p2 = os.path.join(dir_, v2name)
    p1 = os.path.join(dir_, v1name)
    if os.path.exists(p2):
        print(f"  Loading v2: {v2name}")
        return pd.read_csv(p2), True
    print(f"  Loading v1 (fallback): {v1name}")
    return pd.read_csv(p1), False

isimip = pd.read_csv(os.path.join(PROJ_DIR, 'isimip3b_yield_projection_panel.csv'))
fce_hist, fce_v2 = _load(FCE_DIR, 'fce_final_panel_v2.csv', 'fce_final_panel.csv')
fte_hist, _      = _load(FTE_DIR, 'fte_panel_v2.csv',       'fte_panel.csv')
master,   _      = _load(REG_DIR, 'master_panel_v2.csv',    'master_panel.csv')
reg_results, _   = _load(REG_DIR, 'main_regression_results_v2.csv', 'main_regression_results.csv')

# Normalise column names across v1/v2 FCE schemas
if fce_v2 and 'growing_season_flood_km2' in fce_hist.columns and 'flooded_cropland_km2' not in fce_hist.columns:
    fce_hist = fce_hist.rename(columns={'growing_season_flood_km2': 'flooded_cropland_km2'})

print(f"ISIMIP: {len(isimip):,}r, SSPs: {isimip['ssp'].unique()}, years: {isimip['year'].min()}-{isimip['year'].max()}")
print(f"FCE hist: {len(fce_hist):,}r  (v2={fce_v2})")
print(f"FTE hist: {len(fte_hist):,}r")

# ── 2. Get regression β (preferred spec S3) ───────────────────────────────────
s3 = reg_results[reg_results['spec']=='S3_ERA5']
if len(s3) == 0:
    s3 = reg_results[reg_results['spec']=='S2_ENSO']  # fallback
beta_fte = float(s3['beta_fte'].iloc[0])
se_fte   = float(s3['se_fte'].iloc[0])
print(f"\nUsing β={beta_fte:.4f} (SE={se_fte:.4f}) from {s3['spec'].iloc[0]}")
print(f"  β_lo={beta_fte - 1.96*se_fte:.4f}, β_hi={beta_fte + 1.96*se_fte:.4f}")

# ── 3. Map ISIMIP crops to FCE crops ─────────────────────────────────────────
# ISIMIP: maize, rice_s1, rice_s2, soybeans, spring_wheat, winter_wheat
# FCE:    maize, rice, soybeans, wheat
CROP_MAP = {
    'maize': 'maize',
    'rice_s1': 'rice',
    'rice_s2': 'rice',
    'soybeans': 'soybeans',
    'spring_wheat': 'wheat',
    'winter_wheat': 'wheat',
}
isimip['crop_fce'] = isimip['crop'].map(CROP_MAP)

# For rice and wheat, average across sub-types (s1/s2 or spring/winter)
iso_crop = (isimip.groupby(['iso3','year','ssp','crop_fce'], as_index=False)
            ['yield_anomaly_ensemble'].mean())
iso_crop = iso_crop.rename(columns={'yield_anomaly_ensemble':'yield_anom_proj',
                                    'crop_fce':'crop'})
print(f"\nISIMIP mapped: {len(iso_crop):,}r, crops: {iso_crop['crop'].unique()}")

# ── 4. Historical flood parameters per country-crop ──────────────────────────
# Use 2015-2021 average flooded_cropland_km2 per country-crop as flood exposure baseline
flood_baseline = (fce_hist[fce_hist['year']>=2015]
                  .groupby(['iso3','crop'], as_index=False)
                  .agg(flood_km2_baseline=('flooded_cropland_km2','mean'),
                       harvested_ha_baseline=('harvested_ha','mean')))
flood_baseline['crop_ha_share'] = flood_baseline.groupby('iso3')['harvested_ha_baseline'].transform(
    lambda x: x / x.sum() if x.sum() > 0 else 0
)
print(f"Flood baseline: {len(flood_baseline):,}r")

# ── 5. Compute projected FCE ──────────────────────────────────────────────────
# Damage weight: negative yield anomaly → |anomaly|.clip(0,1); positive → 0
iso_crop['damage_w_proj'] = np.where(
    iso_crop['yield_anom_proj'] < 0,
    iso_crop['yield_anom_proj'].abs().clip(0, 1),
    0.0
)

# Merge with flood baseline
fce_proj = iso_crop.merge(flood_baseline[['iso3','crop','flood_km2_baseline','crop_ha_share']],
                           on=['iso3','crop'], how='left')
fce_proj['flood_km2_baseline'] = fce_proj['flood_km2_baseline'].fillna(0)
fce_proj['crop_ha_share']      = fce_proj['crop_ha_share'].fillna(0)

# FCE projected = flood_km2_baseline × crop_ha_share × damage_w
fce_proj['fce_proj'] = (fce_proj['flood_km2_baseline'] *
                         fce_proj['crop_ha_share'] *
                         fce_proj['damage_w_proj'])

print(f"Projected FCE: {len(fce_proj):,}r, FCE range: {fce_proj['fce_proj'].min():.1f} – {fce_proj['fce_proj'].max():.1f}")

# ── 6. Trade shares (2018-2021 average as fixed baseline) ─────────────────────
fte_base = fte_hist.copy()
# fte_panel has: year, iso3_importer, crop, fte, fte_low, fte_high, ...
# We need trade shares from fce → fte step
# Since fte_panel already encodes FCE×trade_share, compute ratio: fte / fce_exporter
# Simpler: compute FTE_proj directly from projected FCE using historical trade shares

# Get historical FCE per exporter-crop-year to compute shares
fce_exporter = (fce_hist.groupby(['iso3','crop','year'], as_index=False)
                ['fce_central'].sum()
                .rename(columns={'iso3':'iso3_exporter','fce_central':'fce_hist'}))

# Compute weights: share(importer ← exporter) × FCE(exporter)
# From fte_panel, the implied formula is FTE_i = Σ_j w_ij × FCE_j
# We need (share_ij) at baseline. Reconstruct from BACI trade.
# Use the latest 4 years (2018-2021) from fte_panel as the weight matrix proxy:
# FTE_proj(i,c,t) = sum_j [share_ij_baseline × fce_proj(j,c,t)]

# Load BACI for trade shares
BACI_FILE = os.path.join(BASE, 'outputs', 'trade', 'baci_trade_panel.csv')
baci = pd.read_csv(BACI_FILE)
print(f"\nBACE loaded: {len(baci):,}r")

# Filter to baseline period (2018-2021)
baci_base = baci[(baci['year']>=2018)&(baci['year']<=2021)].copy()

# Compute average trade shares by importer-exporter-crop over baseline
shares = (baci_base.groupby(['importer','exporter','commodity'], as_index=False)
          ['quantity_t'].mean())
shares = shares.rename(columns={'commodity':'crop', 'quantity_t':'qty_avg',
                                'importer':'iso3_importer','exporter':'iso3_exporter'})

# Normalize within importer-crop
tot = shares.groupby(['iso3_importer','crop'])['qty_avg'].transform('sum')
shares['trade_share'] = shares['qty_avg'] / tot.replace(0, np.nan)
shares = shares.dropna(subset=['trade_share'])
shares['trade_share'] = shares['trade_share'].clip(0, 1)

print(f"Trade shares (2018-2021 avg): {len(shares):,}r, {shares['iso3_importer'].nunique()} importers")

# ── 7. Compute projected FTE ──────────────────────────────────────────────────
# FTE_proj(i,c,t) = Σ_j share(i,j,c) × FCE_proj(j,c,t,ssp)
fce_proj_merge = fce_proj[['iso3','crop','year','ssp','fce_proj']].rename(
    columns={'iso3':'iso3_exporter'}
)

fte_proj = shares.merge(fce_proj_merge, on=['iso3_exporter','crop'], how='inner')
fte_proj['fte_contrib'] = fte_proj['trade_share'] * fte_proj['fce_proj']

# Sum contributions per importer-crop-year-ssp
fte_proj_sum = (fte_proj.groupby(['iso3_importer','crop','year','ssp'], as_index=False)
                ['fte_contrib'].sum()
                .rename(columns={'iso3_importer':'iso3','fte_contrib':'fte_crop_proj'}))

# Aggregate to importer-level (caloric weighting)
CAL_WEIGHTS = {'rice': 0.36, 'wheat': 0.28, 'maize': 0.18, 'soybeans': 0.09}
fte_proj_sum['cal_weight'] = fte_proj_sum['crop'].map(CAL_WEIGHTS).fillna(0.09)
fte_proj_sum['fte_weighted'] = fte_proj_sum['fte_crop_proj'] * fte_proj_sum['cal_weight']

fte_total_proj = (fte_proj_sum.groupby(['iso3','year','ssp'], as_index=False)
                  ['fte_weighted'].sum()
                  .rename(columns={'fte_weighted':'fte_proj_km2'}))

print(f"Projected FTE: {len(fte_total_proj):,}r, range: {fte_total_proj['fte_proj_km2'].min():.1f} – {fte_total_proj['fte_proj_km2'].max():.1f}")

# ── 8. Compute baseline FTE (2018-2021 average) ──────────────────────────────
fte_baseline_country = (master[master['year']>=2018]
                        .groupby('iso3', as_index=False)['fte_total'].mean()
                        .rename(columns={'fte_total':'fte_baseline_km2'}))
# v2 master stores fte_total in 1000 km² (scaled for regression); convert back to km²
if fce_v2:
    fte_baseline_country['fte_baseline_km2'] *= 1000

# ── 9. Compute ΔFTE and apply β ──────────────────────────────────────────────
proj = fte_total_proj.merge(fte_baseline_country, on='iso3', how='left')
proj['fte_baseline_km2'] = proj['fte_baseline_km2'].fillna(0)
proj['delta_fte_km2']    = proj['fte_proj_km2'] - proj['fte_baseline_km2']
proj['delta_fte_1000']   = proj['delta_fte_km2'] / 1000

# Predicted change in log(kcal/pc/day) = β × ΔFTE_1000
proj['delta_log_kcal_central'] = beta_fte * proj['delta_fte_1000']
proj['delta_log_kcal_lo']      = (beta_fte - 1.96*se_fte) * proj['delta_fte_1000']
proj['delta_log_kcal_hi']      = (beta_fte + 1.96*se_fte) * proj['delta_fte_1000']

# Add metadata
wb = master[['iso3','income_level_id','income_level','region']].drop_duplicates('iso3')
proj = proj.merge(wb, on='iso3', how='left')

# Decade bin
proj['decade'] = (proj['year'] // 10) * 10

print(f"\nFinal projection panel: {len(proj):,}r, {proj['iso3'].nunique()} countries")
print(f"  ΔFTE range: {proj['delta_fte_km2'].min():.1f} – {proj['delta_fte_km2'].max():.1f}")
print(f"  Δlog(kcal) range: {proj['delta_log_kcal_central'].min():.4f} – {proj['delta_log_kcal_central'].max():.4f}")

# ── 10. Aggregate summaries ───────────────────────────────────────────────────
# By SSP × decade × income group
summary_inc = (proj.groupby(['ssp','decade','income_level_id'], as_index=False)
               .agg(delta_log_kcal_mean=('delta_log_kcal_central','mean'),
                    delta_log_kcal_p25=('delta_log_kcal_central', lambda x: np.nanpercentile(x,25)),
                    delta_log_kcal_p75=('delta_log_kcal_central', lambda x: np.nanpercentile(x,75)),
                    n_countries=('iso3','nunique')))

print("\n--- SSP × Decade × Income (median Δlog_kcal) ---")
print(summary_inc[summary_inc['decade'].isin([2030,2050,2070])].to_string(index=False))

# By SSP × decade × region
summary_reg = (proj.groupby(['ssp','decade','region'], as_index=False)
               .agg(delta_log_kcal_mean=('delta_log_kcal_central','mean'),
                    n_countries=('iso3','nunique')))

# ── 11. Save outputs ───────────────────────────────────────────────────────────
proj.to_csv(os.path.join(PROJ_DIR, 'projected_caloric_risk_by_country.csv'), index=False)
summary_inc.to_csv(os.path.join(PROJ_DIR, 'projected_risk_summary_income.csv'), index=False)
summary_reg.to_csv(os.path.join(PROJ_DIR, 'projected_risk_summary_region.csv'), index=False)

print(f"\n=== PHASE 5 COMPLETE ===")
print(f"  ✅ projected_caloric_risk_by_country.csv ({len(proj):,} rows)")
print(f"  ✅ projected_risk_summary_income.csv ({len(summary_inc):,} rows)")
print(f"  ✅ projected_risk_summary_region.csv ({len(summary_reg):,} rows)")
