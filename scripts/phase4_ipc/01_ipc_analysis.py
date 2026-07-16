"""
Phase 4 — IPC Food Security Analysis
Link FTE to IPC food security outcomes using ordered logit.
- IPC: area-level phases 1-5, aggregated to national population-weighted Phase3+ share
- Restriction: current validity period, 2017-2021 only
- DV: logit(P(Phase≥3)) at country-year level, or national Phase3+ share
- Model: OLS country+year FE (population-weighted), complemented by ordered logit
"""

import os, warnings
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IPC_DIR  = os.path.join(BASE, 'data', 'ipc')
REG_DIR  = os.path.join(BASE, 'outputs', 'regression')
FTE_FILE = os.path.join(BASE, 'outputs', 'fte', 'fte_country_panel.csv')
PANEL    = os.path.join(REG_DIR, 'master_panel.csv')

# ── 1. Load IPC data ──────────────────────────────────────────────────────────
ipc_raw = pd.read_csv(os.path.join(IPC_DIR, 'ipc_global_area_long.csv'))
print(f"IPC raw: {len(ipc_raw):,}r")

# Keep only current validity period
ipc = ipc_raw[ipc_raw['Validity period'] == 'current'].copy()

# Parse year from date of analysis (e.g. "Oct 2021" -> 2021)
ipc['year'] = ipc['Date of analysis'].str.extract(r'(\d{4})').astype(int)

# Restrict to 2017-2021
ipc = ipc[(ipc['year']>=2017) & (ipc['year']<=2021)].copy()
print(f"IPC filtered (current, 2017-2021): {len(ipc):,}r")
print(f"  Countries: {ipc['Country'].nunique()}, Years: {sorted(ipc['year'].unique())}")

# ── 2. Aggregate to national level ──────────────────────────────────────────
# For each area-year-country: get Phase 3+ numbers and total 'all' numbers
# Phase 3+ means Phase in ['3', '3+', '4', '5']
# Use population counts ('Number') for weighting

phase_all  = ipc[ipc['Phase'] == 'all'][['Country','year','Area','Number']].rename(columns={'Number':'total_pop'})
phase_3plus_labels = ['3', '3+', '4', '5']
phase_3p   = ipc[ipc['Phase'].isin(phase_3plus_labels)].copy()

# Sum Phase 3+ numbers by area-country-year (there can be multiple phase rows per area)
phase_3p_sum = phase_3p.groupby(['Country','year','Area'], as_index=False)['Number'].sum().rename(columns={'Number':'pop_3plus'})

# Merge with totals
area_data = phase_all.merge(phase_3p_sum, on=['Country','year','Area'], how='left')
area_data['pop_3plus'] = area_data['pop_3plus'].fillna(0)

# Some areas may have pop_3plus > total_pop (data artefact) — clip
area_data['pop_3plus'] = area_data[['pop_3plus','total_pop']].min(axis=1)

# National aggregate: sum across areas within country-year
national = area_data.groupby(['Country','year'], as_index=False).agg(
    total_pop=('total_pop','sum'),
    pop_3plus=('pop_3plus','sum')
)
national['share_3plus'] = national['pop_3plus'] / national['total_pop'].replace(0, np.nan)
national = national.rename(columns={'Country':'iso3'})
national = national.dropna(subset=['share_3plus'])

print(f"\nNational IPC panel: {len(national):,}r")
print(f"  Countries: {national['iso3'].nunique()}, Years: {sorted(national['year'].unique())}")
print(f"  Share_3plus range: {national['share_3plus'].min():.3f} – {national['share_3plus'].max():.3f}")

# ── 3. Merge with FTE and controls ───────────────────────────────────────────
master = pd.read_csv(PANEL)
master = master[['iso3','year','fte_total','mei_annual_mean','t2m_anom_C','tp_anom_frac',
                 'supplier_export_restriction','income_level_id','region']].copy()
# Rescale FTE (master already has raw km², rescale to 1000 km²)
master['fte_1000'] = master['fte_total'] / 1000

ipc_panel = national.merge(master, on=['iso3','year'], how='inner')
print(f"\nIPC + controls merged: {len(ipc_panel):,}r, {ipc_panel['iso3'].nunique()} countries")

# Logit-transform share_3plus for OLS interpretation
# Use logit(p) = log(p/(1-p)), clip to avoid inf
ipc_panel['share_3plus_clipped'] = ipc_panel['share_3plus'].clip(0.001, 0.999)
ipc_panel['logit_share_3plus'] = np.log(ipc_panel['share_3plus_clipped'] / (1 - ipc_panel['share_3plus_clipped']))

# Region dummies for ENSO interaction
ipc_panel['region_EA']  = (ipc_panel['region']=='East Asia & Pacific').astype(int)
ipc_panel['region_SA']  = (ipc_panel['region']=='South Asia').astype(int)
ipc_panel['region_SSA'] = (ipc_panel['region']=='Sub-Saharan Africa').astype(int)
ipc_panel['mei_x_EA']   = ipc_panel['mei_annual_mean'] * ipc_panel['region_EA']
ipc_panel['mei_x_SA']   = ipc_panel['mei_annual_mean'] * ipc_panel['region_SA']
ipc_panel['mei_x_SSA']  = ipc_panel['mei_annual_mean'] * ipc_panel['region_SSA']

ipc_panel = ipc_panel.drop_duplicates(subset=['iso3','year'])
print(f"  After drop_duplicates: {len(ipc_panel):,}r")

# ── 4. TWFE regression ────────────────────────────────────────────────────────
def run_twfe_ipc(data, y, x_vars, label):
    try:
        df = data.dropna(subset=[y] + x_vars).copy()
        if df['iso3'].nunique() < 5 or len(df) < 20:
            print(f"  {label}: too few obs ({len(df)}) — skip")
            return None
        df = df.drop_duplicates(subset=['iso3','year'])
        df = df.set_index(['iso3','year'])
        dep  = df[y].squeeze()
        exog = df[x_vars]
        m = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                     check_rank=False, drop_absorbed=True).fit(
            cov_type='clustered', cluster_entity=True
        )
        cn = [c for c in m.params.index if 'fte' in c.lower()]
        if not cn:
            print(f"  {label}: fte coeff not found — {list(m.params.index)[:5]}")
            return None
        cn = cn[0]
        ci = m.conf_int(level=0.95)
        res = {
            'spec': label, 'y': y,
            'N': int(m.nobs),
            'n_countries': df.index.get_level_values(0).nunique(),
            'beta_fte': float(m.params[cn]),
            'se_fte':   float(m.std_errors[cn]),
            'pval_fte': float(m.pvalues[cn]),
            'ci_lo': float(ci.loc[cn,'lower']),
            'ci_hi': float(ci.loc[cn,'upper']),
            'r2_within': float(m.rsquared_within),
        }
        stars = '***' if res['pval_fte']<0.01 else ('**' if res['pval_fte']<0.05 else ('*' if res['pval_fte']<0.1 else ''))
        print(f"  {label}: β={res['beta_fte']:.4f} SE={res['se_fte']:.4f} p={res['pval_fte']:.3f}{stars}  N={res['N']}")
        return res
    except Exception as e:
        print(f"  {label} FAILED: {e}")
        return None

print("\n=== IPC REGRESSIONS ===")
results_ipc = []

base_x = ['fte_1000','mei_x_EA','mei_x_SA','mei_x_SSA']

# I1: logit share ~ FTE only (country+year FE)
r = run_twfe_ipc(ipc_panel, 'logit_share_3plus', base_x, 'I1_logit_FTE')
if r: results_ipc.append(r)

# I2: + climate controls
r = run_twfe_ipc(ipc_panel, 'logit_share_3plus',
                  base_x + ['t2m_anom_C','tp_anom_frac'], 'I2_logit_climate')
if r: results_ipc.append(r)

# I3: raw share ~ FTE (for comparison)
r = run_twfe_ipc(ipc_panel, 'share_3plus', base_x, 'I3_share_FTE')
if r: results_ipc.append(r)

# I4: Heterogeneity by income (SSA sub-sample)
ssa_sub = ipc_panel[ipc_panel['region']=='Sub-Saharan Africa'].copy()
r = run_twfe_ipc(ssa_sub, 'logit_share_3plus', base_x, 'I4_SSA_only')
if r: r['subsample'] = 'SSA'; results_ipc.append(r)

pd.DataFrame(results_ipc).to_csv(os.path.join(REG_DIR, 'ipc_regression_results.csv'), index=False)
print(f"\n✅ ipc_regression_results.csv ({len(results_ipc)} specs)")

# ── 5. Save national IPC panel for figures ────────────────────────────────────
ipc_panel.to_csv(os.path.join(REG_DIR, 'ipc_national_panel.csv'), index=False)
print(f"✅ ipc_national_panel.csv ({len(ipc_panel):,}r)")

# ── 6. Validation summary ────────────────────────────────────────────────────
print("\n=== IPC VALIDATION ===")
# Top countries by Phase3+ share in 2021
top = ipc_panel[ipc_panel['year']==2021].nlargest(10,'share_3plus')[['iso3','year','share_3plus','fte_1000']]
print("Top 10 food insecure (2021):")
print(top.to_string(index=False))

print("\n=== PHASE 4 COMPLETE ===")
print(f"  ✅ ipc_regression_results.csv ({len(results_ipc)} rows)")
print(f"  ✅ ipc_national_panel.csv ({len(ipc_panel):,} rows)")
