"""
Phase 2 — FTE v2 Construction using FCE v2 (monthly+seasonal filter)

Changes from v1:
  - Input: fce_final_panel_v2.csv (seasonal-filtered FCE)
  - Oil crops explicitly excluded (FCE=0 by design — oil_crops_excluded flag)
  - Caloric weights renormalised to 4 crops: rice=0.40, wheat=0.31, maize=0.20, soybeans=0.09
  - Produces fte_low, fte_central, fte_high from three FCE bounds
  - Computes Bartik FTE (trade shares fixed at 2000-2004 average) for IV instrument

Inputs:
  outputs/trade/baci_trade_panel.csv
  outputs/fce/fce_final_panel_v2.csv
Outputs:
  outputs/fte/fte_panel_v2.csv
  outputs/fte/fte_country_panel_v2.csv
  outputs/fte/bartik_fte_panel_v2.csv
"""
import os, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FTE_DIR = os.path.join(ROOT, "outputs", "fte")
os.makedirs(FTE_DIR, exist_ok=True)

# ── Load inputs ────────────────────────────────────────────────────────────────
baci = pd.read_csv(os.path.join(ROOT, "outputs", "trade", "baci_trade_panel.csv"))
fce  = pd.read_csv(os.path.join(ROOT, "outputs", "fce",   "fce_final_panel_v2.csv"))

print(f"BACI: {len(baci):,}r  FCE v2: {len(fce):,}r")
print(f"BACI commodities: {sorted(baci['commodity'].unique())}")

# ── Crop mapping: BACI commodity → FCE crop ───────────────────────────────────
CROP_MAP = {'rice':'rice','wheat':'wheat','maize':'maize',
            'soybeans':'soybeans','veg_oils':'oil_crops'}

# ── Trade shares ──────────────────────────────────────────────────────────────
baci_agg = (baci.groupby(['year','exporter','importer','commodity'], as_index=False)
            ['quantity_t'].sum().rename(columns={'quantity_t':'qty_t'}))

baci_agg['total_imports'] = baci_agg.groupby(['year','importer','commodity'])['qty_t'].transform('sum')
baci_agg['trade_share']   = np.where(baci_agg['total_imports'] > 0,
                                      baci_agg['qty_t'] / baci_agg['total_imports'], 0.0)
baci_agg['crop'] = baci_agg['commodity'].map(CROP_MAP)

# Verify shares
share_check = baci_agg.groupby(['year','importer','commodity'])['trade_share'].sum()
bad = (np.abs(share_check - 1) > 0.001).sum()
print(f"Trade shares: sum-to-1 violations (>0.001): {bad}")

# ── FCE exporter lookup ───────────────────────────────────────────────────────
fce_exp = (fce[['iso3','crop','year','fce_low','fce_central','fce_high',
                'oil_crops_excluded']]
           .rename(columns={'iso3':'exporter'}))

# For oil_crops, FCE should already be 0 — but explicitly ensure it
fce_exp.loc[fce_exp['oil_crops_excluded']==1, ['fce_low','fce_central','fce_high']] = 0.0

# ── FTE = Σ_j [ share(i←j,c,t) × FCE(j,c,t) ] ───────────────────────────────
joined = baci_agg.merge(fce_exp, on=['exporter','crop','year'], how='left')
for col in ('fce_low','fce_central','fce_high'):
    joined[col] = joined[col].fillna(0)

joined['fte_central_contrib'] = joined['trade_share'] * joined['fce_central']
joined['fte_low_contrib']     = joined['trade_share'] * joined['fce_low']
joined['fte_high_contrib']    = joined['trade_share'] * joined['fce_high']

fte = joined.groupby(['year','importer','crop'], as_index=False).agg(
    fte_central   = ('fte_central_contrib', 'sum'),
    fte_low       = ('fte_low_contrib',     'sum'),
    fte_high      = ('fte_high_contrib',    'sum'),
    n_suppliers   = ('exporter',            'count'),
)

# Top supplier
top_sup = (baci_agg.sort_values('trade_share', ascending=False)
           .groupby(['year','importer','crop'], as_index=False)
           .first()[['year','importer','crop','exporter','trade_share']]
           .rename(columns={'exporter':'top_supplier_iso3',
                            'trade_share':'top_supplier_share'}))
fte = fte.merge(top_sup, on=['year','importer','crop'], how='left')

# Oil crops column
fte['oil_crops_excluded'] = (fte['crop'] == 'oil_crops').astype(int)

fte = fte.rename(columns={'importer':'iso3'})
fte = fte.sort_values(['iso3','crop','year']).reset_index(drop=True)
fte.to_csv(os.path.join(FTE_DIR, "fte_panel_v2.csv"), index=False)
print(f"\nfte_panel_v2.csv: {len(fte):,}r  "
      f"importers:{fte['iso3'].nunique()}  crops:{fte['crop'].nunique()}  "
      f"years:{fte['year'].nunique()}")
print(f"  Trade shares sum-check: {bad} violations")
print(f"  FTE central range: {fte['fte_central'].min():.2f}–{fte['fte_central'].max():.2f}")
print(f"  Non-zero FTE: {(fte['fte_central']>0).sum():,} ({(fte['fte_central']>0).mean():.1%})")

# ── Country-level FTE (caloric-weighted, 4 crops, oil crops excluded) ─────────
# Oil crops excluded from caloric weighting; weights renormalised to sum=1
CALORIC_WEIGHTS = {'rice':0.40, 'wheat':0.31, 'maize':0.20, 'soybeans':0.09}
print(f"\nCaloric weights: {CALORIC_WEIGHTS}  sum={sum(CALORIC_WEIGHTS.values())}")

fte_4cr = fte[fte['crop'].isin(CALORIC_WEIGHTS.keys())].copy()
fte_4cr['caloric_weight'] = fte_4cr['crop'].map(CALORIC_WEIGHTS)

fte_ctry = (fte_4cr.groupby(['iso3','year'], as_index=False)
            .apply(lambda g: pd.Series({
                'fte_total':     (g['fte_central'] * g['caloric_weight']).sum(),
                'fte_total_low': (g['fte_low']     * g['caloric_weight']).sum(),
                'fte_total_high':(g['fte_high']    * g['caloric_weight']).sum(),
            })).reset_index(drop=True))

fte_ctry = fte_ctry.sort_values(['iso3','year']).reset_index(drop=True)
fte_ctry.to_csv(os.path.join(FTE_DIR, "fte_country_panel_v2.csv"), index=False)
print(f"\nfte_country_panel_v2.csv: {len(fte_ctry):,}r  "
      f"importers:{fte_ctry['iso3'].nunique()}  years:{fte_ctry['year'].nunique()}")
print(f"  FTE_total range: {fte_ctry['fte_total'].min():.4f}–{fte_ctry['fte_total'].max():.4f}")
print(f"  Non-zero: {(fte_ctry['fte_total']>0).sum():,} ({(fte_ctry['fte_total']>0).mean():.1%})")
print("  Top 10 exposed importers (mean):")
top10 = fte_ctry.groupby('iso3')['fte_total'].mean().nlargest(10).reset_index()
print(top10.to_string(index=False))

# ── Bartik instrument: FTE using 2000-2004 avg trade shares ──────────────────
bartik_shares = (baci_agg[baci_agg['year'].between(2000,2004)]
                 .groupby(['exporter','importer','crop'], as_index=False)['trade_share']
                 .mean().rename(columns={'trade_share':'bartik_share'}))

# Renormalise Bartik shares to sum to 1 per importer-crop
bartik_shares['tot_bar'] = bartik_shares.groupby(['importer','crop'])['bartik_share'].transform('sum')
bartik_shares['bartik_share'] = np.where(bartik_shares['tot_bar'] > 0,
    bartik_shares['bartik_share'] / bartik_shares['tot_bar'], 0.0)

# Join with FCE v2 for each year
bartik_joined = bartik_shares.merge(
    fce_exp[['exporter','crop','year','fce_central']],
    on=['exporter','crop'], how='left'
)
bartik_joined['fce_central'] = bartik_joined['fce_central'].fillna(0)
bartik_joined['fte_bartik_contrib'] = bartik_joined['bartik_share'] * bartik_joined['fce_central']

bartik_fte = (bartik_joined.groupby(['year','importer','crop'], as_index=False)
              ['fte_bartik_contrib'].sum()
              .rename(columns={'fte_bartik_contrib':'fte_bartik','importer':'iso3'}))

# Country-level Bartik FTE
bartik_4cr = bartik_fte[bartik_fte['crop'].isin(CALORIC_WEIGHTS.keys())].copy()
bartik_4cr['caloric_weight'] = bartik_4cr['crop'].map(CALORIC_WEIGHTS)

bartik_ctry = (bartik_4cr.groupby(['iso3','year'], as_index=False)
               .apply(lambda g: pd.Series({
                   'fte_bartik_total': (g['fte_bartik'] * g['caloric_weight']).sum(),
               })).reset_index(drop=True))

# Merge into country panel
fte_ctry = fte_ctry.merge(bartik_ctry, on=['iso3','year'], how='left')
fte_ctry.to_csv(os.path.join(FTE_DIR, "fte_country_panel_v2.csv"), index=False)

bartik_fte.to_csv(os.path.join(FTE_DIR, "bartik_fte_panel_v2.csv"), index=False)
print(f"\nbartik_fte_panel_v2.csv: {len(bartik_fte):,}r")
print(f"  Bartik FTE total range: {fte_ctry['fte_bartik_total'].min():.4f}–"
      f"{fte_ctry['fte_bartik_total'].max():.4f}")
print(f"  Bartik-FTE correlation with FTE: "
      f"{fte_ctry[['fte_total','fte_bartik_total']].dropna().corr().iloc[0,1]:.4f}")

# ── Comparison v1 vs v2 ───────────────────────────────────────────────────────
v1_ctry = pd.read_csv(os.path.join(FTE_DIR, "archive", "fte_country_panel_v1.csv"))
if 'iso3_importer' in v1_ctry.columns:
    v1_ctry = v1_ctry.rename(columns={'iso3_importer':'iso3','fte_total':'fte_total_v1'})
print("\nTop 10 most exposed importers — v1 vs v2:")
comp_ctry = fte_ctry.groupby('iso3')['fte_total'].mean().reset_index().rename(columns={'fte_total':'fte_v2'})
if 'iso3' in v1_ctry.columns and 'fte_total_v1' in v1_ctry.columns:
    comp_v1 = v1_ctry.groupby('iso3')['fte_total_v1'].mean().reset_index()
    comp = comp_ctry.merge(comp_v1, on='iso3', how='inner')
    comp = comp.nlargest(10, 'fte_v2')
    print(comp.to_string(index=False))
    corr = comp_ctry.merge(comp_v1, on='iso3')
    r = corr[['fte_v2','fte_total_v1']].corr().iloc[0,1]
    print(f"\nFTE v1-v2 correlation: {r:.4f}")
else:
    print("(v1 country panel structure differs — skipping direct comparison)")
    print(f"v1 cols: {list(v1_ctry.columns)}")

print("\nFTE v2 build complete.")
