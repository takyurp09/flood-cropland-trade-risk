"""
Phase 2 — FTE (Flood Trade Exposure) Construction
Inputs:
  outputs/trade/baci_trade_panel.csv
  outputs/fce/fce_final_panel.csv
Outputs:
  outputs/fte/fte_panel.csv         (importer × crop × year)
  outputs/fte/fte_country_panel.csv (importer × year, caloric-weighted)
"""
import pandas as pd, numpy as np, warnings, os
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(os.path.join(ROOT, "outputs", "fte"), exist_ok=True)
FTE_DIR = os.path.join(ROOT, "outputs", "fte")

# ── Load inputs ───────────────────────────────────────────────────────────────
baci = pd.read_csv(os.path.join(ROOT, "outputs", "trade", "baci_trade_panel.csv"))
fce  = pd.read_csv(os.path.join(ROOT, "outputs", "fce",   "fce_final_panel.csv"))

print(f"BACI: {len(baci):,}r  FCE: {len(fce):,}r")

# ── Step 2a — Trade shares ─────────────────────────────────────────────────────
# Use quantity_t for caloric calculations (volume-based)
# Group to exporter-importer-commodity-year
baci_agg = (baci.groupby(['year','exporter','importer','commodity'], as_index=False)
            ['quantity_t'].sum()
            .rename(columns={'quantity_t':'qty_t'}))

# Total imports per importer-commodity-year
baci_agg['total_imports'] = baci_agg.groupby(['year','importer','commodity'])['qty_t'].transform('sum')

baci_agg['trade_share'] = np.where(
    baci_agg['total_imports'] > 0,
    baci_agg['qty_t'] / baci_agg['total_imports'],
    0.0
)

# Verify shares sum to 1 per importer-crop-year
share_check = baci_agg.groupby(['year','importer','commodity'])['trade_share'].sum()
bad = (np.abs(share_check - 1) > 0.01).sum()
print(f"Trade shares sum-to-1 violations: {bad} (should be 0)")

# Flag dangerous concentration
top_share = baci_agg.groupby(['year','importer','commodity'])['trade_share'].max()
conc = (top_share > 0.95).sum()
print(f"Importer-crop-years with >95%% concentration: {conc:,}")

# ── Step 2b — FTE = Σ_j [ share(i,j,c,t) × FCE(j,c,t) ] ─────────────────────
# FCE needs to be exporter-centric: exporter=iso3, crop, year
fce_exp = fce[['iso3','crop','year','fce_central','fce_low','fce_high']].copy()
fce_exp = fce_exp.rename(columns={'iso3':'exporter'})

# Map BACI commodity to FCE crop names
crop_map = {'rice':'rice','wheat':'wheat','maize':'maize','soybeans':'soybeans','veg_oils':'oil_crops'}
baci_agg['crop'] = baci_agg['commodity'].map(crop_map)

# Join FCE onto trade shares
joined = baci_agg.merge(
    fce_exp[['exporter','crop','year','fce_central','fce_low','fce_high']],
    on=['exporter','crop','year'], how='left'
)
joined['fce_central'] = joined['fce_central'].fillna(0)
joined['fce_low']     = joined['fce_low'].fillna(0)
joined['fce_high']    = joined['fce_high'].fillna(0)

# FTE contribution = trade_share × FCE
joined['fte_contrib']      = joined['trade_share'] * joined['fce_central']
joined['fte_low_contrib']  = joined['trade_share'] * joined['fce_low']
joined['fte_high_contrib'] = joined['trade_share'] * joined['fce_high']

# Aggregate to importer-crop-year
fte = joined.groupby(['year','importer','crop'], as_index=False).agg(
    fte        = ('fte_contrib', 'sum'),
    fte_low    = ('fte_low_contrib', 'sum'),
    fte_high   = ('fte_high_contrib', 'sum'),
    n_suppliers= ('exporter', 'count'),
)
# Add top supplier info
top_sup = (baci_agg.sort_values('trade_share', ascending=False)
           .groupby(['year','importer','crop'], as_index=False)
           .first()[['year','importer','crop','exporter','trade_share']]
           .rename(columns={'exporter':'top_supplier_iso3','trade_share':'top_supplier_share'}))
fte = fte.merge(top_sup, on=['year','importer','crop'], how='left')
fte = fte.rename(columns={'importer':'iso3_importer'})

# Sort
fte = fte.sort_values(['iso3_importer','crop','year']).reset_index(drop=True)
fte.to_csv(os.path.join(FTE_DIR, "fte_panel.csv"), index=False)
print(f"fte_panel: {len(fte):,}r  cols={list(fte.columns)}")
print(f"  Importers:{fte['iso3_importer'].nunique()}  Crops:{fte['crop'].nunique()}  Years:{fte['year'].nunique()}")
print(f"  FTE range: {fte['fte'].min():.2f}–{fte['fte'].max():.2f}  mean:{fte['fte'].mean():.2f}")
print(f"  Non-zero FTE: {(fte['fte']>0).sum():,} ({(fte['fte']>0).mean():.1%})")

# ── Step 2c — Country-level FTE (caloric-weighted across crops) ───────────────
# Caloric weights per crop (fallback set; use FAO FBS shares if available later)
CALORIC_WEIGHTS = {'rice':0.36,'wheat':0.28,'maize':0.18,'soybeans':0.09,'oil_crops':0.09}

fte['caloric_weight'] = fte['crop'].map(CALORIC_WEIGHTS)

# Weighted sum
fte_ctry = fte.groupby(['iso3_importer','year'], as_index=False).apply(
    lambda g: pd.Series({
        'fte_total': (g['fte'] * g['caloric_weight']).sum(),
        'fte_low':   (g['fte_low'] * g['caloric_weight']).sum(),
        'fte_high':  (g['fte_high'] * g['caloric_weight']).sum(),
    })
).reset_index(drop=True)

fte_ctry = fte_ctry.sort_values(['iso3_importer','year']).reset_index(drop=True)
fte_ctry.to_csv(os.path.join(FTE_DIR, "fte_country_panel.csv"), index=False)
print(f"\nfte_country_panel: {len(fte_ctry):,}r  cols={list(fte_ctry.columns)}")
print(f"  Importers:{fte_ctry['iso3_importer'].nunique()}  Years:{fte_ctry['year'].nunique()}")
print(f"  FTE_total range: {fte_ctry['fte_total'].min():.3f}–{fte_ctry['fte_total'].max():.3f}  mean:{fte_ctry['fte_total'].mean():.3f}")
print(f"  Non-zero FTE_total: {(fte_ctry['fte_total']>0).sum():,} ({(fte_ctry['fte_total']>0).mean():.1%})")

# Top FTE country-years
top_fte = fte_ctry.nlargest(10,'fte_total')[['iso3_importer','year','fte_total']]
print(f"\n  Top 10 FTE country-years:\n{top_fte.to_string(index=False)}")
