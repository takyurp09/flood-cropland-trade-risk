"""
Phase 3 — Master Panel Build + All Regressions
Uses pre-computed:
  outputs/fte/fte_country_panel.csv
  outputs/fce/fce_final_panel.csv
  outputs/outcome/{fao_fbs_panel, enso_panel, era5_climate_panel, gta_export_restrictions}.csv
  data/raw/wb_income_groups.csv
  data/ipc/ipc_global_area_long.csv

Outputs:
  outputs/regression/master_panel.csv
  outputs/regression/main_regression_results.csv
  outputs/regression/heterogeneity_income_results.csv
  outputs/regression/heterogeneity_crop_results.csv
  outputs/regression/export_restriction_decomposition.csv
  outputs/regression/placebo_results.csv
"""
import pandas as pd, numpy as np, warnings, os
warnings.filterwarnings("ignore")

ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REG_DIR = os.path.join(ROOT, "outputs", "regression")
os.makedirs(REG_DIR, exist_ok=True)

try:
    import pycountry
    MANUAL_MAP = {
        'Afghanistan':'AFG','Albania':'ALB','Algeria':'DZA','Angola':'AGO',
        'Argentina':'ARG','Armenia':'ARM','Australia':'AUS','Austria':'AUT',
        'Azerbaijan':'AZE','Bangladesh':'BGD','Belarus':'BLR','Belgium':'BEL',
        'Belize':'BLZ','Benin':'BEN','Bolivia':'BOL',
        'Bolivia (Plurinational State of)':'BOL',
        'Bosnia and Herz.':'BIH','Bosnia and Herzegovina':'BIH','Botswana':'BWA',
        'Brazil':'BRA','Bulgaria':'BGR','Burkina Faso':'BFA','Burundi':'BDI',
        'Cambodia':'KHM','Cameroon':'CMR','Canada':'CAN',
        'Central African Rep.':'CAF','Central African Republic':'CAF',
        'Chad':'TCD','Chile':'CHL','China':'CHN','China, mainland':'CHN',
        'China, Hong Kong SAR':'HKG','China, Macao SAR':'MAC',
        'China, Taiwan Province of':'TWN','Colombia':'COL','Congo':'COG',
        'Costa Rica':'CRI','Croatia':'HRV','Cuba':'CUB','Czechia':'CZE',
        'Czech Republic':'CZE','Dem. Rep. Congo':'COD',
        'Democratic Republic of the Congo':'COD',
        'Dem. Rep. of the Congo':'COD','Denmark':'DNK','Dominican Rep.':'DOM',
        'Dominican Republic':'DOM','Ecuador':'ECU','Egypt':'EGY',
        'El Salvador':'SLV','Eritrea':'ERI','Estonia':'EST','Ethiopia':'ETH',
        'Fiji':'FJI','Finland':'FIN','France':'FRA','Gabon':'GAB',
        'Gambia':'GMB','Georgia':'GEO','Germany':'DEU','Ghana':'GHA',
        'Greece':'GRC','Guatemala':'GTM','Guinea':'GIN',
        'Guinea-Bissau':'GNB','Guyana':'GUY','Haiti':'HTI',
        'Honduras':'HND','Hungary':'HUN','India':'IND','Indonesia':'IDN',
        'Iran':'IRN','Iran (Islamic Republic of)':'IRN','Iraq':'IRQ',
        'Ireland':'IRL','Israel':'ISR','Italy':'ITA',
        "Côte d'Ivoire":'CIV',"CÃ´te d'Ivoire":'CIV','Ivory Coast':'CIV',
        'Jamaica':'JAM','Japan':'JPN','Jordan':'JOR','Kazakhstan':'KAZ',
        'Kenya':'KEN','Kuwait':'KWT','Kyrgyzstan':'KGZ','Laos':'LAO',
        "Lao People's Democratic Republic":'LAO','Lao PDR':'LAO',
        'Latvia':'LVA','Lebanon':'LBN','Lesotho':'LSO','Liberia':'LBR',
        'Libya':'LBY','Lithuania':'LTU','Luxembourg':'LUX',
        'Madagascar':'MDG','Malawi':'MWI','Malaysia':'MYS','Mali':'MLI',
        'Mauritania':'MRT','Mexico':'MEX','Moldova':'MDA',
        'Republic of Moldova':'MDA','Mongolia':'MNG','Montenegro':'MNE',
        'Morocco':'MAR','Mozambique':'MOZ','Myanmar':'MMR',
        'Namibia':'NAM','Nepal':'NPL','Netherlands':'NLD',
        'New Zealand':'NZL','Nicaragua':'NIC','Niger':'NER','Nigeria':'NGA',
        'North Korea':'PRK',"Democratic People's Republic of Korea":'PRK',
        'North Macedonia':'MKD','Norway':'NOR','Oman':'OMN',
        'Pakistan':'PAK','Palestine':'PSE','Panama':'PAN',
        'Papua New Guinea':'PNG','Paraguay':'PRY','Peru':'PER',
        'Philippines':'PHL','Poland':'POL','Portugal':'PRT','Qatar':'QAT',
        'Romania':'ROU','Russia':'RUS','Russian Federation':'RUS',
        'Rwanda':'RWA','Saudi Arabia':'SAU','Saudi Arabia, Kingdom of':'SAU',
        'Senegal':'SEN','Serbia':'SRB','Sierra Leone':'SLE',
        'Slovakia':'SVK','Slovenia':'SVN','Somalia':'SOM',
        'South Africa':'ZAF','South Korea':'KOR',
        'Republic of Korea':'KOR','South Sudan':'SSD','Spain':'ESP',
        'Sri Lanka':'LKA','Sudan':'SDN','Suriname':'SUR','Sweden':'SWE',
        'Switzerland':'CHE','Syria':'SYR','Syrian Arab Republic':'SYR',
        'Taiwan':'TWN','Tajikistan':'TJK','Tanzania':'TZA',
        'United Republic of Tanzania':'TZA','Thailand':'THA',
        'Timor-Leste':'TLS','Togo':'TGO','Trinidad and Tobago':'TTO',
        'Tunisia':'TUN','Turkey':'TUR','Türkiye':'TUR','Turkmenistan':'TKM',
        'Uganda':'UGA','Ukraine':'UKR','United Arab Emirates':'ARE',
        'United Kingdom':'GBR','United Kingdom of Great Britain and Northern Ireland':'GBR',
        'United States of America':'USA','United States':'USA',
        'Uruguay':'URY','Uzbekistan':'UZB','Venezuela':'VEN',
        'Venezuela (Bolivarian Republic of)':'VEN',
        'Viet Nam':'VNM','Vietnam':'VNM','Yemen':'YEM','Zambia':'ZMB',
        'Zimbabwe':'ZWE','Eswatini':'SWZ','Swaziland':'SWZ',
        'Bhutan':'BTN','Comoros':'COM','Cabo Verde':'CPV','Cape Verde':'CPV',
        'Maldives':'MDV',
    }
    _cache = {}
    def get_iso3(name):
        if pd.isna(name): return None
        n = str(name).strip()
        if n in _cache: return _cache[n]
        r = MANUAL_MAP.get(n)
        if r is None:
            try: r = pycountry.countries.search_fuzzy(n)[0].alpha_3
            except: r = None
        _cache[n] = r; return r
except ImportError:
    def get_iso3(name): return MANUAL_MAP.get(str(name).strip()) if not pd.isna(name) else None

# ── 1. FBS → caloric import outcome ──────────────────────────────────────────
fbs_raw = pd.read_csv(os.path.join(ROOT, "outputs", "outcome", "fao_fbs_panel.csv"))

# Grand Total kcal/pc/day (dependent variable approach A: total food supply)
kcal_tot = (fbs_raw[(fbs_raw['Item']=='Grand Total')&(fbs_raw['element']=='kcal_pc_day')]
            [['country','year','value']].copy().rename(columns={'value':'kcal_pc_day'}))

# Staple imports (kt) — summed across 5 commodity groups
IMP_ITEMS = ['Rice and products','Wheat and products','Maize and products',
             'Soyabeans','Vegetable Oils']
imp_kt = (fbs_raw[(fbs_raw['element']=='import_qty_kt') & fbs_raw['Item'].isin(IMP_ITEMS)]
          .groupby(['country','year'], as_index=False)['value'].sum()
          .rename(columns={'value':'staple_import_kt'}))

fbs = kcal_tot.merge(imp_kt, on=['country','year'], how='left')
fbs['iso3'] = fbs['country'].map(get_iso3)
# Drop aggregate rows (Africa, Americas, etc.)
fbs = fbs[fbs['iso3'].notna()].copy()
# Log DV: log(kcal_pc_day)
fbs['log_kcal_pc_day'] = np.log(fbs['kcal_pc_day'].clip(lower=1))
# Log imports
fbs['log_staple_import_kt'] = np.log(fbs['staple_import_kt'].clip(lower=1))
print(f"FBS clean: {len(fbs):,}r  {fbs['iso3'].nunique()} countries")

# ── 2. FTE country panel ──────────────────────────────────────────────────────
fte = pd.read_csv(os.path.join(ROOT, "outputs", "fte", "fte_country_panel.csv"))
fte = fte.rename(columns={'iso3_importer':'iso3'})
# Normalise FTE: km² → per 1000 ha of importer's own cropland (make units interpretable)
# Actually keep raw km² for now — coefficient will be interpreted as km² of flood-affected
# exporter cropland exposure per unit change in log imports
print(f"FTE: {len(fte):,}r  {fte['iso3'].nunique()} importers")

# ── 3. ENSO ───────────────────────────────────────────────────────────────────
enso = pd.read_csv(os.path.join(ROOT, "outputs", "outcome", "enso_panel.csv"))
print(f"ENSO: {len(enso):,}r  years {enso['year'].min()}–{enso['year'].max()}")

# ── 4. ERA5 domestic climate ──────────────────────────────────────────────────
era5 = pd.read_csv(os.path.join(ROOT, "outputs", "outcome", "era5_climate_panel.csv"))
era5 = era5[['iso3','year','t2m_anom_C','tp_anom_frac']].copy()
print(f"ERA5: {len(era5):,}r  {era5['iso3'].nunique()} countries")

# ── 5. GTA export restriction ─────────────────────────────────────────────────
gta_raw = pd.read_csv(os.path.join(ROOT, "outputs", "outcome", "gta_export_restrictions.csv"))
gta_raw['iso3'] = gta_raw['country'].map(get_iso3)
gta_raw = gta_raw[gta_raw['iso3'].notna()]
# Aggregate to exporter-year (any commodity restriction = 1)
gta = (gta_raw.groupby(['iso3','year'], as_index=False)['export_restriction']
       .max().rename(columns={'iso3':'iso3_exporter','export_restriction':'any_export_restriction'}))
# For importing country: we want whether their top suppliers had restrictions
# Merge into master later as exporter-level control; for now create importer-level version
# by checking if any top-5 supplier had restriction
fte_crop = pd.read_csv(os.path.join(ROOT, "outputs", "fte", "fte_panel.csv"))
fte_crop = fte_crop.rename(columns={'iso3_importer':'iso3'})
top_sup = fte_crop[fte_crop['n_suppliers']>=1][['iso3','year','top_supplier_iso3']].drop_duplicates()
top_sup = top_sup.merge(gta.rename(columns={'iso3_exporter':'top_supplier_iso3'}),
                        on=['top_supplier_iso3','year'], how='left')
top_sup['any_export_restriction'] = top_sup['any_export_restriction'].fillna(0)
# Importer-year: max across crops
gta_imp = (top_sup.groupby(['iso3','year'], as_index=False)['any_export_restriction']
           .max().rename(columns={'any_export_restriction':'supplier_export_restriction'}))
print(f"GTA (importer-year): {len(gta_imp):,}r  restrictions: {gta_imp['supplier_export_restriction'].sum():.0f}")

# ── 6. WB income groups ───────────────────────────────────────────────────────
wb = pd.read_csv(os.path.join(ROOT, "data", "raw", "wb_income_groups.csv"))
print(f"WB: {len(wb):,}r  groups: {wb['income_level_id'].value_counts().to_dict()}")

# ── 7. Master panel merge ─────────────────────────────────────────────────────
panel = fbs[['iso3','year','log_kcal_pc_day','log_staple_import_kt',
             'kcal_pc_day','staple_import_kt']].copy()
panel = panel.merge(fte[['iso3','year','fte_total','fte_low','fte_high']], on=['iso3','year'], how='left')
panel = panel.merge(enso[['year','mei_annual_mean']], on='year', how='left')
panel = panel.merge(era5[['iso3','year','t2m_anom_C','tp_anom_frac']], on=['iso3','year'], how='left')
panel = panel.merge(gta_imp, on=['iso3','year'], how='left')
panel = panel.merge(wb[['iso3','income_level_id','income_level','region']], on='iso3', how='left')

panel['supplier_export_restriction'] = panel['supplier_export_restriction'].fillna(0)
panel['fte_total']                   = panel['fte_total'].fillna(0)
panel['fte_low']                     = panel['fte_low'].fillna(0)
panel['fte_high']                    = panel['fte_high'].fillna(0)

# Filter to study years
panel = panel[(panel['year']>=2000)&(panel['year']<=2021)].copy()

# Deduplicate any iso3-year pairs (e.g. from two FBS country variants mapping to same ISO3)
dup_before = panel.duplicated(subset=['iso3','year']).sum()
if dup_before > 0:
    print(f"  ⚠️  Deduplicating {dup_before} duplicate iso3/year rows (avg of numeric cols)")
    num_cols = panel.select_dtypes(include='number').columns.tolist()
    cat_cols = [c for c in panel.columns if c not in num_cols and c not in ['iso3','year']]
    agg = {c: 'mean' for c in num_cols}
    agg.update({c: 'first' for c in cat_cols})
    panel = panel.groupby(['iso3','year'], as_index=False).agg(agg)

# Coverage report
print(f"\nMaster panel: {len(panel):,}r  {panel['iso3'].nunique()} countries")
for col in ['log_kcal_pc_day','fte_total','mei_annual_mean','t2m_anom_C','tp_anom_frac',
            'supplier_export_restriction','income_level_id']:
    miss = panel[col].isna().sum()
    pct  = miss/len(panel)
    flag = '⚠️ ' if pct>0.1 else '✅'
    print(f"  {flag} {col}: {miss} nulls ({pct:.1%})")

# Regression ready: drop rows missing DV or key exposure
reg = panel.dropna(subset=['log_kcal_pc_day','fte_total']).copy()
print(f"  Regression-ready: {len(reg):,}r  {reg['iso3'].nunique()} countries")

# Add region dummies for ENSO interaction
reg['region_EA'] = (reg['region']=='East Asia & Pacific').astype(int)
reg['region_SA'] = (reg['region']=='South Asia').astype(int)
reg['region_SSA']= (reg['region']=='Sub-Saharan Africa').astype(int)
reg['region_MENA']=(reg['region']=='Middle East & North Africa').astype(int)

reg.to_csv(os.path.join(REG_DIR, "master_panel.csv"), index=False)
print(f"\n✅ master_panel.csv: {len(reg):,}r, {list(reg.columns)}")

# Rescale FTE to 1000 km² for interpretable coefficients
# β_fte will then = log-point change in kcal per 1000 km² additional FTE
for col in ['fte_total','fte_low','fte_high']:
    reg[col] = reg[col] / 1000

# ── 8. REGRESSIONS ────────────────────────────────────────────────────────────

from linearmodels.panel import PanelOLS
import statsmodels.formula.api as smf

def run_twfe(data, y, x_vars, label, cluster_var='iso3', drop_absorbed=False):
    """Run TWFE with country+year FE using linearmodels PanelOLS (within transformation)."""
    try:
        df = data.dropna(subset=[y] + x_vars).copy()
        # Set panel index
        df = df.set_index([cluster_var, 'year'])
        dep = df[y].squeeze()  # ensure Series
        exog_cols = x_vars.copy()
        exog = df[exog_cols]
        # Add constant absorbed by FE; use entity_effects + time_effects
        m = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                     drop_absorbed=drop_absorbed, check_rank=False).fit(
            cov_type='clustered', cluster_entity=True
        )
        # FTE coeff
        fte_coef_name = [c for c in m.params.index if 'fte_total' in c or 'fte_' in c.lower()]
        if not fte_coef_name:
            print(f"  {label}: fte coeff not found in params — {list(m.params.index)[:5]}")
            return None
        cn = fte_coef_name[0]
        ci = m.conf_int(level=0.95)
        res = {
            'spec': label,
            'y': y,
            'N': int(m.nobs),
            'n_countries': df.index.get_level_values(0).nunique(),
            'beta_fte': float(m.params[cn]),
            'se_fte':   float(m.std_errors[cn]),
            'pval_fte': float(m.pvalues[cn]),
            'ci_lo':    float(ci.loc[cn,'lower']),
            'ci_hi':    float(ci.loc[cn,'upper']),
            'r2_within': float(m.rsquared_within),
        }
        stars = '***' if res['pval_fte']<0.01 else ('**' if res['pval_fte']<0.05 else ('*' if res['pval_fte']<0.1 else ''))
        print(f"  {label}: β={res['beta_fte']:.4f} SE={res['se_fte']:.4f} p={res['pval_fte']:.3f}{stars}  N={res['N']}")
        return res
    except Exception as e:
        print(f"  {label} FAILED: {e}")
        return None

print("\n=== MAIN REGRESSIONS ===")
results = []

# S1: FTE only
s1 = run_twfe(reg, 'log_kcal_pc_day', ['fte_total'], 'S1_FTE_only')
if s1: results.append(s1)

# S2: + ENSO×region interactions (mei_annual_mean itself absorbed by year FE — use interactions only)
reg['mei_x_EA']  = reg['mei_annual_mean'] * reg['region_EA']
reg['mei_x_SA']  = reg['mei_annual_mean'] * reg['region_SA']
reg['mei_x_SSA'] = reg['mei_annual_mean'] * reg['region_SSA']
s2 = run_twfe(reg, 'log_kcal_pc_day',
              ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA'], 'S2_ENSO')
if s2: results.append(s2)

# S3: + ERA5 domestic climate (PREFERRED)
reg3 = reg.dropna(subset=['t2m_anom_C','tp_anom_frac']).copy()
reg3['mei_x_EA']  = reg3['mei_annual_mean'] * reg3['region_EA']
reg3['mei_x_SA']  = reg3['mei_annual_mean'] * reg3['region_SA']
reg3['mei_x_SSA'] = reg3['mei_annual_mean'] * reg3['region_SSA']
s3 = run_twfe(reg3, 'log_kcal_pc_day',
              ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA',
               't2m_anom_C','tp_anom_frac'], 'S3_ERA5')
if s3: results.append(s3)

# S4: + export restriction (FULL)
s4 = run_twfe(reg3, 'log_kcal_pc_day',
              ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA',
               't2m_anom_C','tp_anom_frac',
               'supplier_export_restriction'], 'S4_Full')
if s4: results.append(s4)

pd.DataFrame(results).to_csv(os.path.join(REG_DIR, "main_regression_results.csv"), index=False)
print(f"✅ main_regression_results.csv ({len(results)} specs)")

# ── 9. Heterogeneity by income group ─────────────────────────────────────────
print("\n=== HETEROGENEITY: INCOME GROUP ===")
inc_results = []
base_x = ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac']
for grp_id, grp_label in [('LIC','Low income'),('LMC','Lower middle'),('UMC','Upper middle'),('HIC','High income')]:
    sub = reg3[reg3['income_level_id']==grp_id].copy()
    if len(sub) < 50: print(f"  {grp_id}: too few obs ({len(sub)}) — skip"); continue
    r = run_twfe(sub, 'log_kcal_pc_day', base_x, f'S3_{grp_id}', drop_absorbed=True)
    if r: r['income_group'] = grp_id; inc_results.append(r)

pd.DataFrame(inc_results).to_csv(os.path.join(REG_DIR, "heterogeneity_income_results.csv"), index=False)
print(f"✅ heterogeneity_income_results.csv ({len(inc_results)} specs)")

# ── 10. Heterogeneity by crop ─────────────────────────────────────────────────
print("\n=== HETEROGENEITY: CROP ===")
fte_crop2 = fte_crop.rename(columns={'iso3_importer':'iso3','fte':'fte_crop'}).copy()
fte_crop2['fte_crop'] = fte_crop2['fte_crop'] / 1000  # rescale to 1000 km²
crop_results = []
for crop in ['rice','wheat','maize','soybeans','oil_crops']:
    crop_col = f'fte_{crop}'
    fc = (fte_crop2[fte_crop2['crop']==crop]
          .groupby(['iso3','year'], as_index=False)['fte_crop'].sum()
          .rename(columns={'fte_crop': crop_col}))
    sub = reg3.merge(fc, on=['iso3','year'], how='left')
    sub[crop_col] = sub[crop_col].fillna(0)
    # Deduplicate in case merge introduced duplicates
    sub = sub.drop_duplicates(subset=['iso3','year'])
    # run_twfe looks for fte_total in params — use wrapper that renames
    # Drop original fte_total to avoid duplicate column after rename
    sub2 = sub.drop(columns=['fte_total','fte_low','fte_high'], errors='ignore')
    sub2 = sub2.rename(columns={crop_col:'fte_total'})
    r = run_twfe(sub2, 'log_kcal_pc_day',
                 ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac'], f'S3_{crop}')
    if r:
        r['crop'] = crop
        crop_results.append(r)

pd.DataFrame(crop_results).to_csv(os.path.join(REG_DIR, "heterogeneity_crop_results.csv"), index=False)
print(f"✅ heterogeneity_crop_results.csv ({len(crop_results)} specs)")

# ── 11. Export restriction decomposition ─────────────────────────────────────
print("\n=== EXPORT RESTRICTION DECOMPOSITION ===")
decomp = []
if s3: decomp.append({**s3, 'note':'S3_preferred_without_ER'})
if s4: decomp.append({**s4, 'note':'S4_full_with_ER'})
if s3 and s4:
    diff = s3['beta_fte'] - s4['beta_fte']
    print(f"  Direct β (S3): {s3['beta_fte']:.4f}")
    print(f"  Policy-adj β (S4): {s4['beta_fte']:.4f}")
    print(f"  Difference (policy channel): {diff:.4f} ({diff/s3['beta_fte']*100:.1f}% of direct)")
    decomp.append({'spec':'decomp_policy_channel','note':'S3_beta - S4_beta','beta_fte':diff})
pd.DataFrame(decomp).to_csv(os.path.join(REG_DIR, "export_restriction_decomposition.csv"), index=False)
print(f"✅ export_restriction_decomposition.csv")

# ── 12. Placebo tests ─────────────────────────────────────────────────────────
print("\n=== PLACEBO TESTS ===")
placebo_results = []

# Temporal placebo: FTE leads
for lead in [1, 2]:
    reg_lead = reg3.copy()
    lead_col = f'fte_lead{lead}'
    reg_lead[lead_col] = reg_lead.groupby('iso3')['fte_total'].shift(-lead)
    reg_lead2 = reg_lead.dropna(subset=[lead_col,'t2m_anom_C'])
    reg_lead2 = reg_lead2.drop_duplicates(subset=['iso3','year'])
    # Drop original fte_total to avoid duplicate column after rename
    reg_lead2 = reg_lead2.drop(columns=['fte_total','fte_low','fte_high'], errors='ignore')
    reg_lead2 = reg_lead2.rename(columns={lead_col:'fte_total'})
    r = run_twfe(reg_lead2, 'log_kcal_pc_day',
                 ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac'],
                 f'Placebo_FTE_lead{lead}')
    if r: r['placebo_type'] = f'temporal_lead{lead}'; placebo_results.append(r)

# Geographic placebo: random reassignment of FTE
np.random.seed(42)
reg_geo = reg3.copy()
iso3_list = reg3['iso3'].unique()
iso3_shuf = iso3_list.copy()
np.random.shuffle(iso3_shuf)
iso_map = dict(zip(iso3_list, iso3_shuf))
reg_geo['iso3_random'] = reg_geo['iso3'].map(iso_map)
fte_rand = fte.rename(columns={'iso3':'iso3_random'})
reg_geo = reg_geo.drop(columns=['fte_total','fte_low','fte_high'], errors='ignore')
reg_geo = reg_geo.merge(fte_rand[['iso3_random','year','fte_total']], on=['iso3_random','year'], how='left')
reg_geo['fte_total'] = reg_geo['fte_total'].fillna(0) / 1000  # rescale consistent with main
r = run_twfe(reg_geo.dropna(subset=['t2m_anom_C']), 'log_kcal_pc_day',
             ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac'],
             'Placebo_Geographic')
if r: r['placebo_type'] = 'geographic'; placebo_results.append(r)

pd.DataFrame(placebo_results).to_csv(os.path.join(REG_DIR, "placebo_results.csv"), index=False)
print(f"✅ placebo_results.csv ({len(placebo_results)} tests)")

print("\n=== ALL PHASE 3 OUTPUTS COMPLETE ===")
for f in ['master_panel.csv','main_regression_results.csv','heterogeneity_income_results.csv',
          'heterogeneity_crop_results.csv','export_restriction_decomposition.csv','placebo_results.csv']:
    path = os.path.join(REG_DIR, f)
    exists = os.path.exists(path)
    try:
        rows = len(pd.read_csv(path)) if exists else 0
    except Exception:
        rows = 0
    print(f"  {'✅' if exists else '❌'} {f} ({rows} rows)")
