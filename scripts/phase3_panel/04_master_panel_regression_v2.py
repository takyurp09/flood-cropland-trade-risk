"""
Phase 3 v2 — Master Panel Build + All Regressions
Fixes vs v1:
  A. Two-way clustered SE (country + year)
  B. GDP_pc control (WB WDI NY.GDP.PCAP.KD, via wbgapi)
  C. Bartik IV for FTE
  D. ERA5 dropped-country diagnostic
  E. New specs: S2b, S3b, S4_IV
  F. Herfindahl heterogeneity
  G. Leads t+1,t+2,t+3 placebo + lags t-1,t-2

Inputs:
  outputs/fte/fte_country_panel_v2.csv
  outputs/fte/fte_panel_v2.csv
  outputs/outcome/{fao_fbs_panel, enso_panel, era5_climate_panel, gta_export_restrictions}.csv
  data/raw/wb_income_groups.csv

Outputs (all _v2 suffix):
  outputs/regression/master_panel_v2.csv
  outputs/regression/main_regression_results_v2.csv
  outputs/regression/heterogeneity_income_v2.csv
  outputs/regression/heterogeneity_crop_v2.csv
  outputs/regression/heterogeneity_herfindahl_v2.csv
  outputs/regression/export_restriction_decomposition_v2.csv
  outputs/regression/placebo_results_v2.csv
  outputs/regression/iv_results_v2.csv
  outputs/regression/era5_dropped_countries.csv
"""
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REG_DIR = os.path.join(ROOT, "outputs", "regression")
os.makedirs(REG_DIR, exist_ok=True)

# ── Name → ISO3 helper (reused from v1) ───────────────────────────────────────
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
    'Democratic Republic of the Congo':'COD','Dem. Rep. of the Congo':'COD',
    'Denmark':'DNK','Dominican Rep.':'DOM','Dominican Republic':'DOM',
    'Ecuador':'ECU','Egypt':'EGY','El Salvador':'SLV','Eritrea':'ERI',
    'Estonia':'EST','Ethiopia':'ETH','Fiji':'FJI','Finland':'FIN',
    'France':'FRA','Gabon':'GAB','Gambia':'GMB','Georgia':'GEO',
    'Germany':'DEU','Ghana':'GHA','Greece':'GRC','Guatemala':'GTM',
    'Guinea':'GIN','Guinea-Bissau':'GNB','Guyana':'GUY','Haiti':'HTI',
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
    'South Africa':'ZAF','South Korea':'KOR','Republic of Korea':'KOR',
    'South Sudan':'SSD','Spain':'ESP','Sri Lanka':'LKA','Sudan':'SDN',
    'Suriname':'SUR','Sweden':'SWE','Switzerland':'CHE','Syria':'SYR',
    'Syrian Arab Republic':'SYR','Taiwan':'TWN','Tajikistan':'TJK',
    'Tanzania':'TZA','United Republic of Tanzania':'TZA','Thailand':'THA',
    'Timor-Leste':'TLS','Togo':'TGO','Trinidad and Tobago':'TTO',
    'Tunisia':'TUN','Turkey':'TUR','Türkiye':'TUR','Turkmenistan':'TKM',
    'Uganda':'UGA','Ukraine':'UKR','United Arab Emirates':'ARE',
    'United Kingdom':'GBR',
    'United Kingdom of Great Britain and Northern Ireland':'GBR',
    'United States of America':'USA','United States':'USA',
    'Uruguay':'URY','Uzbekistan':'UZB','Venezuela':'VEN',
    'Venezuela (Bolivarian Republic of)':'VEN',
    'Viet Nam':'VNM','Vietnam':'VNM','Yemen':'YEM','Zambia':'ZMB',
    'Zimbabwe':'ZWE','Eswatini':'SWZ','Swaziland':'SWZ',
    'Bhutan':'BTN','Comoros':'COM','Cabo Verde':'CPV','Cape Verde':'CPV',
    'Maldives':'MDV',
}

try:
    import pycountry
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
kcal_tot = (fbs_raw[(fbs_raw['Item']=='Grand Total') & (fbs_raw['element']=='kcal_pc_day')]
            [['country','year','value']].copy().rename(columns={'value':'kcal_pc_day'}))
IMP_ITEMS = ['Rice and products','Wheat and products','Maize and products',
             'Soyabeans','Vegetable Oils']
imp_kt = (fbs_raw[(fbs_raw['element']=='import_qty_kt') & fbs_raw['Item'].isin(IMP_ITEMS)]
          .groupby(['country','year'], as_index=False)['value'].sum()
          .rename(columns={'value':'staple_import_kt'}))
fbs = kcal_tot.merge(imp_kt, on=['country','year'], how='left')
fbs['iso3'] = fbs['country'].map(get_iso3)
fbs = fbs[fbs['iso3'].notna()].copy()
fbs['log_kcal_pc_day']       = np.log(fbs['kcal_pc_day'].clip(lower=1))
fbs['log_staple_import_kt']  = np.log(fbs['staple_import_kt'].clip(lower=1))
print(f"FBS clean: {len(fbs):,}r  {fbs['iso3'].nunique()} countries")

# ── 2. FTE v2 country panel ───────────────────────────────────────────────────
fte = pd.read_csv(os.path.join(ROOT, "outputs", "fte", "fte_country_panel_v2.csv"))
# columns: iso3, year, fte_total, fte_total_low, fte_total_high, fte_bartik_total
fte = fte.rename(columns={'fte_total_low':'fte_low','fte_total_high':'fte_high'})
fte['fte_total']        = fte['fte_total']        / 1000  # km² → 1000 km²
fte['fte_low']          = fte['fte_low']          / 1000
fte['fte_high']         = fte['fte_high']         / 1000
fte['fte_bartik_total'] = fte['fte_bartik_total'] / 1000
print(f"FTE v2: {len(fte):,}r  {fte['iso3'].nunique()} importers")

# ── 3. FTE v2 crop panel (for crop heterogeneity) ────────────────────────────
fte_crop = pd.read_csv(os.path.join(ROOT, "outputs", "fte", "fte_panel_v2.csv"))
# columns: year, iso3, crop, fte_central, fte_low, fte_high, n_suppliers, top_supplier_iso3, top_supplier_share, oil_crops_excluded
print(f"FTE crop v2: {len(fte_crop):,}r")

# ── 4. ENSO ───────────────────────────────────────────────────────────────────
enso = pd.read_csv(os.path.join(ROOT, "outputs", "outcome", "enso_panel.csv"))
print(f"ENSO: {len(enso):,}r  years {enso['year'].min()}–{enso['year'].max()}")

# ── 5. ERA5 domestic climate ──────────────────────────────────────────────────
era5 = pd.read_csv(os.path.join(ROOT, "outputs", "outcome", "era5_climate_panel.csv"))
era5 = era5[['iso3','year','t2m_anom_C','tp_anom_frac']].copy()
era5_countries = set(era5['iso3'].unique())
print(f"ERA5: {len(era5):,}r  {len(era5_countries)} countries")

# ── 6. GTA export restrictions ────────────────────────────────────────────────
gta_raw = pd.read_csv(os.path.join(ROOT, "outputs", "outcome", "gta_export_restrictions.csv"))
gta_raw['iso3'] = gta_raw['country'].map(get_iso3)
gta_raw = gta_raw[gta_raw['iso3'].notna()]
gta = (gta_raw.groupby(['iso3','year'], as_index=False)['export_restriction']
       .max().rename(columns={'iso3':'iso3_exporter','export_restriction':'any_export_restriction'}))
top_sup = (fte_crop[fte_crop['n_suppliers'] >= 1]
           [['iso3','year','top_supplier_iso3']].drop_duplicates())
top_sup = top_sup.merge(gta.rename(columns={'iso3_exporter':'top_supplier_iso3'}),
                        on=['top_supplier_iso3','year'], how='left')
top_sup['any_export_restriction'] = top_sup['any_export_restriction'].fillna(0)
gta_imp = (top_sup.groupby(['iso3','year'], as_index=False)['any_export_restriction']
           .max().rename(columns={'any_export_restriction':'supplier_export_restriction'}))
print(f"GTA importer-year: {len(gta_imp):,}r  restrictions: {gta_imp['supplier_export_restriction'].sum():.0f}")

# ── 7. WB income groups ───────────────────────────────────────────────────────
wb = pd.read_csv(os.path.join(ROOT, "data", "raw", "wb_income_groups.csv"))
print(f"WB income: {len(wb):,}r")

# ── 8. GDP per capita (WB WDI NY.GDP.PCAP.KD) ────────────────────────────────
GDP_CACHE = os.path.join(ROOT, "data", "raw", "wb_gdp_pc.csv")
if os.path.exists(GDP_CACHE):
    gdp_pc = pd.read_csv(GDP_CACHE)
    print(f"GDP_pc from cache: {len(gdp_pc):,}r")
else:
    print("Downloading GDP_pc from World Bank API (wbgapi)...")
    try:
        import wbgapi as wb_api
        raw_gdp = wb_api.data.DataFrame('NY.GDP.PCAP.KD', time=range(2000, 2022),
                                         labels=False, numericTimeKeys=True)
        raw_gdp = raw_gdp.reset_index()
        # pivot: rows=economy, cols=year
        gdp_long = raw_gdp.melt(id_vars=['economy'], var_name='year', value_name='gdp_pc')
        gdp_long.columns = ['iso3','year','gdp_pc']
        gdp_long['year'] = gdp_long['year'].astype(int)
        gdp_long = gdp_long.dropna(subset=['gdp_pc'])
        gdp_pc = gdp_long
        gdp_pc.to_csv(GDP_CACHE, index=False)
        print(f"GDP_pc downloaded: {len(gdp_pc):,}r  {gdp_pc['iso3'].nunique()} countries")
    except Exception as e:
        print(f"  ⚠️  wbgapi failed: {e}  — GDP_pc will be excluded from S2b/S3/S3b")
        gdp_pc = pd.DataFrame(columns=['iso3','year','gdp_pc'])

gdp_pc['log_gdp_pc'] = np.log(gdp_pc['gdp_pc'].clip(lower=1))

# ── 9. Herfindahl-Hirschman Index (import concentration per importer-crop-year) ──
# HHI = sum(s_i^2) where s_i = bilateral import share from supplier i
# Use fte_crop's top_supplier_share as proxy; full HHI needs BACI shares
BACI_PATH = os.path.join(ROOT, "outputs", "trade", "baci_trade_panel.csv")
hhi_panel = None
if os.path.exists(BACI_PATH):
    try:
        baci = pd.read_csv(BACI_PATH)
        # Normalise column names: importer → iso3_importer, value col → val
        imp_col = 'iso3_importer' if 'iso3_importer' in baci.columns else (
                  'importer_iso3' if 'importer_iso3' in baci.columns else 'importer')
        val_col = ('trade_value_usd' if 'trade_value_usd' in baci.columns else
                   'value_kusd' if 'value_kusd' in baci.columns else 'value')
        baci = baci.rename(columns={imp_col: 'iso3_importer', val_col: 'trade_value_usd'})
        # compute importer-crop-year total trade
        baci_crop_share = baci.copy()
        tot = (baci_crop_share.groupby(['year','iso3_importer','commodity'], as_index=False)
               ['trade_value_usd'].sum().rename(columns={'trade_value_usd':'tot_val'}))
        baci_crop_share = baci_crop_share.merge(tot, on=['year','iso3_importer','commodity'], how='left')
        baci_crop_share['share'] = baci_crop_share['trade_value_usd'] / baci_crop_share['tot_val'].clip(lower=1)
        hhi_crop = (baci_crop_share.groupby(['year','iso3_importer','commodity'])[['share']]
                    .apply(lambda df: (df['share']**2).sum())
                    .reset_index(name='hhi'))
        # aggregate to importer-year: mean HHI across commodities
        hhi_imp = hhi_crop.groupby(['year','iso3_importer'], as_index=False)['hhi'].mean()
        hhi_imp = hhi_imp.rename(columns={'iso3_importer':'iso3'})
        # quartile
        hhi_imp['hhi_q'] = pd.qcut(hhi_imp['hhi'], q=4, labels=['Q1','Q2','Q3','Q4'])
        hhi_panel = hhi_imp
        print(f"HHI panel: {len(hhi_panel):,}r  mean={hhi_panel['hhi'].mean():.3f}")
    except Exception as e:
        print(f"  ⚠️  HHI computation failed: {e}")

# ── 10. Master panel merge ────────────────────────────────────────────────────
panel = fbs[['iso3','year','log_kcal_pc_day','log_staple_import_kt',
             'kcal_pc_day','staple_import_kt']].copy()
panel = panel.merge(fte[['iso3','year','fte_total','fte_low','fte_high','fte_bartik_total']],
                    on=['iso3','year'], how='left')
panel = panel.merge(enso[['year','mei_annual_mean']], on='year', how='left')
panel = panel.merge(era5[['iso3','year','t2m_anom_C','tp_anom_frac']], on=['iso3','year'], how='left')
panel = panel.merge(gta_imp, on=['iso3','year'], how='left')
panel = panel.merge(wb[['iso3','income_level_id','income_level','region']], on='iso3', how='left')
panel = panel.merge(gdp_pc[['iso3','year','log_gdp_pc']], on=['iso3','year'], how='left')
if hhi_panel is not None:
    panel = panel.merge(hhi_panel[['iso3','year','hhi','hhi_q']], on=['iso3','year'], how='left')

panel['supplier_export_restriction'] = panel['supplier_export_restriction'].fillna(0)
panel['fte_total']                   = panel['fte_total'].fillna(0) 
panel['fte_low']                     = panel['fte_low'].fillna(0)
panel['fte_high']                    = panel['fte_high'].fillna(0)
panel['fte_bartik_total']            = panel['fte_bartik_total'].fillna(0)

# MEI × region interactions (same as v1)
for reg in ['EA','SA','SSA']:
    panel[f'mei_x_{reg}'] = panel['mei_annual_mean'] * (panel['region'] == reg).astype(float)

panel = panel[(panel['year'] >= 2000) & (panel['year'] <= 2021)].copy()

# Deduplicate
dup = panel.duplicated(subset=['iso3','year']).sum()
if dup > 0:
    print(f"  ⚠️  Deduplicating {dup} duplicate iso3/year rows")
    num_cols = panel.select_dtypes(include='number').columns.tolist()
    cat_cols = [c for c in panel.columns if c not in num_cols and c not in ['iso3','year']]
    agg = {c: 'mean' for c in num_cols}; agg.update({c: 'first' for c in cat_cols})
    panel = panel.groupby(['iso3','year'], as_index=False).agg(agg)

print(f"\nMaster panel: {len(panel):,}r  {panel['iso3'].nunique()} countries")
for col in ['log_kcal_pc_day','fte_total','mei_annual_mean','t2m_anom_C','tp_anom_frac',
            'log_gdp_pc','supplier_export_restriction']:
    miss = panel[col].isna().sum()
    pct  = miss / len(panel) * 100
    print(f"  {col}: {miss} missing ({pct:.1f}%)")

panel.to_csv(os.path.join(REG_DIR, "master_panel_v2.csv"), index=False)
print(f"✅ master_panel_v2.csv ({len(panel):,}r)")

# ── 11. ERA5 dropped-country diagnostic ──────────────────────────────────────
print("\n=== ERA5 DROPPED COUNTRIES ===")
full_sample_countries = set(panel.dropna(subset=['log_kcal_pc_day','fte_total'])['iso3'].unique())
era5_sample_countries = set(panel.dropna(subset=['log_kcal_pc_day','fte_total',
                                                  't2m_anom_C','tp_anom_frac'])['iso3'].unique())
dropped_by_era5 = sorted(full_sample_countries - era5_sample_countries)
print(f"  Full sample:   {len(full_sample_countries)} countries")
print(f"  ERA5 subsample:{len(era5_sample_countries)} countries")
print(f"  Dropped:       {len(dropped_by_era5)} countries")
era5_drop_df = pd.DataFrame({'iso3': dropped_by_era5})
era5_drop_df = era5_drop_df.merge(wb[['iso3','income_level_id','region']], on='iso3', how='left')
era5_drop_df.to_csv(os.path.join(REG_DIR, "era5_dropped_countries.csv"), index=False)
print(f"✅ era5_dropped_countries.csv ({len(era5_drop_df)}r)")
print(f"  Dropped by income: {era5_drop_df['income_level_id'].value_counts().to_dict()}")
print(f"  Dropped by region: {era5_drop_df['region'].value_counts().to_dict()}")

# ── 12. TWFE regression helper (TWO-WAY CLUSTERED) ────────────────────────────
try:
    from linearmodels import PanelOLS
    HAS_LM = True
except ImportError:
    HAS_LM = False
    print("⚠️  linearmodels not available — regressions will be skipped")

def run_twfe(df, y_col, x_cols, spec_name, drop_absorbed=False):
    """Two-way FE (entity + time) with two-way clustered SE (entity + time)."""
    if not HAS_LM:
        return None
    req = [y_col] + x_cols + ['iso3','year']
    sub = df[req].dropna().copy()
    num_x = [c for c in x_cols if pd.api.types.is_numeric_dtype(sub[c])]
    sub = sub[np.isfinite(sub[num_x]).all(axis=1)]
    sub = sub.drop_duplicates(subset=['iso3','year'])
    if len(sub) < 50:
        print(f"  {spec_name}: too few obs ({len(sub)}) — skip")
        return None
    try:
        idx = pd.MultiIndex.from_arrays([sub['iso3'], sub['year']])
        pdata = sub.set_index(idx)
        model = PanelOLS(pdata[y_col], pdata[x_cols],
                         entity_effects=True, time_effects=True,
                         drop_absorbed=drop_absorbed, check_rank=False)
        # Two-way clustered SE
        res = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
        fte_b  = res.params.get('fte_total', np.nan)
        fte_se = res.std_errors.get('fte_total', np.nan)
        fte_p  = res.pvalues.get('fte_total', np.nan)
        row = {
            'spec': spec_name, 'n_obs': res.nobs, 'n_countries': sub['iso3'].nunique(),
            'n_years': sub['year'].nunique(),
            'beta_fte': fte_b, 'se_fte': fte_se, 'pval_fte': fte_p,
            'ci95_lo': fte_b - 1.96*fte_se if not np.isnan(fte_b) else np.nan,
            'ci95_hi': fte_b + 1.96*fte_se if not np.isnan(fte_b) else np.nan,
            'rsquared_within': res.rsquared_within,
        }
        sig = '***' if fte_p < 0.01 else ('**' if fte_p < 0.05 else ('*' if fte_p < 0.10 else ''))
        print(f"  {spec_name}: β={fte_b:.5f} se={fte_se:.5f} p={fte_p:.3f}{sig}  N={res.nobs}")
        return row
    except Exception as e:
        print(f"  {spec_name}: ERROR — {e}")
        return None

# ── 13. Regression samples ────────────────────────────────────────────────────
# rescale FTE: already in 1000 km² units (divided by 1000 above)

# Sample definitions
base_req   = ['log_kcal_pc_day','fte_total']
era5_req   = base_req + ['t2m_anom_C','tp_anom_frac']
gdp_req    = base_req + ['log_gdp_pc']

reg_base = panel.dropna(subset=base_req).copy()
reg_era5 = panel.dropna(subset=era5_req).copy()
reg_gdp  = panel.dropna(subset=gdp_req).copy()
reg_full = panel.dropna(subset=era5_req + ['log_gdp_pc']).copy()

print(f"\nSample sizes: base={len(reg_base):,}  era5={len(reg_era5):,}  gdp={len(reg_gdp):,}  full={len(reg_full):,}")

# ── 14. Main regression specifications ───────────────────────────────────────
print("\n=== MAIN REGRESSIONS ===")
results = []

# S1: FTE + TWFE (full base sample)
s1 = run_twfe(reg_base, 'log_kcal_pc_day', ['fte_total'], 'S1_bivariate', drop_absorbed=True)
if s1: s1['description'] = 'FTE only'; results.append(s1)

# S2: + MEI×region (base sample)
s2 = run_twfe(reg_base, 'log_kcal_pc_day',
              ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA'], 'S2_ENSO', drop_absorbed=True)
if s2: s2['description'] = 'FTE + ENSO×region'; results.append(s2)

# S2b: + MEI×region + log_gdp_pc (gdp sample) [NEW]
s2b = run_twfe(reg_gdp, 'log_kcal_pc_day',
               ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','log_gdp_pc'], 'S2b_ENSO_GDP', drop_absorbed=True)
if s2b: s2b['description'] = 'FTE + ENSO×region + log_GDP_pc'; results.append(s2b)

# S3: + ERA5 (era5 subsample) — preferred spec
s3 = run_twfe(reg_era5, 'log_kcal_pc_day',
              ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac'],
              'S3_ERA5', drop_absorbed=True)
if s3: s3['description'] = 'FTE + ENSO×region + ERA5 (preferred)'; results.append(s3)

# S3b: + GDP_pc but no ERA5 (full gdp sample) [NEW]
s3b = run_twfe(reg_gdp, 'log_kcal_pc_day',
               ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','log_gdp_pc'],
               'S3b_ENSO_GDP_noERA5', drop_absorbed=True)
if s3b: s3b['description'] = 'FTE + ENSO×region + log_GDP_pc (no ERA5)'; results.append(s3b)

# S3_full: ERA5 + GDP_pc (full intersection sample)
s3_full = run_twfe(reg_full, 'log_kcal_pc_day',
                   ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA',
                    't2m_anom_C','tp_anom_frac','log_gdp_pc'],
                   'S3_full_ERA5_GDP', drop_absorbed=True)
if s3_full: s3_full['description'] = 'FTE + ENSO×region + ERA5 + log_GDP_pc'; results.append(s3_full)

# S4: + supplier_export_restriction (era5 subsample)
s4 = run_twfe(reg_era5, 'log_kcal_pc_day',
              ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac',
               'supplier_export_restriction'],
              'S4_ERA5_ER', drop_absorbed=True)
if s4: s4['description'] = 'FTE + ENSO×region + ERA5 + export_restriction'; results.append(s4)

# FTE bounds: low and high (S3 spec)
for bound, col_map in [('low','fte_low'), ('high','fte_high')]:
    sub = reg_era5.drop(columns=['fte_total'], errors='ignore').copy()
    sub = sub.rename(columns={col_map: 'fte_total'})
    r = run_twfe(sub, 'log_kcal_pc_day',
                 ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac'],
                 f'S3_fte_{bound}', drop_absorbed=True)
    if r: r['description'] = f'S3 with FTE {bound} bound'; results.append(r)

pd.DataFrame(results).to_csv(os.path.join(REG_DIR, "main_regression_results_v2.csv"), index=False)
print(f"✅ main_regression_results_v2.csv ({len(results)} specs)")

# ── 15. IV regression (Bartik instrument) ─────────────────────────────────────
print("\n=== IV REGRESSION (BARTIK) ===")
iv_results = []
if HAS_LM:
    try:
        from linearmodels import IV2SLS
        iv_req = era5_req + ['fte_bartik_total']
        sub_iv = panel.dropna(subset=iv_req).copy()
        sub_iv = sub_iv[np.isfinite(sub_iv[['fte_total','fte_bartik_total',
                                             't2m_anom_C','tp_anom_frac']]).all(axis=1)]
        sub_iv = sub_iv.drop_duplicates(subset=['iso3','year'])
        print(f"  IV sample: {len(sub_iv):,}r  {sub_iv['iso3'].nunique()} countries")

        # Demean for within estimator (entity + time FE via demeaning)
        for col in ['log_kcal_pc_day','fte_total','fte_bartik_total',
                    'mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac']:
            # within-entity demean
            sub_iv[f'{col}_wde'] = sub_iv[col] - sub_iv.groupby('iso3')[col].transform('mean')
        for col in ['log_kcal_pc_day_wde','fte_total_wde','fte_bartik_total_wde',
                    'mei_x_EA_wde','mei_x_SA_wde','mei_x_SSA_wde',
                    't2m_anom_C_wde','tp_anom_frac_wde']:
            # within-time demean (double-demeaning)
            sub_iv[col] = sub_iv[col] - sub_iv.groupby('year')[col].transform('mean')

        import statsmodels.api as sm
        # First stage: fte_total_wde ~ fte_bartik_total_wde + controls
        fs_X = sm.add_constant(sub_iv[['fte_bartik_total_wde','mei_x_EA_wde','mei_x_SA_wde',
                                        'mei_x_SSA_wde','t2m_anom_C_wde','tp_anom_frac_wde']])
        fs = sm.OLS(sub_iv['fte_total_wde'], fs_X).fit()
        fs_f = fs.fvalue
        print(f"  First stage F-stat (Bartik): {fs_f:.2f}")

        # Second stage: use IV2SLS on demeaned data (no FE needed after demeaning)
        dep_wde     = sub_iv['log_kcal_pc_day_wde']
        endog_wde   = sub_iv[['fte_total_wde']]
        # MEI×region are purely time-varying (same across all countries in a year)
        # → absorbed by time FE demeaning → drop from within-demeaned IV exog
        exog_wde    = sub_iv[['t2m_anom_C_wde','tp_anom_frac_wde']]
        instrum_wde = sub_iv[['fte_bartik_total_wde']]
        # Drop zero-variance columns
        exog_wde = exog_wde.loc[:, exog_wde.std() > 1e-10]
        iv_model = IV2SLS(dep_wde, exog_wde if len(exog_wde.columns) > 0 else None,
                          endog_wde, instrum_wde)
        iv_res   = iv_model.fit()
        fte_b  = float(iv_res.params.iloc[-1])  # last param is the endogenous variable
        fte_se = float(iv_res.std_errors.iloc[-1])
        fte_p  = float(iv_res.pvalues.iloc[-1])
        sig = '***' if fte_p < 0.01 else ('**' if fte_p < 0.05 else ('*' if fte_p < 0.10 else ''))
        print(f"  S4_IV: β={fte_b:.5f} se={fte_se:.5f} p={fte_p:.3f}{sig}  F={fs_f:.2f}")
        iv_results.append({
            'spec': 'S4_IV_Bartik',
            'description': 'IV: Bartik FTE instrument (demeaned 2SLS)',
            'n_obs': len(sub_iv),
            'n_countries': sub_iv['iso3'].nunique(),
            'beta_fte': fte_b, 'se_fte': fte_se, 'pval_fte': fte_p,
            'ci95_lo': fte_b - 1.96*fte_se,
            'ci95_hi': fte_b + 1.96*fte_se,
            'first_stage_F': fs_f,
        })
    except Exception as e:
        print(f"  IV regression failed: {e}")

pd.DataFrame(iv_results).to_csv(os.path.join(REG_DIR, "iv_results_v2.csv"), index=False)
print(f"✅ iv_results_v2.csv ({len(iv_results)} specs)")

# ── 16. Heterogeneity: income group ──────────────────────────────────────────
print("\n=== HETEROGENEITY: INCOME GROUP ===")
inc_results = []
base_x = ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac']
for grp_id, grp_label in [('LIC','Low income'),('LMC','Lower middle'),
                            ('UMC','Upper middle'),('HIC','High income')]:
    sub = reg_era5[reg_era5['income_level_id'] == grp_id].copy()
    if len(sub) < 50:
        print(f"  {grp_id}: too few obs ({len(sub)}) — skip")
        continue
    r = run_twfe(sub, 'log_kcal_pc_day', base_x, f'S3_{grp_id}', drop_absorbed=True)
    if r:
        r['income_group'] = grp_id
        r['income_label'] = grp_label
        inc_results.append(r)

pd.DataFrame(inc_results).to_csv(os.path.join(REG_DIR, "heterogeneity_income_v2.csv"), index=False)
print(f"✅ heterogeneity_income_v2.csv ({len(inc_results)} specs)")

# ── 17. Heterogeneity: crop ───────────────────────────────────────────────────
print("\n=== HETEROGENEITY: CROP ===")
crop_results = []
for crop in ['rice','wheat','maize','soybeans','oil_crops']:
    crop_col = f'fte_{crop}'
    fc = (fte_crop[fte_crop['crop'] == crop]
          .groupby(['iso3','year'], as_index=False)['fte_central'].sum()
          .rename(columns={'fte_central': crop_col}))
    fc[crop_col] = fc[crop_col] / 1000  # km² → 1000 km²
    sub = reg_era5.merge(fc, on=['iso3','year'], how='left')
    sub[crop_col] = sub[crop_col].fillna(0)
    sub = sub.drop_duplicates(subset=['iso3','year'])
    sub2 = sub.drop(columns=['fte_total','fte_low','fte_high'], errors='ignore')
    sub2 = sub2.rename(columns={crop_col: 'fte_total'})
    r = run_twfe(sub2, 'log_kcal_pc_day',
                 ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac'],
                 f'S3_{crop}', drop_absorbed=True)
    if r:
        r['crop'] = crop
        crop_results.append(r)

pd.DataFrame(crop_results).to_csv(os.path.join(REG_DIR, "heterogeneity_crop_v2.csv"), index=False)
print(f"✅ heterogeneity_crop_v2.csv ({len(crop_results)} specs)")

# ── 18. Heterogeneity: Herfindahl quartile ────────────────────────────────────
print("\n=== HETEROGENEITY: HERFINDAHL ===")
hhi_results = []
if hhi_panel is not None and 'hhi_q' in panel.columns:
    for q in ['Q1','Q2','Q3','Q4']:
        sub = reg_era5[reg_era5['hhi_q'] == q].copy()
        if len(sub) < 50:
            print(f"  HHI {q}: too few obs ({len(sub)}) — skip")
            continue
        r = run_twfe(sub, 'log_kcal_pc_day', base_x, f'S3_HHI_{q}', drop_absorbed=True)
        if r:
            r['hhi_quartile'] = q
            hhi_results.append(r)
else:
    print("  HHI panel not available — skipping")

pd.DataFrame(hhi_results).to_csv(os.path.join(REG_DIR, "heterogeneity_herfindahl_v2.csv"), index=False)
print(f"✅ heterogeneity_herfindahl_v2.csv ({len(hhi_results)} specs)")

# ── 19. Export restriction decomposition ─────────────────────────────────────
print("\n=== EXPORT RESTRICTION DECOMPOSITION ===")
decomp = []
if s3: decomp.append({**s3, 'note': 'S3_without_ER'})
if s4: decomp.append({**s4, 'note': 'S4_with_ER'})
if s3 and s4:
    diff = s3['beta_fte'] - s4['beta_fte']
    pct  = diff / s3['beta_fte'] * 100 if s3['beta_fte'] != 0 else np.nan
    print(f"  Direct β (S3): {s3['beta_fte']:.5f}")
    print(f"  Policy-adj β (S4): {s4['beta_fte']:.5f}")
    print(f"  Difference (policy channel): {diff:.5f} ({pct:.1f}% of direct)")
    decomp.append({'spec': 'decomp_policy_channel',
                   'note': 'S3_beta minus S4_beta',
                   'beta_fte': diff,
                   'pct_of_direct': pct})
pd.DataFrame(decomp).to_csv(os.path.join(REG_DIR, "export_restriction_decomposition_v2.csv"), index=False)
print(f"✅ export_restriction_decomposition_v2.csv")

# ── 20. Placebo tests ─────────────────────────────────────────────────────────
print("\n=== PLACEBO TESTS ===")
placebo_results = []

# Temporal: FTE leads (t+1, t+2, t+3)
for lead in [1, 2, 3]:
    reg_lead = reg_era5.copy()
    lead_col = f'fte_lead{lead}'
    reg_lead[lead_col] = reg_lead.groupby('iso3')['fte_total'].shift(-lead)
    reg_lead = reg_lead.dropna(subset=[lead_col, 't2m_anom_C'])
    reg_lead = reg_lead.drop_duplicates(subset=['iso3','year'])
    reg_lead2 = reg_lead.drop(columns=['fte_total','fte_low','fte_high'], errors='ignore')
    reg_lead2 = reg_lead2.rename(columns={lead_col: 'fte_total'})
    r = run_twfe(reg_lead2, 'log_kcal_pc_day',
                 ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac'],
                 f'Placebo_FTE_lead{lead}', drop_absorbed=True)
    if r:
        r['placebo_type'] = f'temporal_lead{lead}'
        placebo_results.append(r)

# Temporal: FTE lags (t-1, t-2)
for lag in [1, 2]:
    reg_lag = reg_era5.copy()
    lag_col = f'fte_lag{lag}'
    reg_lag[lag_col] = reg_lag.groupby('iso3')['fte_total'].shift(lag)
    reg_lag = reg_lag.dropna(subset=[lag_col, 't2m_anom_C'])
    reg_lag = reg_lag.drop_duplicates(subset=['iso3','year'])
    reg_lag2 = reg_lag.drop(columns=['fte_total','fte_low','fte_high'], errors='ignore')
    reg_lag2 = reg_lag2.rename(columns={lag_col: 'fte_total'})
    r = run_twfe(reg_lag2, 'log_kcal_pc_day',
                 ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac'],
                 f'Placebo_FTE_lag{lag}', drop_absorbed=True)
    if r:
        r['placebo_type'] = f'temporal_lag{lag}'
        placebo_results.append(r)

# Geographic: random FTE reassignment
np.random.seed(42)
reg_geo   = reg_era5.copy()
iso3_list = reg_era5['iso3'].unique()
iso3_shuf = iso3_list.copy()
np.random.shuffle(iso3_shuf)
iso_map   = dict(zip(iso3_list, iso3_shuf))
reg_geo['iso3_random'] = reg_geo['iso3'].map(iso_map)
fte_rand  = fte[['iso3','year','fte_total']].rename(columns={'iso3':'iso3_random',
                                                              'fte_total':'fte_rand'})
reg_geo   = reg_geo.drop(columns=['fte_total','fte_low','fte_high'], errors='ignore')
reg_geo   = reg_geo.merge(fte_rand, on=['iso3_random','year'], how='left')
reg_geo['fte_rand'] = reg_geo['fte_rand'].fillna(0)
reg_geo2  = reg_geo.rename(columns={'fte_rand': 'fte_total'})
r = run_twfe(reg_geo2, 'log_kcal_pc_day',
             ['fte_total','mei_x_EA','mei_x_SA','mei_x_SSA','t2m_anom_C','tp_anom_frac'],
             'Placebo_Geographic', drop_absorbed=True)
if r:
    r['placebo_type'] = 'geographic'
    placebo_results.append(r)

pd.DataFrame(placebo_results).to_csv(os.path.join(REG_DIR, "placebo_results_v2.csv"), index=False)
print(f"✅ placebo_results_v2.csv ({len(placebo_results)} tests)")

# ── 21. Summary ───────────────────────────────────────────────────────────────
print("\n=== ALL PHASE 3 v2 OUTPUTS ===")
outputs = [
    'master_panel_v2.csv',
    'main_regression_results_v2.csv',
    'heterogeneity_income_v2.csv',
    'heterogeneity_crop_v2.csv',
    'heterogeneity_herfindahl_v2.csv',
    'export_restriction_decomposition_v2.csv',
    'placebo_results_v2.csv',
    'iv_results_v2.csv',
    'era5_dropped_countries.csv',
]
for fname in outputs:
    path = os.path.join(REG_DIR, fname)
    exists = os.path.exists(path)
    rows = 0
    if exists:
        try: rows = len(pd.read_csv(path))
        except: pass
    print(f"  {'✅' if exists else '❌'} {fname} ({rows}r)")
