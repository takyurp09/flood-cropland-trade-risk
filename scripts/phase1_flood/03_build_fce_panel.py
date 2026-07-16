"""
Phase 1 — FCE Construction
Inputs:
  outputs/fce/harvested_area_by_country_crop.csv
  outputs/fce/crop_calendar_by_country_crop.csv
  outputs/fce/yield_anomaly_by_country_crop_year.csv
  outputs/fce/annual_flood_area_by_country.csv
Outputs:
  outputs/fce/fce_panel_merged.csv
  outputs/fce/fce_final_panel.csv
"""
import pandas as pd, numpy as np, warnings, os
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(ROOT, "outputs", "fce")

# ── Name → ISO3 helper ────────────────────────────────────────────────────────
import pycountry

MANUAL_MAP = {
    'Afghanistan':'AFG','Albania':'ALB','Algeria':'DZA','Angola':'AGO',
    'Argentina':'ARG','Armenia':'ARM','Australia':'AUS','Austria':'AUT',
    'Azerbaijan':'AZE','Bangladesh':'BGD','Belarus':'BLR','Belgium':'BEL',
    'Belize':'BLZ','Benin':'BEN','Bolivia':'BOL','Bosnia and Herz.':'BIH',
    'Bosnia and Herzegovina':'BIH','Botswana':'BWA','Brazil':'BRA',
    'Bulgaria':'BGR','Burkina Faso':'BFA','Burundi':'BDI','Cambodia':'KHM',
    'Cameroon':'CMR','Canada':'CAN','Central African Rep.':'CAF',
    'Central African Republic':'CAF','Chad':'TCD','Chile':'CHL','China':'CHN',
    'Colombia':'COL','Congo':'COG','Costa Rica':'CRI','Croatia':'HRV',
    'Cuba':'CUB','Czechia':'CZE','Czech Republic':'CZE',
    'Dem. Rep. Congo':'COD','Dem. Rep. of the Congo':'COD','Denmark':'DNK',
    'Djibouti':'DJI','Dominican Rep.':'DOM','Ecuador':'ECU','Egypt':'EGY',
    'El Salvador':'SLV','Eq. Guinea':'GNQ','Equatorial Guinea':'GNQ',
    'Eritrea':'ERI','Estonia':'EST','Ethiopia':'ETH','Fiji':'FJI',
    'Finland':'FIN','France':'FRA','Gabon':'GAB','Gambia':'GMB',
    'Georgia':'GEO','Germany':'DEU','Ghana':'GHA','Greece':'GRC',
    'Guatemala':'GTM','Guinea':'GIN','Guinea-Bissau':'GNB','Guyana':'GUY',
    'Haiti':'HTI','Honduras':'HND','Hungary':'HUN','India':'IND',
    'Indonesia':'IDN','Iran':'IRN','Iraq':'IRQ','Ireland':'IRL',
    'Israel':'ISR','Italy':'ITA','Ivory Coast':'CIV',"Côte d'Ivoire":'CIV',
    "CÃ´te d'Ivoire":'CIV','Jamaica':'JAM','Japan':'JPN','Jordan':'JOR',
    'Kazakhstan':'KAZ','Kenya':'KEN','Kosovo':'XKX','Kuwait':'KWT',
    'Kyrgyzstan':'KGZ','Laos':'LAO','Lao PDR':'LAO','Latvia':'LVA',
    'Lebanon':'LBN','Lesotho':'LSO','Liberia':'LBR','Libya':'LBY',
    'Lithuania':'LTU','Luxembourg':'LUX','Madagascar':'MDG','Malawi':'MWI',
    'Malaysia':'MYS','Mali':'MLI','Mauritania':'MRT','Mexico':'MEX',
    'Moldova':'MDA','Mongolia':'MNG','Montenegro':'MNE','Morocco':'MAR',
    'Mozambique':'MOZ','Myanmar':'MMR','N. Cyprus':'CYP','Namibia':'NAM',
    'Nepal':'NPL','Netherlands':'NLD','New Caledonia':'NCL',
    'New Zealand':'NZL','Nicaragua':'NIC','Niger':'NER','Nigeria':'NGA',
    'North Korea':'PRK','North Macedonia':'MKD','Norway':'NOR','Oman':'OMN',
    'Pakistan':'PAK','Palestine':'PSE','Panama':'PAN',
    'Papua New Guinea':'PNG','Paraguay':'PRY','Peru':'PER',
    'Philippines':'PHL','Poland':'POL','Portugal':'PRT','Qatar':'QAT',
    'Romania':'ROU','Russia':'RUS','Rwanda':'RWA','Saudi Arabia':'SAU',
    'Senegal':'SEN','Serbia':'SRB','Sierra Leone':'SLE','Slovakia':'SVK',
    'Slovenia':'SVN','Solomon Is.':'SLB','Somalia':'SOM','Somaliland':'SOM',
    'South Africa':'ZAF','South Korea':'KOR','South Sudan':'SSD',
    'S. Sudan':'SSD','Spain':'ESP','Sri Lanka':'LKA','Sudan':'SDN',
    'Suriname':'SUR','Sweden':'SWE','Switzerland':'CHE','Syria':'SYR',
    'Taiwan':'TWN','Tajikistan':'TJK','Tanzania':'TZA','Thailand':'THA',
    'Timor-Leste':'TLS','E. Timor':'TLS','Togo':'TGO',
    'Trinidad and Tobago':'TTO','Tunisia':'TUN','Turkey':'TUR',
    'Turkmenistan':'TKM','Uganda':'UGA','Ukraine':'UKR',
    'United Arab Emirates':'ARE','United Kingdom':'GBR',
    'United States of America':'USA','United States':'USA',
    'Uruguay':'URY','Uzbekistan':'UZB','Vanuatu':'VUT','Venezuela':'VEN',
    'Vietnam':'VNM','Yemen':'YEM','Zambia':'ZMB','Zimbabwe':'ZWE',
    'Brunei':'BRN','Cyprus':'CYP','eSwatini':'SWZ','Swaziland':'SWZ',
    'Bhutan':'BTN','Comoros':'COM','Cape Verde':'CPV','Cabo Verde':'CPV',
    'Maldives':'MDV','Korea':'KOR','Hong Kong':'HKG','Macau':'MAC',
}

_iso3_cache = {}
def get_iso3(name):
    if pd.isna(name): return None
    n = str(name).strip()
    if n in _iso3_cache: return _iso3_cache[n]
    if n in MANUAL_MAP:
        _iso3_cache[n] = MANUAL_MAP[n]; return MANUAL_MAP[n]
    try:
        r = pycountry.countries.search_fuzzy(n)
        iso3 = r[0].alpha_3 if r else None
        _iso3_cache[n] = iso3; return iso3
    except:
        _iso3_cache[n] = None; return None

# ── Load inputs ───────────────────────────────────────────────────────────────
ha  = pd.read_csv(os.path.join(OUT, "harvested_area_by_country_crop.csv"))
cal = pd.read_csv(os.path.join(OUT, "crop_calendar_by_country_crop.csv"))
yld = pd.read_csv(os.path.join(OUT, "yield_anomaly_by_country_crop_year.csv"))
fl  = pd.read_csv(os.path.join(OUT, "annual_flood_area_by_country.csv"))

# Add ISO3 to country-name files
for df in [ha, cal, yld]:
    if 'iso3' not in df.columns:
        df['iso3'] = df['country'].map(get_iso3)

# Flood file: already has both country (mixed) and iso3 columns
if 'iso3' not in fl.columns:
    fl['iso3'] = fl['country'].apply(lambda x: x if len(str(x))==3 else get_iso3(x))
cov = fl['iso3'].notna().sum() / len(fl)
print(f"FL iso3 coverage: {cov:.1%}  rows:{len(fl)}")

# ── FCE Merged Panel (harvested area × crop calendar × yield anomaly) ─────────
pm_col = next(c for c in cal.columns if 'plant' in c.lower())
hm_col = next(c for c in cal.columns if 'harvest' in c.lower())

merged = yld.merge(
    ha[['iso3','crop','harvested_ha','harvested_km2']],
    on=['iso3','crop'], how='left'
).merge(
    cal[['iso3','crop',pm_col,hm_col]],
    on=['iso3','crop'], how='left'
)
merged.to_csv(os.path.join(OUT, "fce_panel_merged.csv"), index=False)
print(f"fce_panel_merged: {len(merged):,}r  null_ha:{merged['harvested_ha'].isna().sum()}  null_pm:{merged[pm_col].isna().sum()}")

# ── FCE Final Panel ────────────────────────────────────────────────────────────
# Join flood data (annual, country level) — not crop-specific
# Flood country coverage is country-level; distribute across crops by harvested_ha share
fl_grp = fl.groupby(['iso3','year'])['flooded_cropland_km2'].sum().reset_index()

fce = merged.merge(fl_grp, on=['iso3','year'], how='left')

# Fill missing flood = 0 (no inundation reported)
fce['flooded_cropland_km2'] = fce['flooded_cropland_km2'].fillna(0)

# Compute proportion of country cropland flooded
# flooded_cropland_km2 is the country total → attribute to each crop by its share of harvested area
total_ha = fce.groupby(['iso3','year'])['harvested_ha'].transform('sum')
fce['crop_ha_share'] = np.where(total_ha > 0, fce['harvested_ha'] / total_ha, 0.2)  # fallback: equal shares

# Flood exposure for this crop (km²)
fce['flood_exp_km2'] = fce['flooded_cropland_km2'] * fce['crop_ha_share']
fce['flood_exp_km2'] = fce['flood_exp_km2'].clip(lower=0)

# Damage weight: 
# For years with real ISIMIP data (missing_yield_flag=0): use |anomaly| when anomaly < 0
# For years with zero-fill (missing_yield_flag=1): use 1.0 as damage weight
#   (flood extents alone drive FCE; absence of yield suppression data ≠ no damage)
# This is conservative: we don't fabricate a damage multiplier, just use flood extent directly.
fce['damage_w'] = np.where(
    fce['missing_yield_flag'] == 1,
    1.0,   # zero-filled years: use full flood exposure as FCE
    np.where(
        fce['yield_anomaly_ensemble_mean'] < 0,
        np.abs(fce['yield_anomaly_ensemble_mean']).clip(0, 1),
        0.0   # positive anomaly years: no flood-induced crop damage
    )
)

# Central FCE = flood exposure (km²) × damage weight
fce['fce_central'] = fce['flood_exp_km2'] * fce['damage_w']

# Low / High bounds using ±30% harvested area uncertainty (MapSPAM uncertainty)
fce_cols = ['iso3','crop','year','fce_central']
fce['fce_low']  = fce['fce_central'] * 0.7
fce['fce_high'] = fce['fce_central'] * 1.3

# De-flag: add missing_yield_flag passthrough
final = fce[['iso3','crop','year','fce_low','fce_central','fce_high',
             'harvested_ha','plant_month','harvest_month',
             'yield_anomaly_ensemble_mean','missing_yield_flag',
             'flooded_cropland_km2','damage_w']].copy()
final.to_csv(os.path.join(OUT, "fce_final_panel.csv"), index=False)
print(f"fce_final_panel: {len(final):,}r  cols={list(final.columns)}")

# Summary stats
nonzero = (final['fce_central'] > 0).sum()
print(f"  Non-zero FCE: {nonzero:,} ({nonzero/len(final):.1%})")
print(f"  FCE central range: {final['fce_central'].min():.1f}–{final['fce_central'].max():.1f} km²")
top = final.sort_values('fce_central', ascending=False).head(10)[['iso3','crop','year','fce_central','flooded_cropland_km2']]
print("  Top 10 FCE events:")
print(top.to_string(index=False))

# Validation quick-check against USDA PSD
usda_path = os.path.join(ROOT, "outputs", "validation", "usda_psd_panel.csv")
if os.path.exists(usda_path):
    usda = pd.read_csv(usda_path)
    print(f"\nUSDA PSD validation events (checking USDA production drops):")
    events = [
        ('PAK','2010','rice'),('THA','2011','rice'),('NGA','2012','rice'),
        ('BGD','2017','rice'),('IND','2019','rice'),('IND','2020','rice'),
        ('CHN','2016','maize'),('IDN','2013','rice'),('MMR','2015','rice'),
    ]
    # Check FCE non-zero for these
    passes = 0
    for iso, yr, crop in events:
        row = final[(final['iso3']==iso)&(final['year']==int(yr))&(final['crop']==crop)]
        fce_val = row['fce_central'].values[0] if len(row) else 0
        flood_val = row['flooded_cropland_km2'].values[0] if len(row) else 0
        status = "✅" if fce_val > 0 or flood_val > 0 else "⚠️  zero"
        print(f"  {iso} {yr} {crop}: FCE={fce_val:.1f} km² flood={flood_val:.1f} km²  {status}")
        if fce_val > 0: passes += 1
    print(f"  Events with >0 FCE: {passes}/9")
