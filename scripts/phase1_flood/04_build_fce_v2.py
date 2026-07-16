"""
Phase 1 — FCE v2 Construction with Monthly Flood + Crop Calendar Filter
Methodological fix: applies growing-season filter using monthly flood data.
Uses cropland-fraction scaling from annual file to convert total→cropland flood.

Inputs:
  data/takyurp09/monthly_flood_area_fixed_2000_2021.csv
  outputs/fce/annual_flood_area_by_country.csv
  outputs/fce/crop_calendar_by_country_crop.csv
  outputs/fce/harvested_area_by_country_crop.csv
  outputs/fce/yield_anomaly_by_country_crop_year.csv
Outputs:
  outputs/fce/fce_final_panel_v2.csv
  outputs/fce/fce_comparison_v1_vs_v2.csv
"""
import os, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "takyurp09")
OUT  = os.path.join(ROOT, "outputs", "fce")

mfl = pd.read_csv(os.path.join(DATA, "monthly_flood_area_fixed_2000_2021.csv"))
ann = pd.read_csv(os.path.join(OUT,  "annual_flood_area_by_country.csv"))
ha  = pd.read_csv(os.path.join(OUT,  "harvested_area_by_country_crop.csv"))
cal = pd.read_csv(os.path.join(OUT,  "crop_calendar_by_country_crop.csv"))
yld = pd.read_csv(os.path.join(OUT,  "yield_anomaly_by_country_crop_year.csv"))

print(f"Monthly flood: {len(mfl):,}r  countries:{mfl['ADM0_NAME'].nunique()}")
print(f"HA: {len(ha):,}r  crops:{sorted(ha['crop'].unique())}")
print(f"Cal: {len(cal):,}r  Yield: {len(yld):,}r  years:{yld['year'].min()}-{yld['year'].max()}")

import geopandas as gpd
shp = gpd.read_file(os.path.join(ROOT,"data","country_shapes","ne_110m_admin_0_countries.shp"))
name2iso = dict(zip(shp['ADMIN'], shp['ISO_A3']))
name2iso.update({
    'Russia':'RUS','United States':'USA','China':'CHN','South Korea':'KOR',
    'North Korea':'PRK','Taiwan':'TWN','Syria':'SYR','Tanzania':'TZA',
    'Bolivia':'BOL','Venezuela':'VEN','Iran':'IRN','Turkey':'TUR',
    'Vietnam':'VNM','Laos':'LAO','Ivory Coast':'CIV',"Cote d'Ivoire":'CIV',
    'Kosovo':'XKX','North Macedonia':'MKD','Swaziland':'SWZ','eSwatini':'SWZ',
    'Cape Verde':'CPV','Congo':'COG','Democratic Republic of the Congo':'COD',
    'Central African Rep.':'CAF','Bosnia and Herz.':'BIH',
    'S. Sudan':'SSD','South Sudan':'SSD','Somaliland':'SOM','W. Sahara':'ESH',
    'Dem. Rep. Congo':'COD','Guinea-Bissau':'GNB','Trinidad and Tobago':'TTO',
})
def get_iso3(n): return name2iso.get(str(n).strip(), None)

NAME_FIX = {
    'Russian Federation':'Russia','United States of America':'United States',
    'China/India':'China','Lao PDR':'Laos','Viet Nam':'Vietnam',
    "Cote d'Ivoire":'Ivory Coast',"Côte d'Ivoire":'Ivory Coast',
    'The former Yugoslav Republic of Macedonia':'North Macedonia',
    'Bosnia and Herzegovina':'Bosnia and Herz.','eSwatini':'Swaziland',
    'Cabo Verde':'Cape Verde','Central African Republic':'Central African Rep.',
}
mfl['country'] = mfl['ADM0_NAME'].map(lambda x: NAME_FIX.get(str(x).strip(), str(x).strip()))
mfl['iso3'] = mfl['country'].map(get_iso3)
mfl_mapped = mfl[mfl['iso3'].notna()].copy()
print(f"Monthly rows with iso3: {len(mfl_mapped):,}/{len(mfl):,} ({mfl_mapped['iso3'].nunique()} countries)")

for df in [ha, cal, yld]:
    if 'iso3' not in df.columns:
        df['iso3'] = df['country'].map(get_iso3)

# Cropland fraction from annual file
ann_valid = ann[(ann['flooded_km2']>0) & ann['iso3'].notna()].copy()
ann_valid['cl_frac'] = (ann_valid['flooded_cropland_km2']/ann_valid['flooded_km2']).clip(0,1)
cropland_frac = ann_valid.groupby('iso3')['cl_frac'].mean()
med_clf = cropland_frac.median()
print(f"Cropland frac: {len(cropland_frac)} countries  median={med_clf:.4f}  mean={cropland_frac.mean():.4f}")

mfl_mapped = mfl_mapped.copy()
mfl_mapped['cl_frac'] = mfl_mapped['iso3'].map(cropland_frac).fillna(med_clf)
mfl_mapped['flooded_cropland_km2'] = mfl_mapped['flooded_km2'] * mfl_mapped['cl_frac']

# Winsorise yield anomaly
for cn in yld['crop'].unique():
    m = yld['crop']==cn
    v = yld.loc[m,'yield_anomaly_ensemble_mean']
    lo,hi = v.mean()-2.5*v.std(), v.mean()+2.5*v.std()
    yld.loc[m,'yield_anomaly_ensemble_mean'] = v.clip(lo,hi)
print("Yield anomaly winsorised")

# Pivot monthly cropland flood
mfl_pivot = mfl_mapped.pivot_table(
    index=['iso3','year'], columns='month',
    values='flooded_cropland_km2', aggfunc='sum'
).reset_index()
mfl_pivot.columns.name = None
for m in range(1,13):
    if m not in mfl_pivot.columns: mfl_pivot[m] = 0.0

# Annual total flood (total, not cropland-scaled) for reference column
ann_tot = mfl_mapped.groupby(['iso3','year'])['flooded_km2'].sum().reset_index()
ann_tot.rename(columns={'flooded_km2':'annual_flood_km2'}, inplace=True)
print(f"Flood pivot: {len(mfl_pivot):,} country-years")

def gs_months(pm, hm):
    if pd.isna(pm) or pd.isna(hm): return []
    pm,hm = int(pm),int(hm)
    return list(range(pm,hm+1)) if pm<=hm else list(range(pm,13))+list(range(1,hm+1))

# Deduplicate crop calendar
cal_dedup = (cal.dropna(subset=['iso3'])
               .sort_values(['iso3','crop'])
               .drop_duplicates(subset=['iso3','crop'], keep='first'))
print(f"Cal dedup: {len(cal_dedup):,}/{len(cal):,}")

panel = yld.merge(
    cal_dedup[['iso3','crop','plant_month','harvest_month']],
    on=['iso3','crop'], how='left'
).merge(
    ha[['iso3','crop','harvested_ha','harvested_km2','flood_adapted_exclude']].dropna(subset=['iso3']),
    on=['iso3','crop'], how='left'
)
print(f"Panel (4 crops): {len(panel):,}r")

oil_base = ha[ha['crop']=='oil_crops'].dropna(subset=['iso3']).copy()
oil_rows=[]
for _,row in oil_base.iterrows():
    for yr in range(2000,2022):
        oil_rows.append({'iso3':row['iso3'],'country':row['country'],'crop':'oil_crops',
            'year':yr,'yield_anomaly_ensemble_mean':np.nan,'missing_yield_flag':1,
            'plant_month':np.nan,'harvest_month':np.nan,
            'harvested_ha':row['harvested_ha'],'harvested_km2':row['harvested_km2'],
            'flood_adapted_exclude':row['flood_adapted_exclude']})
panel = pd.concat([panel, pd.DataFrame(oil_rows)], ignore_index=True)
cmap = ha.dropna(subset=['iso3']).set_index('iso3')['country'].to_dict()
panel['country'] = panel['country'].fillna(panel['iso3'].map(cmap))
print(f"Panel (5 crops): {len(panel):,}r")

panel = panel.merge(mfl_pivot[['iso3','year']+list(range(1,13))], on=['iso3','year'], how='left')
panel = panel.merge(ann_tot, on=['iso3','year'], how='left')
for m in range(1,13): panel[m] = panel[m].fillna(0.0)
panel['annual_flood_km2'] = panel['annual_flood_km2'].fillna(0.0)

panel['growing_season_flood_km2'] = panel.apply(
    lambda r: sum(r.get(m,0) for m in gs_months(r.get('plant_month'),r.get('harvest_month'))), axis=1)
panel['ann_cropland_flood_km2'] = panel[[1,2,3,4,5,6,7,8,9,10,11,12]].sum(axis=1)
panel['growing_season_fraction'] = np.where(
    panel['ann_cropland_flood_km2']>0,
    panel['growing_season_flood_km2']/panel['ann_cropland_flood_km2'], 0.0)

mask_gs = (panel['crop']!='oil_crops') & (panel['ann_cropland_flood_km2']>0)
print(f"GS fraction (non-oil, non-zero): mean={panel.loc[mask_gs,'growing_season_fraction'].mean():.3f}")

tot_ha = ha.dropna(subset=['iso3']).groupby('iso3')['harvested_ha'].sum().reset_index()
tot_ha.rename(columns={'harvested_ha':'total_harvested_ha'}, inplace=True)
panel = panel.merge(tot_ha, on='iso3', how='left')
panel['harvested_ha'] = pd.to_numeric(panel['harvested_ha'],errors='coerce').fillna(0)
panel['total_harvested_ha'] = pd.to_numeric(panel['total_harvested_ha'],errors='coerce').fillna(0)
panel['harvested_share'] = np.where(panel['total_harvested_ha']>0,
                                     panel['harvested_ha']/panel['total_harvested_ha'], 0.0)

panel['yield_anomaly_weight'] = np.where(
    panel['crop']=='oil_crops', np.nan,
    np.where(panel['missing_yield_flag']==1, 1.0,
             np.where(panel['yield_anomaly_ensemble_mean']<0,
                      np.abs(panel['yield_anomaly_ensemble_mean']), 0.0)))

panel['fce_central'] = (panel['growing_season_flood_km2']
                        * panel['yield_anomaly_weight'].fillna(0)
                        * panel['harvested_share'])
panel['fce_low']  = panel['fce_central'] * 0.7
panel['fce_high'] = panel['fce_central'] * 1.3

oil_m  = panel['crop']=='oil_crops'
excl_m = panel['flood_adapted_exclude'].fillna(False).astype(bool)
tiny_m = panel['harvested_ha'] < 100
nocal_m= panel['plant_month'].isna() & ~oil_m
for col in ('fce_low','fce_central','fce_high'):
    panel.loc[oil_m|excl_m|tiny_m|nocal_m, col] = 0.0
panel['oil_crops_excluded'] = oil_m.astype(int)

out_cols = ['iso3','crop','year','fce_low','fce_central','fce_high',
            'growing_season_flood_km2','annual_flood_km2',
            'growing_season_fraction','yield_anomaly_weight',
            'missing_yield_flag','oil_crops_excluded','harvested_ha','harvested_share']
fce_v2 = panel[out_cols].sort_values(['iso3','crop','year']).reset_index(drop=True)
fce_v2.to_csv(os.path.join(OUT,"fce_final_panel_v2.csv"), index=False)
nz = (fce_v2['fce_central']>0).sum()
print(f"\nfce_final_panel_v2.csv: {len(fce_v2):,}r  non-zero:{nz:,} ({nz/len(fce_v2):.1%})")
print(f"  FCE central max: {fce_v2['fce_central'].max():.2f} km²")
print("  Top 10:")
print(fce_v2.sort_values('fce_central',ascending=False).head(10)[
    ['iso3','crop','year','fce_central','growing_season_flood_km2','growing_season_fraction']
].to_string(index=False))

v1_path = os.path.join(OUT,"fce_final_panel.csv")
if os.path.exists(v1_path):
    v1 = pd.read_csv(v1_path)
    comp = (v1[['iso3','crop','year','fce_central']].rename(columns={'fce_central':'fce_v1'})
            .merge(fce_v2[['iso3','crop','year','fce_central']].rename(columns={'fce_central':'fce_v2'}),
                   on=['iso3','crop','year'], how='outer'))
    comp[['fce_v1','fce_v2']] = comp[['fce_v1','fce_v2']].fillna(0)
    comp['pct_change'] = np.where(comp['fce_v1']>0,
                                   (comp['fce_v2']-comp['fce_v1'])/comp['fce_v1']*100, np.nan)
    comp['direction'] = np.where(comp['fce_v2']>comp['fce_v1']*1.01,'higher',
                         np.where(comp['fce_v2']<comp['fce_v1']*0.99,'lower','unchanged'))
    comp.to_csv(os.path.join(OUT,"fce_comparison_v1_vs_v2.csv"), index=False)
    pct = comp['pct_change'].dropna()
    print(f"\nComparison: {len(comp):,}r  mean_delta={pct.mean():.1f}%  median_delta={pct.median():.1f}%")
    print(f"  Direction: {comp['direction'].value_counts().to_dict()}")
    print("  By major exporter:")
    for iso in sorted({'USA','IND','CHN','BRA','ARG','RUS','UKR','AUS','THA','PAK','IDN','CAN','FRA','DEU'}):
        mv = comp[comp['iso3']==iso]['pct_change'].dropna().mean()
        print(f"    {iso}: {mv:.1f}%")
    print("  By crop:")
    for cr in ['rice','wheat','maize','soybeans']:
        mv = comp[comp['crop']==cr]['pct_change'].dropna().mean()
        print(f"    {cr}: {mv:.1f}%")

print("\nFCE v2 build complete.")
