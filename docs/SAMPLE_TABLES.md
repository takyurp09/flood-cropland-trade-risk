# Sample Tables

The following examples are small real-data extracts from processed intermediate products. They are included to document data structure and variable definitions, not to release the full dataset or final empirical result tables.

## Flood Cropland Exposure

| exporter_iso3 | crop | year | fce_low_km2 | fce_central_km2 | fce_high_km2 | harvested_area_ha | plant_month | harvest_month | flooded_cropland_km2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CHN | wheat | 2018 | 1762.1 | 2517.3 | 3272.4 | 31050072.5 | 3 | 7 | 12856.5 |
| IND | rice | 2018 | 628.0 | 897.2 | 1166.3 | 6529105.5 | 6 | 11 | 15332.0 |
| USA | maize | 2018 | 57.5 | 82.1 | 106.7 | 1334175.2 | 5 | 10 | 6809.6 |

## Flood Trade Exposure

| importer_iso3 | crop | year | fte | fte_low | fte_high | n_suppliers | top_supplier_iso3 | top_supplier_share |
|---|---|---:|---:|---:|---:|---:|---|---:|
| HND | maize | 2018 | 82.0148 | 57.4103 | 106.6192 | 9 | USA | 0.998 |
| NPL | rice | 2018 | 1778.8855 | 1245.2198 | 2312.5511 | 12 | IND | 0.990 |
| QAT | wheat | 2018 | 2533.5911 | 1773.5138 | 3293.6684 | 26 | RUS | 0.404 |

## Selected Panel Fields

| country_iso3 | year | kcal_supply_pc_day | flood_trade_exposure | domestic_temperature_anomaly_C | export_restriction_flag | income_group |
|---|---:|---:|---:|---:|---:|---|
| AFG | 2018 | 2261.79 | 683.2786 | 0.206 | 1.0 | LIC |
| HND | 2018 | 2633.02 | 2149.5848 | 0.156 | 0.0 | LMC |
| NPL | 2018 | 2854.67 | 3762.8556 | 0.189 | 1.0 | LMC |

## Interpretation

- `fce_central_km2`: central flood cropland exposure measure for an exporter-crop-year.
- `fce_low_km2` and `fce_high_km2`: lower and upper sensitivity measures used in robustness checks.
- `fte`: import-weighted exposure to flooded crop suppliers.
- `top_supplier_share`: import share of the largest supplier in the importer-crop-year record.
- `flood_trade_exposure`: selected country-year exposure field used in the applied-economics panel.
