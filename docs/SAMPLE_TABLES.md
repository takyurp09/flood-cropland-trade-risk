# Sample Tables

The following examples are illustrative schemas only. They are not empirical results and should not be interpreted as reported estimates.

## Flood Cropland Exposure

| exporter_iso3 | crop | year | flooded_cropland_km2 | harvested_area_km2 | exposure_share | sensor_source |
|---|---|---:|---:|---:|---:|---|
| AAA | rice | 2018 | 125.4 | 9810.0 | 0.0128 | JRC_GFD |
| BBB | wheat | 2019 | 74.2 | 6230.0 | 0.0119 | JRC_GSW |
| CCC | maize | 2020 | 211.8 | 14520.0 | 0.0146 | JRC_GFD |

## Flood Trade Exposure

| importer_iso3 | crop | year | top_exposed_exporter | import_share | exporter_exposure_share | flood_trade_exposure |
|---|---|---:|---|---:|---:|---:|
| DDD | rice | 2018 | AAA | 0.42 | 0.0128 | 0.0054 |
| EEE | wheat | 2019 | BBB | 0.31 | 0.0119 | 0.0037 |
| FFF | maize | 2020 | CCC | 0.27 | 0.0146 | 0.0039 |

## Regression Panel Schema

| country_iso3 | year | kcal_supply_pc_day | flood_trade_exposure | domestic_climate_index | export_restriction_flag | ipc_observed |
|---|---:|---:|---:|---:|---:|---:|
| DDD | 2018 | 2410 | 0.0054 | 0.18 | 0 | 1 |
| EEE | 2019 | 2275 | 0.0037 | -0.05 | 1 | 0 |
| FFF | 2020 | 2560 | 0.0039 | 0.22 | 0 | 1 |

## Interpretation

- `flooded_cropland_km2`: area of crop-weighted flooded cropland.
- `exposure_share`: flooded cropland divided by harvested area.
- `flood_trade_exposure`: import-weighted exposure to flooded exporters.
- `ipc_observed`: indicator for whether IPC data are observed for validation in a country-year.
