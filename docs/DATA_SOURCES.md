# Data Sources

This project combines remote-sensing, agricultural, trade, climate, and food-security data.

## Flood and Cropland Exposure

| Source | Role | Notes |
|---|---|---|
| JRC Global Flood Database / Global Surface Water | Flood detection and water-history layers | Accessed through Google Earth Engine where possible |
| MapSPAM harvested area | Crop-specific spatial weights | Used to translate inundation into crop-exposure measures |
| Crop-calendar data | Growing-season exposure filter | Aligns floods with crop vulnerability windows |
| ISIMIP crop-yield anomalies | Damage-intensity and projection inputs | Used as historical or forward-looking crop-shock information |
| Dartmouth Flood Observatory | Supporting validation or gap-checking | Event-level flood catalog, not the primary pixel layer |

## Trade, Food, and Policy Variables

| Source | Role |
|---|---|
| BACI bilateral trade data | Import-weighted exposure to flood-affected exporters |
| FAO Food Balance Sheets | Caloric availability and food-supply outcomes |
| Global Trade Alert | Export-restriction and policy-control variables |
| IPC food-security classifications | Food-security validation and vulnerability checks |

## Climate and Validation Inputs

| Source | Role |
|---|---|
| ERA5 | Domestic climate controls |
| Multivariate ENSO Index | Global climate-cycle controls |
| USDA PSD | Production validation |
| FAO GIEWS | Crop assessment validation |
| MODIS NDVI | Vegetation-response validation |
| ISIMIP3b / ESGF products | Forward-looking hydrologic and crop-risk projections |

## Public Repository Policy

Raw data are excluded because several source datasets require separate downloads, licenses, registration, or API access. Users should acquire data directly from the original providers and configure local paths through environment variables or a local configuration file that is not committed.
