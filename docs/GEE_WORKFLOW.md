# Google Earth Engine Workflow

The Earth Engine portion of the workflow converts global flood products into country-crop-year exposure panels.

## Main Tasks

| Step | Purpose | Representative script |
|---|---|---|
| Flood extraction | Aggregate inundation layers to country-year or country-month units | `scripts/phase1_flood/01_jrc_flood_extraction_gee.py` |
| JRC / GFD processing | Download and intersect flood products with analysis units | `scripts/phase1_flood/01b_jrc_gfd_download_intersect.py` |
| Crop weights | Harmonize harvested area and crop-calendar information | `scripts/phase1_flood/02_prep_harvested_calendar_yield.py` |
| Exposure panel | Build flood cropland exposure measures | `scripts/phase1_flood/03_build_fce_panel.py` and `scripts/phase1_flood/04_build_fce_v2.py` |
| Adaptation mask | Create exposure variants that account for repeated inundation | `scripts/phase1_flood/05_build_adapted_mask.py` |

## Scale Demonstrated

The pipeline is designed for global country-crop-year processing across multiple staple and traded crop groups. It combines satellite flood layers, crop calendars, harvested-area weights, and trade networks into reproducible panel datasets.

## Credentials

Do not commit credentials. Use environment variables or local authentication flows for:

- Google Earth Engine
- NASA Earthdata
- IPC API access, if used
- Any other provider-specific tokens

Example:

```bash
export EARTHDATA_TOKEN="..."
export IPC_API_TOKEN="..."
```

These values should live in a private shell profile or untracked `.env` file.
