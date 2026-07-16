# Flood Cropland and Food-Trade Risk

This repository contains a public portfolio version of a research pipeline linking satellite-detected flood exposure in crop-producing regions to food-import risk through international trade networks.

The workflow is designed for applied economics, agricultural economics, and climate-risk settings where the empirical object is not only local flood damage, but also how production shocks propagate through commodity markets and import dependence.

## Research Workflow

1. Construct flood cropland exposure from Earth observation products and harvested-area weights.
2. Harmonize crop calendars, crop groups, and country-year panels.
3. Build bilateral food-trade exposure measures using BACI trade flows.
4. Merge flood-trade exposure with food-balance, climate, policy, and food-security outcomes.
5. Run panel regressions, robustness checks, validation exercises, and forward-looking risk projections.

## Repository Structure

```text
scripts/
  phase1_flood/          Flood, crop-calendar, harvested-area, and yield-anomaly processing
  phase2_trade/          BACI trade ingestion and flood-trade exposure construction
  phase3_outcome/        FAO, ENSO, ERA5, trade-policy, and food-security data pulls
  phase3_panel/          Regression-panel construction and empirical specifications
  phase4_validation/     External validation using production and vegetation indicators
  phase4_ipc/            Food-security validation workflow
  phase5_projection*/    Forward-looking exposure and caloric-risk projection scripts
  figures/               Figure-building scripts
docs/
  DATA_SOURCES.md        Data provenance and access notes
  GEE_WORKFLOW.md        Google Earth Engine workflow notes
  REPRODUCIBILITY.md     Reproduction workflow and version-control practice
  PORTFOLIO_NOTES.md     Public-release scope
  PUBLIC_RELEASE_CHECKLIST.md  Checklist for safe public updates
figures/selected/
  Selected public-facing diagnostics and workflow visuals
```

## Selected Visuals

![Global flood-trade exposure map](figures/selected/global_flood_trade_exposure_map.png)

![Portfolio workflow](figures/selected/workflow_overview.svg)

## Data Availability

Raw data are not included. The pipeline relies on public or access-controlled sources such as Google Earth Engine flood products, MapSPAM harvested area, crop calendars, ISIMIP, BACI, FAO Food Balance Sheets, ERA5, ENSO indices, Global Trade Alert, IPC, USDA PSD, and FAO GIEWS.

See `docs/DATA_SOURCES.md` for details.

## Reproducibility

This public version is intended to demonstrate coding style, project organization, geospatial processing, and version-control practice. Some scripts require registered API access or manually downloaded data.

See `docs/REPRODUCIBILITY.md` for environment setup and execution order.

## Public-Release Scope

This repository excludes raw data, unpublished manuscript files, private research notes, local logs, private tokens, and full thesis outputs. Selected figures are included only to document workflow scale and coding capability.
