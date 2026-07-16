# Method Overview

This project builds exposure measures that connect satellite-observed floods in crop-producing areas to food-import exposure in downstream countries.

## Core Objects

| Object | Unit | Description |
|---|---|---|
| Flood cropland exposure | exporter country x crop x year | Flooded harvested area during crop-relevant growing windows |
| Flood trade exposure | importer country x crop x year | Import-weighted exposure to flooded exporters |
| Outcome panel | country x year | Food-balance, food-security, climate, and policy variables merged with exposure measures |

## Conceptual Steps

1. Use satellite flood products to identify flooded area by country and period.
2. Intersect flood layers with harvested-area and crop-calendar information.
3. Convert local flood exposure into crop-country-year exposure measures.
4. Weight exporter exposure by bilateral commodity trade shares.
5. Merge exposure measures with food-supply and food-security outcomes.
6. Validate exposure measures against independent crop, production, vegetation, and food-security sources.

## Why This Is Useful

The pipeline supports applied questions where a flood in an exporting region can matter for consumers elsewhere through trade dependence. This structure is relevant for agricultural monitoring, food-security early warning, climate-risk analysis, and policy evaluation.

## Public Release Note

This repository is a portfolio-safe release. It documents the research-computing workflow without publishing raw data, private notes, full result tables, or manuscript-only materials.
