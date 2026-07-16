# Reproducibility

This repository is organized as a staged research pipeline. The public version is a coding-portfolio release and does not include raw data or final manuscript artifacts.

## Environment

Recommended Python version: 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some scripts also require provider-specific authentication, especially Google Earth Engine and APIs used for validation data.

## Suggested Run Order

1. `scripts/phase1_flood/`  
   Build flood cropland exposure from satellite products, crop calendars, harvested-area weights, and yield-anomaly inputs.

2. `scripts/phase2_trade/`  
   Prepare BACI trade data and construct flood trade exposure.

3. `scripts/phase3_outcome/`  
   Fetch or prepare food-balance, climate, policy, and food-security variables.

4. `scripts/phase3_panel/`  
   Build the regression panel and run empirical specifications.

5. `scripts/phase4_validation/` and `scripts/phase4_ipc/`  
   Run validation checks against external production, vegetation, and food-security indicators.

6. `scripts/phase5_projection/` and `scripts/phase5_projections/`  
   Produce forward-looking exposure and caloric-risk projection inputs.

## Version-Control Practice

The repository is intended to demonstrate good version-control hygiene:

- source code, documentation, and selected visuals are tracked;
- raw data, credentials, logs, cache files, and large outputs are ignored;
- the public release is organized into interpretable commits;
- local paths and private tokens are excluded from committed files.

## Public Example Outputs

The `examples/` directory contains small illustrative CSV schemas for the major data products. These files are intentionally synthetic and are provided so reviewers can see column names, units, and data-flow expectations without exposing unpublished empirical results.

The `figures/selected/` directory contains a curated set of public visuals. See `docs/VISUAL_GALLERY.md` for figure notes.

## Expected Non-Reproducible Elements

Some outputs cannot be regenerated without external data access, data-provider credentials, or large local downloads. This is documented intentionally rather than hidden.
