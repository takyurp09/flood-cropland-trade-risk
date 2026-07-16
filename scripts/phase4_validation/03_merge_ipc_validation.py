"""
Phase 4 — IPC Validation Panel Merge
=====================================
Merges IPC API national-level data (data/ipc/ipc_api_national.csv) with
the main regression panel on country_iso3 × year, restricted to 2017-2021.

Also joins log_staple_import_kt from BACI trade panel and runs an ordered
logit regression: ipc_phase ~ fte_total + log_gdp_pc + log_staple_import_kt + year_FE

Prefers the API-fetched version over any manual download.

Input  : data/ipc/ipc_api_national.csv  (from fetch_ipc_api.py)
         outputs/regression/master_panel_v2.csv  (main regression panel)
         outputs/trade/baci_trade_panel.csv
Output : outputs/validation/ipc_validation_panel.csv
         outputs/validation/ipc_ordered_logit_results.csv

Usage:
    python 03_merge_ipc_validation.py

Requires:
    pip install pandas statsmodels
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT   = Path(__file__).resolve().parents[2]
IPC_DIR   = PROJECT / "data" / "ipc"
OUT_DIR   = PROJECT / "outputs" / "validation"
REG_DIR   = PROJECT / "outputs" / "regression"
TRADE_DIR = PROJECT / "outputs" / "trade"

OUT_DIR.mkdir(parents=True, exist_ok=True)

IPC_API_PATH  = IPC_DIR / "ipc_api_national.csv"
PANEL_PATH    = REG_DIR / "master_panel_v2.csv"
BACI_PATH     = TRADE_DIR / "baci_trade_panel.csv"
OUT_PATH      = OUT_DIR / "ipc_validation_panel.csv"
LOGIT_PATH    = OUT_DIR / "ipc_ordered_logit_results.csv"

IPC_YEARS = [2017, 2018, 2019, 2020, 2021]
# Staple crops for import volume (caloric staples used in FTE)
STAPLE_CROPS = ["rice", "wheat", "maize", "soybeans"]

# ---------------------------------------------------------------------------
# Load IPC data — prefer API version, fall back to manual if API missing
# ---------------------------------------------------------------------------
def load_ipc():
    if IPC_API_PATH.exists():
        print(f"Loading IPC data (API version): {IPC_API_PATH.name}")
        df = pd.read_csv(IPC_API_PATH)
        # Ensure canonical column names
        if "country_iso3" not in df.columns:
            raise ValueError(f"Expected 'country_iso3' column in {IPC_API_PATH}")
        return df[["country_iso3", "year", "ipc_phase", "population_affected"]]

    # Fallback: manual download (area-level, needs aggregation to national)
    manual_candidates = [
        IPC_DIR / "ipc_global_national_long_latest.csv",
        IPC_DIR / "ipc_global_area_long.csv",
    ]
    manual_path = next((p for p in manual_candidates if p.exists()), None)
    if manual_path is None:
        raise FileNotFoundError(
            f"No IPC data found. Run fetch_ipc_api.py first, or place a manual "
            f"download at {IPC_API_PATH}"
        )

    print(f"WARNING: API file not found. Falling back to manual: {manual_path.name}")
    manual = pd.read_csv(manual_path, low_memory=False)

    # Identify country and phase columns
    country_col = next(
        (c for c in ("country_iso3", "iso3", "Country", "country") if c in manual.columns),
        None,
    )
    phase_col = next(
        (c for c in ("ipc_phase", "overall_phase", "Phase", "phase") if c in manual.columns),
        None,
    )
    year_col = next(
        (c for c in ("year", "Year") if c in manual.columns),
        None,
    )

    if not all([country_col, phase_col, year_col]):
        raise ValueError(
            f"Could not auto-detect required columns in {manual_path.name}. "
            f"Columns found: {manual.columns.tolist()}"
        )

    df = manual[[country_col, year_col, phase_col]].copy()
    df.columns = ["country_iso3", "year", "ipc_phase"]
    df = df.dropna(subset=["country_iso3", "year", "ipc_phase"])
    df["ipc_phase"] = pd.to_numeric(df["ipc_phase"], errors="coerce")
    df = df.dropna(subset=["ipc_phase"])
    df["ipc_phase"] = df["ipc_phase"].astype(int)
    df["population_affected"] = None

    # Keep only valid phases 1–5
    df = df[df["ipc_phase"].between(1, 5)]
    # National-level: one phase per country×year (take max phase = worst case)
    df = df.groupby(["country_iso3", "year"], as_index=False)["ipc_phase"].max()
    df["population_affected"] = None

    return df[["country_iso3", "year", "ipc_phase", "population_affected"]]


# ---------------------------------------------------------------------------
# Load main regression panel
# ---------------------------------------------------------------------------
def load_panel():
    if not PANEL_PATH.exists():
        # Try alternate name
        alt = REG_DIR / "master_panel.csv"
        if alt.exists():
            print(f"master_panel_v2.csv not found — using {alt.name}")
            return pd.read_csv(alt, low_memory=False)
        raise FileNotFoundError(
            f"Regression panel not found at {PANEL_PATH} or {alt}"
        )
    return pd.read_csv(PANEL_PATH, low_memory=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("IPC Validation Panel Merge")
    print("=" * 60)

    # ---- Load IPC data ----
    ipc = load_ipc()
    ipc["year"] = ipc["year"].astype(int)

    print(f"  IPC rows loaded   : {len(ipc):,}")
    print(f"  IPC years present : {sorted(ipc['year'].unique().tolist())}")
    print(f"  IPC countries     : {ipc['country_iso3'].nunique()}")

    # Filter to 2017-2021 only
    ipc = ipc[ipc["year"].isin(IPC_YEARS)].copy()
    print(f"  IPC rows (2017–2021): {len(ipc):,}")

    # ---- Load main panel ----
    panel = load_panel()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel = panel.rename(columns={"iso3": "country_iso3"})   # harmonise key name

    panel_ipc_window = panel[panel["year"].isin(IPC_YEARS)].copy()
    print(f"\n  Panel rows (2017–2021) : {len(panel_ipc_window):,}")
    print(f"  Panel countries         : {panel_ipc_window['country_iso3'].nunique()}")

    # ---- Merge ----
    merged = panel_ipc_window.merge(
        ipc,
        on=["country_iso3", "year"],
        how="left",
        validate="1:1",
    )

    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    print(f"  Output rows : {len(merged):,}")

    # -----------------------------------------------------------------------
    # Coverage report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MERGE COVERAGE REPORT")
    print("=" * 60)

    n_total    = len(merged)
    n_matched  = merged["ipc_phase"].notna().sum()
    merge_rate = n_matched / n_total * 100 if n_total > 0 else 0.0

    print(f"\nMerge rate          : {n_matched:,} / {n_total:,}  ({merge_rate:.1f}%)")

    covered_countries = merged.loc[merged["ipc_phase"].notna(), "country_iso3"].nunique()
    total_countries   = merged["country_iso3"].nunique()
    print(f"Country coverage    : {covered_countries} / {total_countries} countries have IPC data")

    covered_years = sorted(
        merged.loc[merged["ipc_phase"].notna(), "year"].dropna().unique().tolist()
    )
    print(f"Year coverage       : {covered_years}")

    # Countries in panel but not matched
    unmatched = (
        merged[merged["ipc_phase"].isna()]["country_iso3"]
        .dropna().unique().tolist()
    )
    if unmatched:
        print(f"\nPanel countries with NO IPC data ({len(unmatched)}):")
        print("  " + ", ".join(sorted(unmatched)[:50]))
        if len(unmatched) > 50:
            print(f"  … and {len(unmatched) - 50} more")
    else:
        print("\nAll panel countries matched to IPC data.")

    # IPC countries not in panel
    ipc_only = set(ipc["country_iso3"]) - set(panel_ipc_window["country_iso3"].dropna())
    if ipc_only:
        print(f"\nIPC countries NOT in regression panel ({len(ipc_only)}):")
        print("  " + ", ".join(sorted(ipc_only)[:50]))
    else:
        print("\nAll IPC countries are in the regression panel.")

    # -----------------------------------------------------------------------
    # Add log_staple_import_kt from BACI trade panel (if not already in merged)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ADDING log_staple_import_kt FROM BACI")
    print("=" * 60)

    if "log_staple_import_kt" in merged.columns:
        # master_panel_v2 already has this column precomputed — use it directly
        n_import_filled = merged["log_staple_import_kt"].notna().sum()
        print(f"  log_staple_import_kt already present in master panel")
        print(f"  Rows with log_staple_import_kt: {n_import_filled:,} / {len(merged):,}")
        if n_import_filled == 0:
            print(f"  → All null for 2017–2021 (FAO FBS import_qty_kt ends 2013); filling from BACI …")
            # Drop null column and recompute from BACI
            merged = merged.drop(columns=["log_staple_import_kt",
                                           *(["staple_import_kt"] if "staple_import_kt" in merged.columns else [])])
            n_import_filled = -1  # trigger BACI join below

    if "log_staple_import_kt" not in merged.columns:
        baci = pd.read_csv(BACI_PATH, low_memory=False)
        baci["year"] = baci["year"].astype(int)
        baci_staples = baci[
            baci["commodity"].isin(STAPLE_CROPS) & baci["year"].isin(IPC_YEARS)
        ].copy()
        # Aggregate: total imports per importer × year
        import_agg = (
            baci_staples.groupby(["importer", "year"], as_index=False)["quantity_t"]
            .sum()
            .rename(columns={"importer": "country_iso3", "quantity_t": "staple_import_t"})
        )
        import_agg["staple_import_kt"] = import_agg["staple_import_t"] / 1_000
        import_agg["log_staple_import_kt"] = np.log1p(import_agg["staple_import_kt"])
        import_agg = import_agg[["country_iso3", "year", "log_staple_import_kt"]]

        merged["year"] = merged["year"].astype(int)
        import_agg["year"] = import_agg["year"].astype(int)
        merged = merged.merge(import_agg, on=["country_iso3", "year"], how="left")
        n_import_filled = merged["log_staple_import_kt"].notna().sum()
        print(f"  BACI importer-years joined: {len(import_agg):,}")
        print(f"  BACI importers covered: {import_agg['country_iso3'].nunique()}")

    n_import_filled = merged["log_staple_import_kt"].notna().sum()
    print(f"  Rows with log_staple_import_kt: {n_import_filled:,} / {len(merged):,}")
    if n_import_filled > 0:
        print(f"  Range: {merged['log_staple_import_kt'].min():.2f} – {merged['log_staple_import_kt'].max():.2f}")
        print(f"  Rows with log_staple_import_kt: {n_import_filled:,} / {len(merged):,}")

    # Re-save with new/confirmed column
    merged.to_csv(OUT_PATH, index=False)
    print(f"\nUpdated → {OUT_PATH}  ({len(merged):,}r, log_staple_import_kt non-null: {n_import_filled:,})")

    # -----------------------------------------------------------------------
    # Ordered Logit: ipc_phase ~ fte_total + log_gdp_pc + log_staple_import_kt + year_FE
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ORDERED LOGIT VALIDATION")
    print("=" * 60)

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        reg_df = merged[
            merged["ipc_phase"].notna() &
            merged["fte_total"].notna() &
            merged["log_gdp_pc"].notna() &
            merged["log_staple_import_kt"].notna()
        ].copy()

        # Ensure all columns are plain float/int (no nullable Int64)
        for col in ["fte_total", "log_gdp_pc", "log_staple_import_kt"]:
            reg_df[col] = pd.to_numeric(reg_df[col], errors="coerce").astype(float)
        reg_df["year"] = pd.to_numeric(reg_df["year"], errors="coerce").astype(int)

        # Year fixed effects as dummies
        reg_df["ipc_phase_ord"] = reg_df["ipc_phase"].astype(int)
        year_dummies = pd.get_dummies(reg_df["year"].astype(str), prefix="yr", drop_first=True).astype(float)
        reg_df = pd.concat([reg_df.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1)

        yr_cols = [c for c in reg_df.columns if c.startswith("yr_")]
        exog_cols = ["fte_total", "log_gdp_pc", "log_staple_import_kt"] + yr_cols
        exog = reg_df[exog_cols].astype(float)
        endog = reg_df["ipc_phase_ord"].astype(int)

        print(f"  Regression sample: {len(reg_df):,} obs, {reg_df['country_iso3'].nunique()} countries")
        print(f"  IPC phase distribution: {dict(endog.value_counts().sort_index())}")
        print(f"  Fitting OrderedModel (logit) …")

        model = OrderedModel(endog, exog, distr="logit")
        result = model.fit(method="bfgs", disp=False)

        # Build results table
        params = result.params
        pvalues = result.pvalues
        conf = result.conf_int()

        rows = []
        for var in exog_cols:
            if var in params.index:
                rows.append({
                    "variable": var,
                    "coef": params[var],
                    "pvalue": pvalues[var],
                    "ci_lo": conf.loc[var, 0],
                    "ci_hi": conf.loc[var, 1],
                })
        results_df = pd.DataFrame(rows)
        results_df["n_obs"] = len(reg_df)
        results_df["n_countries"] = reg_df["country_iso3"].nunique()
        results_df["llf"] = result.llf
        results_df.to_csv(LOGIT_PATH, index=False)

        print(f"\n  Ordered logit results:")
        print(results_df[["variable", "coef", "pvalue"]].to_string(index=False))
        print(f"\n  ✅ Saved → {LOGIT_PATH}  ({len(results_df):,} rows)")

    except ImportError:
        print("  statsmodels not available — skipping ordered logit")
        print("  Install with: pip install statsmodels")
    except Exception as exc:
        print(f"  ⚠️  Ordered logit failed: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
