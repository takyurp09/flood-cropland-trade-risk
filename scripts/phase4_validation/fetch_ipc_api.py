"""
Phase 4 — IPC-CH API Fetch
===========================
Fetches national-level IPC Acute Food Insecurity Phase classifications
via the IPC-CH Public API for years 2017-2021.

API docs  : https://docs.api.ipcinfo.org/
Base URL  : https://api.ipcinfo.org
Auth      : API key passed as ?key=<token> query parameter (no Bearer header)
Endpoint  : GET /country  — country-level population for latest analysis per year
            GET /analyses — used first to confirm available country×year coverage

Output → data/ipc/ipc_api_national.csv

Columns: country_iso3, country_iso2, year, ipc_phase, population_affected

Usage:
    python fetch_ipc_api.py

Requires:
    pip install requests pandas pycountry python-dotenv
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import pycountry
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------
PROJECT  = Path(__file__).resolve().parents[2]
IPC_DIR  = PROJECT / "data" / "ipc"
IPC_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = IPC_DIR / "ipc_api_national.csv"

# Load API key — must be set in .env at project root
load_dotenv(PROJECT / ".env")
API_KEY = os.getenv("IPC_API_TOKEN")
if not API_KEY:
    raise RuntimeError(
        "IPC_API_TOKEN not found. Add 'IPC_API_TOKEN=<key>' to .env "
        f"at: {PROJECT / '.env'}"
    )

BASE_URL    = "https://api.ipcinfo.org"
YEARS       = range(2017, 2022)   # 2017–2021 inclusive
CONDITION   = "A"                  # A = Acute Food Insecurity

MAX_RETRIES  = 4
BACKOFF_BASE = 2.0                 # seconds; delay = BACKOFF_BASE ** attempt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iso2_to_iso3(alpha2):
    """Convert 2-letter ISO-3166-1 alpha-2 code to alpha-3 code."""
    if not alpha2 or not isinstance(alpha2, str):
        return None
    try:
        country = pycountry.countries.get(alpha_2=alpha2.strip().upper())
        return country.alpha_3 if country else None
    except Exception:
        return None


def get_with_retry(url, params):
    """
    HTTP GET with exponential backoff on rate-limit (429) or server errors (5xx).
    Returns parsed JSON (list or dict) on success, raises RuntimeError on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = BACKOFF_BASE ** attempt
                print(
                    f"    [retry {attempt}/{MAX_RETRIES}] HTTP {resp.status_code} — "
                    f"sleeping {wait:.0f}s …"
                )
                time.sleep(wait)
                continue
            # Non-retryable HTTP error
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            wait = BACKOFF_BASE ** attempt
            print(
                f"    [retry {attempt}/{MAX_RETRIES}] RequestException: {exc} — "
                f"sleeping {wait:.0f}s …"
            )
            time.sleep(wait)
    raise RuntimeError(f"All {MAX_RETRIES} retries exhausted for: {url}")


def fetch_country_year(year):
    """
    Call GET /country?year=YYYY&type=A to retrieve all countries with
    IPC Acute analyses published in that year.

    The endpoint returns one record per country (the most recent analysis
    for that year if all periods are now in the past).

    Returns a list of Area dicts.
    """
    params = {
        "year"   : year,
        "type"   : CONDITION,
        "key"    : API_KEY,
        "format" : "json",
    }
    print(f"  Fetching /country?year={year}&type={CONDITION} …", end=" ", flush=True)
    data = get_with_retry(f"{BASE_URL}/country", params)

    if not isinstance(data, list):
        print(f"WARNING: unexpected response type {type(data).__name__} — skipping year")
        return []

    print(f"{len(data)} records returned")
    return data


def derive_national_phase(phases, estimated_pop):
    """
    Derive a single national IPC phase (int 1-5) from the phases population
    distribution returned by the /country endpoint.

    Rule: national phase = the worst phase (highest number) at which the
    cumulative share of population in that phase *or worse* first reaches
    >= 20 % of the analysed population.  This mirrors IPC area classification
    thresholds and is a standard research proxy for national food-security
    severity.

    If no phase reaches the 20 % threshold (tiny analysis), returns the phase
    with the largest absolute population.
    """
    if not phases or not estimated_pop or estimated_pop == 0:
        return None

    phase_pop = {}
    for p in phases:
        try:
            ph  = int(p.get("phase", 0))
            pop = int(p.get("population") or 0)
            phase_pop[ph] = pop
        except (TypeError, ValueError):
            continue

    if not phase_pop:
        return None

    cumulative = 0
    for ph in range(5, 0, -1):
        cumulative += phase_pop.get(ph, 0)
        if cumulative / estimated_pop >= 0.20:
            return ph

    # Fallback: return the phase with the largest population
    return max(phase_pop, key=phase_pop.get)


def parse_record(rec, year):
    """
    Extract a flat row dict from a /country API response record.
    Returns None for records missing a valid phase or country code.

    The /country endpoint does not expose 'overall_phase' at the national
    level; instead it provides a 'phases' array with population counts per
    phase.  We derive the national phase via derive_national_phase().
    """
    iso2           = rec.get("country", "")
    iso3           = iso2_to_iso3(iso2)
    estimated_pop  = rec.get("estimated_population")   # analysed population
    phases         = rec.get("phases", [])

    if not iso3:
        return None

    phase = derive_national_phase(phases, estimated_pop)
    if phase is None:
        return None

    # population_affected = total in phase 3 or worse
    pop_3plus = None
    if phases and estimated_pop:
        try:
            pop_3plus = sum(
                int(p.get("population") or 0)
                for p in phases
                if int(p.get("phase", 0)) >= 3
            )
        except (TypeError, ValueError):
            pop_3plus = None

    return {
        "country_iso3"       : iso3,
        "country_iso2"       : iso2.strip().upper(),
        "year"               : int(year),
        "ipc_phase"          : int(phase),
        "population_affected": pop_3plus,
        "estimated_population": int(estimated_pop) if estimated_pop is not None else None,
    }


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("IPC-CH API — national-level acute food insecurity fetch")
    print(f"Years    : {list(YEARS)}")
    print(f"Condition: Acute Food Insecurity (type=A)")
    print("=" * 60)

    all_rows = []
    for year in YEARS:
        records = fetch_country_year(year)
        parsed  = [parse_record(rec, year) for rec in records]
        valid   = [r for r in parsed if r is not None]
        skipped = len(records) - len(valid)
        if skipped:
            print(f"    (skipped {skipped} records with missing/invalid phase or ISO code)")
        all_rows.extend(valid)
        time.sleep(0.5)   # polite pause between requests

    if not all_rows:
        print("\nWARNING: No data retrieved. Check API key and network connectivity.")
        return

    df = pd.DataFrame(all_rows)

    # Deduplicate: keep latest for any duplicated country×year
    df = (
        df.sort_values(["country_iso3", "year"])
          .drop_duplicates(subset=["country_iso3", "year"], keep="last")
          .reset_index(drop=True)
    )

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows → {OUT_PATH}")

    # -----------------------------------------------------------------------
    # Step 3 — Validation report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    n_countries = df["country_iso3"].nunique()
    years_present = sorted(df["year"].unique().tolist())
    null_phases  = int(df["ipc_phase"].isna().sum())
    oor_phases   = int((~df["ipc_phase"].between(1, 5)).sum())

    print(f"\nUnique country count : {n_countries}  (expect 80+)")
    print(f"Years present        : {years_present}  (expect 2017–2021)")
    print(f"\nIPC phase distribution (should be integers 1–5 only):")
    print(df["ipc_phase"].value_counts().sort_index().rename("count").to_string())
    print(f"\nNull phases          : {null_phases}  (expect 0)")
    print(f"Out-of-range phases  : {oor_phases}  (expect 0)")

    # -----------------------------------------------------------------------
    # Compare with manual download
    # -----------------------------------------------------------------------
    manual_candidates = [
        IPC_DIR / "ipc_global_national_long_latest.csv",
        IPC_DIR / "ipc_global_area_long.csv",
    ]
    manual_path = next((p for p in manual_candidates if p.exists()), None)

    if manual_path is None:
        print("\nManual download not found — skipping comparison.")
    else:
        print(f"\nComparing with manual download: {manual_path.name}")
        manual = pd.read_csv(manual_path, low_memory=False)

        # Identify the country column — manual file may use ISO3 or country names
        country_col = None
        for candidate in ("country_iso3", "iso3", "Country", "country"):
            if candidate in manual.columns:
                country_col = candidate
                break

        if country_col is None:
            print("  Could not identify country column in manual file — skipping comparison.")
        else:
            manual_countries = set(manual[country_col].dropna().str.strip().unique())
            api_countries    = set(df["country_iso3"].unique())

            # If manual file uses names rather than ISO codes, lengths will differ —
            # flag this but don't crash.
            only_manual = sorted(manual_countries - api_countries)
            only_api    = sorted(api_countries - manual_countries)

            print(f"  Manual rows   : {len(manual):,}")
            print(f"  API rows      : {len(df):,}")
            print(f"  Manual unique {country_col}: {len(manual_countries)}")
            print(f"  API unique ISO3           : {len(api_countries)}")

            if only_manual:
                print(f"\n  In manual NOT in API ({len(only_manual)}):")
                print("  " + ", ".join(only_manual[:40]))
                if len(only_manual) > 40:
                    print(f"  … and {len(only_manual)-40} more")
            else:
                print("\n  No countries in manual only.")

            if only_api:
                print(f"\n  In API NOT in manual ({len(only_api)}):")
                print("  " + ", ".join(only_api[:40]))
                if len(only_api) > 40:
                    print(f"  … and {len(only_api)-40} more")
            else:
                print("\n  No countries in API only.")

    print("\nDone.")


if __name__ == "__main__":
    main()
