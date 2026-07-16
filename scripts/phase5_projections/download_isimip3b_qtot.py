"""
Phase 5 — ISIMIP 3b runoff (qtot) download

Uses ISIMIP REST API v1 to discover and download monthly qtot NetCDF files.

FILE STRUCTURE (confirmed by API discovery 2026-05-30):
  Each model×GCM×scenario combination is a SINGLE file covering 2015–2100.
  Filename pattern:
    {model}_{gcm}_w5e5_{scenario}_2015soc-from-histsoc_default_qtot_global_monthly_2015_2100.nc
  File size: ~250–270 MB per file.

  IMPORTANT API NOTE:
    The ISIMIP API's `impact_model` filter parameter is unreliable — it returns
    files from unrelated sectors. This script queries by GCM × scenario only
    and then selects the correct model file by filename prefix (post-filter).

  Total for all 27 combinations (3 models × 3 GCMs × 3 scenarios):
    27 files  ~6.77 GB  — fully manageable.

Saves to: data/raw/isimip3b/qtot/
Manifest: data/raw/isimip3b/qtot/download_manifest.json

Usage:
    python download_isimip3b_qtot.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import subprocess

import requests

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT / "data" / "raw" / "isimip3b" / "qtot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUT_DIR / "download_manifest.json"

# ── ISIMIP API constants ───────────────────────────────────────────────────────
API_BASE   = "https://data.isimip.org/api/v1/files/"
FILES_BASE = "https://files.isimip.org/"

# ── Download scope ────────────────────────────────────────────────────────────
# 27 files total (3 models × 3 GCMs × 3 scenarios), ~6.77 GB.
# Reduce MODELS to ["cwatm"] for a minimal ~2.3 GB first run.
MODELS    = ["cwatm", "h08", "watergap2-2e"]
GCMS      = ["gfdl-esm4", "mpi-esm1-2-hr", "ipsl-cm6a-lr"]
SCENARIOS = ["ssp126", "ssp370", "ssp585"]

# soc-variant preference order (first match wins)
SOC_PREFERENCE = ["2015soc-from-histsoc", "2015soc"]


# ── API helpers ────────────────────────────────────────────────────────────────

def query_api(
    gcm: str,
    scenario: str,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """
    Return all water_global qtot monthly file records for one GCM × scenario.
    NOTE: The ISIMIP API's impact_model filter is unreliable — query without it
    and post-filter by filename prefix instead.
    Follows API pagination (next link) until exhausted.
    """
    params: Optional[Dict[str, str]] = {
        "simulation_round": "ISIMIP3b",
        "product":          "OutputData",
        "sector":           "water_global",
        "variable":         "qtot",
        "time_step":        "monthly",
        "period":           "future",
        "climate_forcing":  gcm,
        "climate_scenario": scenario,
    }
    results: List[Dict[str, Any]] = []
    url: Optional[str] = API_BASE

    while url:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data.get("results", []))
        url    = data.get("next")   # follow pagination
        params = None               # params sent only on first request

    return results


def select_model_file(
    records: List[Dict[str, Any]],
    model: str,
    soc_prefs: List[str],
) -> Optional[Dict[str, Any]]:
    """
    From a list of API file records, pick the best file for *model*.
    Selects files whose name starts with `model_` and prefers soc variants
    in order of *soc_prefs*. Returns None if the model is not found.
    """
    model_files = [
        r for r in records
        if r.get("name", "").startswith(model + "_")
    ]
    for soc in soc_prefs:
        candidates = [r for r in model_files if soc in r.get("name", "")]
        if candidates:
            return candidates[0]
    return model_files[0] if model_files else None


def file_url_from_record(record: Dict[str, Any]) -> str:
    """
    Resolve download URL from an API file record.
    Tries explicit URL fields first; falls back to constructing from path.
    """
    for key in ("file_url", "download_url"):
        if record.get(key):
            return record[key]
    path = record.get("path", "")
    return FILES_BASE + path.lstrip("/")





# ── Download helper ────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, retries: int = 6) -> bool:
    """
    Download *url* to *dest* using curl with resume support (-C -).
    curl handles macOS SSL errors more reliably than Python requests.
    Retries with exponential backoff (capped at 60 s).
    Returns True on success, False after exhausting retries.
    """
    for attempt in range(retries):
        try:
            cmd = [
                "curl", "-fL",
                "--connect-timeout", "30",
                "--max-time", "900",
                "-C", "-",           # resume from partial file
                "--progress-bar",
                "-o", str(dest),
                url,
            ]
            subprocess.run(cmd, check=True)
            if dest.exists() and dest.stat().st_size > 1_000_000:
                return True
            raise RuntimeError(f"File too small after download ({dest.stat().st_size} B)")

        except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
            if attempt < retries - 1:
                wait = min(2 ** (attempt + 1), 60)   # 2 → 4 → 8 → 16 → 32 → 60 s
                print(f"  ⚠  attempt {attempt + 1} failed ({exc}); "
                      f"retrying in {wait}s …")
                time.sleep(wait)
            else:
                if dest.exists():
                    dest.unlink()   # remove broken file only on final failure
                print(f"  ✗  FAILED after {retries} attempts: {exc}")

    return False


# ── Discover files via API ─────────────────────────────────────────────────────
print("Querying ISIMIP REST API …")
print(f"  Endpoint : {API_BASE}")
print(f"  Models   : {MODELS}")
print(f"  GCMs     : {GCMS}")
print(f"  Scenarios: {SCENARIOS}")
print(f"  soc pref : {SOC_PREFERENCE}")
print()

file_records: List[Dict[str, Any]] = []

for gcm in GCMS:
    for scenario in SCENARIOS:
        try:
            all_records = query_api(gcm, scenario)
        except requests.HTTPError as exc:
            print(f"  API HTTP {exc.response.status_code} for {gcm}/{scenario} — skipping")
            continue
        except Exception as exc:
            print(f"  API error for {gcm}/{scenario}: {exc} — skipping")
            continue

        for model in MODELS:
            label = f"{model:<14} / {gcm:<18} / {scenario}"
            rec = select_model_file(all_records, model, SOC_PREFERENCE)
            if rec is None:
                print(f"  {label}: NOT FOUND in API response — skipping")
                continue
            rec = dict(rec)  # copy so we can annotate
            rec["_model"]    = model
            rec["_gcm"]      = gcm
            rec["_scenario"] = scenario
            file_records.append(rec)
            sz = rec.get("size", 0)
            print(f"  {label}: {rec.get('name', '')[:62]}  ({sz / 1e6:.0f} MB)")

print()
n_files     = len(file_records)
total_bytes = sum(r.get("size", 0) for r in file_records)
total_gb    = total_bytes / 1e9

print(f"Total files discovered : {n_files}")
print(f"Estimated download size: {total_gb:.2f} GB  ({total_bytes / 1e6:.0f} MB)")

if not file_records:
    sys.exit("No files found — check API parameters or network access.")

# ── Confirm before downloading ─────────────────────────────────────────────────
ans = input("\nProceed with download? [y/N] ").strip().lower()
if ans not in ("y", "yes"):
    sys.exit("Download aborted.")

# ── Download loop ──────────────────────────────────────────────────────────────
print()
manifest: List[Dict[str, Any]] = []

for rec in file_records:
    name       = rec.get("name", "unknown.nc")
    size_bytes = rec.get("size", 0)
    dest       = OUT_DIR / name
    url        = file_url_from_record(rec)

    if dest.exists() and dest.stat().st_size > 0:
        status = "skipped"
        print(f"  ↩  {name}  (already present — skipping)")
    else:
        print(f"\n→ {name}  ({size_bytes / 1e6:.0f} MB)")
        ok     = download_file(url, dest)
        status = "ok" if ok else "failed"

    manifest.append({
        "file":     name,
        "model":    rec.get("_model"),
        "gcm":      rec.get("_gcm"),
        "scenario": rec.get("_scenario"),
        "size_mb":  round(size_bytes / 1e6, 1),
        "status":   status,
        "url":      url,
    })

# ── Save manifest ──────────────────────────────────────────────────────────────
with open(MANIFEST_PATH, "w") as fh:
    json.dump(manifest, fh, indent=2)

# ── Print summary table ────────────────────────────────────────────────────────
SEP = "─" * 80
print(f"\n{SEP}")
print(f"{'FILE':<58} {'SIZE_MB':>8}  STATUS")
print(SEP)
for m in manifest:
    print(f"{m['file'][:58]:<58} {m['size_mb']:>8.1f}  {m['status']}")

ok_count   = sum(1 for m in manifest if m["status"] == "ok")
skip_count = sum(1 for m in manifest if m["status"] == "skipped")
fail_count = sum(1 for m in manifest if m["status"] == "failed")

print(f"\nSummary: {ok_count} downloaded, {skip_count} skipped, "
      f"{fail_count} failed")
print(f"Manifest saved → {MANIFEST_PATH}")
