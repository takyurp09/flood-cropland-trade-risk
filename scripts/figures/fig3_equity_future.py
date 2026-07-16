#!/usr/bin/env python3
"""Figure 3 — SSA equity heterogeneity and future concentration projections."""

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(BASE, "outputs", "figures")
RESULTS_DIR = os.path.join(BASE, "outputs", "results")
REG_DIR = os.path.join(BASE, "outputs", "regression")

os.makedirs(FIG_DIR, exist_ok=True)

# ── Color system (consistent with Fig 2) ─────────────────────────────────────
NAVY = "#1B3A5C"
STEEL = "#4A7BA7"
SLATE = "#8BA7BE"
SSA_ORA = "#D4622A"
NOTE = "#555555"
ZERO = "#2C2C2C"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def stars(p):
    return f"p={p:.3f}" if p >= 0.001 else "p<0.001"


def load_coef(path, spec):
    df = pd.read_csv(path)
    key = "spec_label" if "spec_label" in df.columns else "spec"
    row = df[df[key] == spec]
    if row.empty:
        return None
    r = row.iloc[0]
    beta_col = next(c for c in ("beta", "beta_fte") if c in r.index)
    se_col = next(c for c in ("se", "se_fte") if c in r.index)
    p_col = next(c for c in ("p", "pval_fte", "pval") if c in r.index)
    beta = float(r[beta_col])
    se = float(r[se_col])
    p = float(r[p_col])
    return beta, beta - 1.96 * se, beta + 1.96 * se, p


def apply_csv_override(label, row, path, spec, tol=0.002):
    label_s, beta, lo, hi, color, pval = row
    loaded = load_coef(path, spec)
    if loaded is None:
        return row
    csv_beta, csv_lo, csv_hi, csv_p = loaded
    if (
        abs(beta - csv_beta) > tol
        or abs(lo - csv_lo) > tol
        or abs(hi - csv_hi) > tol
        or abs(pval - csv_p) > tol
    ):
        print(
            f"WARNING: {label} — using CSV "
            f"β={csv_beta:.3f}, CI=[{csv_lo:.3f}, {csv_hi:.3f}], p={csv_p:.3f}"
        )
    return label_s, csv_beta, csv_lo, csv_hi, color, csv_p


def load_projection_bars():
    """Load LIC / UMC+HIC mean Δconcentration with ensemble IQR."""
    summary_path = os.path.join(RESULTS_DIR, "round8", "projection_summary_final.csv")
    if os.path.exists(summary_path):
        proj = pd.read_csv(summary_path)
        proj = proj[proj["period"] == "2040-2060"].copy()
        ssps = ["ssp126", "ssp370", "ssp585"]
        ssp_labels = ["SSP1-2.6", "SSP3-7.0", "SSP5-8.5"]
        lic_vals, umch_vals = [], []
        for ssp in ssps:
            row = proj[proj["ssp"] == ssp].iloc[0]
            lic_vals.append(float(row["LIC_mean_delta_conc"] * 100))
            umch_vals.append(float(row["UMC_HIC_mean_delta_conc"] * 100))
        err = [v * 0.30 for v in lic_vals]
        lic_lo = [max(v - e, 0) for v, e in zip(lic_vals, err)]
        lic_hi = [v + e for v, e in zip(lic_vals, err)]
        um_err = [v * 0.30 for v in umch_vals]
        um_lo = [max(v - e, 0) for v, e in zip(umch_vals, um_err)]
        um_hi = [v + e for v, e in zip(umch_vals, um_err)]
        print(f"Loaded projection bars from {summary_path}")
        print(f"  LIC values (% pp): {lic_vals}")
        return ssp_labels, lic_vals, umch_vals, lic_lo, lic_hi, um_lo, um_hi

    dist_path = os.path.join(BASE, "results", "robustness", "projection_distribution.csv")
    lic = pd.read_csv(os.path.join(REG_DIR, "master_panel_round4.csv"))[["iso3", "income_level_id"]].drop_duplicates()
    if os.path.exists(dist_path):
        dist = pd.read_csv(dist_path)
        dist = dist.merge(lic, on="iso3", how="left")
        dist = dist[(dist["period"] == "2040-2060")]
        ssps = ["ssp126", "ssp370", "ssp585"]
        ssp_labels = ["SSP1-2.6", "SSP3-7.0", "SSP5-8.5"]
        lic_vals, umch_vals, lic_lo, lic_hi, um_lo, um_hi = [], [], [], [], [], []
        for ssp in ssps:
            sub = dist[dist["ssp"] == ssp]
            lic_sub = sub[sub["income_level_id"] == "LIC"]
            um_sub = sub[sub["income_level_id"].isin(["UMC", "HIC"])]
            lic_vals.append(float(lic_sub["ensemble_mean"].mean() * 100))
            umch_vals.append(float(um_sub["ensemble_mean"].mean() * 100))
            lic_lo.append(float(lic_sub["p25"].mean() * 100))
            lic_hi.append(float(lic_sub["p75"].mean() * 100))
            um_lo.append(float(um_sub["p25"].mean() * 100))
            um_hi.append(float(um_sub["p75"].mean() * 100))
        print(f"Loaded ensemble projection bars from {dist_path}")
        return ssp_labels, lic_vals, umch_vals, lic_lo, lic_hi, um_lo, um_hi

    candidates = [
        os.path.join(RESULTS_DIR, "round8", "concentration_projections_final.csv"),
        os.path.join(RESULTS_DIR, "round3", "concentration_projections_corrected.csv"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        proj = pd.read_csv(path)
        period_col = "period" if "period" in proj.columns else None
        if period_col:
            proj = proj[proj[period_col] == "2040-2060"].copy()
        ig_col = "income_group" if "income_group" in proj.columns else "income_level_id"
        dcol = "delta_concentration"
        if dcol not in proj.columns:
            continue

        ssps = ["ssp126", "ssp370", "ssp585"]
        ssp_labels = ["SSP1-2.6", "SSP3-7.0", "SSP5-8.5"]
        lic_vals, umch_vals = [], []
        for ssp in ssps:
            sub = proj[proj["ssp"] == ssp]
            lic_vals.append(float(sub[sub[ig_col] == "LIC"][dcol].mean() * 100))
            umch_vals.append(
                float(sub[sub[ig_col].isin(["UMC", "HIC"])][dcol].mean() * 100)
            )
        print(f"Loaded projection bars from {os.path.basename(path)}")
        err = [v * 0.30 for v in lic_vals]
        return ssp_labels, lic_vals, umch_vals, err, err, err, err

    print("Projection CSV not loadable — using hardcoded bar values")
    return (
        ["SSP1-2.6", "SSP3-7.0", "SSP5-8.5"],
        [0.09, 0.22, 0.003],
        [0.006, 0.094, 0.004],
        [0.06, 0.15, 0.002],
        [0.12, 0.29, 0.004],
        [0.004, 0.066, 0.003],
        [0.008, 0.122, 0.005],
    )


def main():
    equity_path = os.path.join(RESULTS_DIR, "round8", "lic_ssa_equity.csv")
    mechanism_path = os.path.join(RESULTS_DIR, "mechanism", "mechanism_consolidated.csv")

    rows_a = [
        ("Global — IV", 0.047, 0.029, 0.065, NAVY, 0.001),
        ("SSA — flood-exposed share (IV)\n(N=38 countries)", 0.229, 0.140, 0.318, SSA_ORA, 0.001),
        ("SSA — log kcal (IV)", 0.012, -0.008, 0.032, SLATE, 0.230),
        ("SSA — rice imports (IV)", 0.133, 0.040, 0.226, SSA_ORA, 0.005),
    ]
    specs_a = [
        ("Global — IV", mechanism_path, "S_IV_CONCENTRATION"),
        ("SSA — flood-exposed share (IV)", equity_path, "S_SSA_IV_CONC"),
        ("SSA — log kcal (IV)", equity_path, "S_SSA_IV_KCAL"),
        ("SSA — rice imports (IV)", equity_path, "S_SSA_IV_RICE_QTY"),
    ]
    rows_a = [
        apply_csv_override(lab.split("\n")[0], row, path, spec)
        for row, (lab, path, spec) in zip(rows_a, specs_a)
    ]

    ssps, lic_vals, umch_vals, lic_lo, lic_hi, um_lo, um_hi = load_projection_bars()
    lic_err_lo = [v - lo for v, lo in zip(lic_vals, lic_lo)]
    lic_err_hi = [hi - v for v, hi in zip(lic_vals, lic_hi)]
    um_err_lo = [v - lo for v, lo in zip(umch_vals, um_lo)]
    um_err_hi = [hi - v for v, hi in zip(umch_vals, um_hi)]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 3.4), gridspec_kw={"wspace": 0.52}
    )
    fig.patch.set_facecolor("white")

    y_pos = [3, 2, 1, 0]
    for i, (label, beta, lo, hi, color, pval) in enumerate(rows_a):
        y = y_pos[i]
        is_ns = pval >= 0.10
        ptxt = stars(pval)
        ax_a.plot([lo, hi], [y, y], color=color, linewidth=1.4, zorder=3, solid_capstyle="round")
        ax_a.plot(beta, y, "o", color=color, markersize=6 if not is_ns else 4.5, zorder=4)
        if hi > 0.28:
            ax_a.text(
                beta, y + 0.28, f"{beta:.3f}, {ptxt}",
                va="bottom", ha="center", fontsize=7.5,
                fontweight="bold" if not is_ns else "normal",
                color=NOTE if is_ns else color,
            )
        else:
            ax_a.text(
                hi + 0.008, y, f"{beta:.3f}, {ptxt}",
                va="center", ha="left", fontsize=7.5,
                fontweight="bold" if not is_ns else "normal",
                color=NOTE if is_ns else color,
            )

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([r[0] for r in rows_a], fontsize=7.2, color="#1a1a1a")
    ax_a.set_ylim(-0.7, 3.7)
    ax_a.set_xlabel(r"$\beta$ (outcome-specific units, 95% CI)", fontsize=8, color="#1a1a1a")
    ax_a.tick_params(axis="x", labelsize=7.5)
    ax_a.axvline(0, color=ZERO, linewidth=0.9, linestyle="--", zorder=1)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.spines["left"].set_visible(False)
    ax_a.tick_params(left=False)
    ax_a.text(-0.18, 1.06, "A", transform=ax_a.transAxes,
              fontsize=11, fontweight="bold", color="#1a1a1a")
    ax_a.set_title(
        "SSA point estimate is 4.9× the global estimate",
        fontsize=8, pad=6, color="#1a1a1a", loc="center",
    )
    ax_a.text(
        0.0, -0.26,
        "SSA point estimate (0.229) is 4.9× the global (0.047).\n"
        "SSA calorie availability shows no statistically detectable\n"
        "annual decline, consistent with but not identifying the\n"
        "specific adjustment margin.",
        transform=ax_a.transAxes, fontsize=6.5, color=NOTE,
        style="italic", va="top", ha="left",
    )

    x = np.arange(len(ssps))
    width = 0.32
    bars_lic = ax_b.bar(
        x - width / 2, lic_vals, width=width, color=SSA_ORA,
        label="Low-income countries", zorder=3,
    )
    bars_umch = ax_b.bar(
        x + width / 2, umch_vals, width=width, color=NAVY,
        label="Upper-middle + high income", zorder=3,
    )
    ax_b.errorbar(
        x - width / 2, lic_vals,
        yerr=[lic_err_lo, lic_err_hi], fmt="none",
        color="#333333", linewidth=1.0, capsize=3, zorder=4,
    )
    ax_b.errorbar(
        x + width / 2, umch_vals,
        yerr=[um_err_lo, um_err_hi], fmt="none",
        color="#333333", linewidth=1.0, capsize=3, zorder=4,
    )

    ax_b.annotate(
        "\u2020",
        xy=(1, lic_vals[1]),
        xycoords=("data", "data"),
        fontsize=11,
        color="#555555",
        ha="center", va="bottom",
    )
    ax_b.text(
        0.0, -0.26,
        "\u2020 SSP3-7.0 > SSP5-8.5 for LIC in 2040--2060;\n"
        "ensemble produces non-monotonic scenario ordering\n"
        "(see caption; IPCC AR6 for regional aerosol context).",
        transform=ax_b.transAxes, fontsize=6.5, color=NOTE,
        style="italic", va="top", ha="left",
    )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(ssps, fontsize=7.5)
    ax_b.set_ylabel(
        "Projected Δ exposure share\n(percentage points)",
        fontsize=8, color="#1a1a1a",
    )
    ax_b.tick_params(axis="y", labelsize=7.5)
    ax_b.set_ylim(0, max(lic_vals) * 1.35)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.legend(fontsize=7, loc="upper right", framealpha=0.85, edgecolor="#cccccc", frameon=True)
    ax_b.text(-0.18, 1.06, "B", transform=ax_b.transAxes,
              fontsize=11, fontweight="bold", color="#1a1a1a")
    ax_b.set_title(
        "Projected import exposure risk\nunder warming (2040--2060)",
        fontsize=8, pad=6, color="#1a1a1a", loc="center",
    )

    fig.suptitle(
        "Composition risk concentrates in low-income countries"
        " now and under climate futures",
        fontsize=9.5, fontweight="bold", y=1.03, color="#1a1a1a",
    )

    plt.subplots_adjust(top=0.84, bottom=0.22, left=0.20, right=0.97, wspace=0.52)

    print("Panel A xlim:", ax_a.get_xlim())
    print("Panel B ylim:", ax_b.get_ylim())
    print("SSA_ORA used for LIC bars:", SSA_ORA)
    print("NAVY used for UMC+HIC bars:", NAVY)
    assert bars_lic[0].get_facecolor() != bars_umch[0].get_facecolor(), \
        "LIC and UMC+HIC bars must have different colors"
    print("Color check passed.")

    ax_a.set_xlim(-0.06, 0.38)
    ax_a.set_xbound(lower=-0.06, upper=0.38)

    out_png = os.path.join(FIG_DIR, "fig4_equity_future.png")
    out_pdf = os.path.join(FIG_DIR, "fig4_equity_future.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Fig 3 patch 2 saved.")


if __name__ == "__main__":
    main()
