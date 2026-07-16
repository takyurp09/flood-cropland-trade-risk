#!/usr/bin/env python3
"""Figure S1 — Identification sensitivity forest plot (supplementary)."""

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

os.makedirs(FIG_DIR, exist_ok=True)

# ── Color system (identical to Figs 2 and 3) ─────────────────────────────────
NAVY = "#1B3A5C"
STEEL = "#4A7BA7"
SENTINEL = "#5B8C5A"
SSA_ORA = "#D4622A"
CAVEAT = "#9E9E9E"
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
    label_s, beta, lo, hi, color, pval, is_caveat = row
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
            f"Override {label.split(chr(10))[0]}: "
            f"β={csv_beta:.3f}, CI=[{csv_lo:.3f}, {csv_hi:.3f}], p={csv_p:.3f}"
        )
    return label_s, csv_beta, csv_lo, csv_hi, color, csv_p, is_caveat


def main():
    round5_path = os.path.join(RESULTS_DIR, "round5", "round5_consolidated.csv")
    mechanism_path = os.path.join(RESULTS_DIR, "mechanism", "mechanism_consolidated.csv")
    equity_path = os.path.join(RESULTS_DIR, "round8", "lic_ssa_equity.csv")

    try:
        pd.read_csv(round5_path)
        print("Loaded round5 results for verification")
    except FileNotFoundError:
        print("CSV not found — using hardcoded values")

    rows = [
        ("IV — full sample\n(2000–2021)", 0.047, 0.029, 0.065, NAVY, 0.001, False),
        ("Bartik shift-share", 0.037, 0.003, 0.071, STEEL, 0.021, False),
        ("Bartik — post-2015 subsample\n(JRC GSW, 2015–2021)", 0.137, 0.088, 0.186, SENTINEL, 0.001, False),
        ("IV + HHI control", 0.060, 0.030, 0.090, NAVY, 0.001, False),
        ("IV — Sub-Saharan Africa\n(N=38 countries)", 0.229, 0.140, 0.318, SSA_ORA, 0.001, False),
        ("Fixed top-3 identity\n(caveat: β<0)", -0.024, -0.040, -0.008, CAVEAT, 0.001, True),
    ]
    specs = [
        (mechanism_path, "S_IV_CONCENTRATION"),
        (mechanism_path, "S_BARTIK_CONCENTRATION"),
        (round5_path, "S_SENTINEL_BARTIK_CONC"),
        (round5_path, "S_HHI_CONTROL_CONC"),
        (equity_path, "S_SSA_IV_CONC"),
        (round5_path, "S_IV_CONC_FIXED"),
    ]
    rows = [
        apply_csv_override(row[0], row, path, spec)
        for row, (path, spec) in zip(rows, specs)
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = list(range(len(rows) - 1, -1, -1))

    for i, (label, beta, lo, hi, color, pval, is_caveat) in enumerate(rows):
        y = y_pos[i]
        ptxt = stars(pval)
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.5, zorder=3, solid_capstyle="round")
        if is_caveat:
            ax.plot(
                beta, y, "o", color="white", markeredgecolor=CAVEAT,
                markeredgewidth=1.5, markersize=7, zorder=4,
            )
        else:
            ax.plot(beta, y, "o", color=color, markersize=7, zorder=4)

        if is_caveat:
            ax.text(
                0.01, y,
                f"\u2212{abs(beta):.3f}, {ptxt}",
                va="center", ha="left",
                fontsize=7.0,
                fontweight="normal",
                color=CAVEAT,
                style="italic",
            )
            ax.text(
                0.01, y - 0.30,
                "(sign flip: see Methods)",
                va="top", ha="left",
                fontsize=6.5,
                color=CAVEAT,
                style="italic",
            )
        elif hi > 0.25:
            ax.text(
                beta, y + 0.32, f"{beta:.3f}, {ptxt}",
                va="bottom", ha="center", fontsize=7.5, fontweight="bold", color=color,
            )
        else:
            ax.text(
                hi + 0.008, y, f"{beta:.3f}, {ptxt}",
                va="center", ha="left", fontsize=7.5, fontweight="bold", color=color,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5, color="#1a1a1a")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(
        "β on top-3 flood-exposed import share (95% CI)",
        fontsize=8.5, color="#1a1a1a",
    )
    ax.tick_params(axis="x", labelsize=7.5)
    ax.axvline(0, color=ZERO, linewidth=0.9, linestyle="--", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.axhline(0.5, color="#DDDDDD", linewidth=0.8, linestyle="-", zorder=0)

    ax.set_title(
        "Sensitivity of flood-exposed import share association\n"
        "across exposure designs",
        fontsize=9.5, fontweight="bold", pad=10, color="#1a1a1a", loc="center",
    )

    fig.text(
        0.12, 0.01,
        "Headline IV specification uses levels (2000–2021). "
        "FTE persistence (ρ=0.77) contaminates\n"
        "distributed-lag pre-trends on exposure; "
        "event-study timing not used for primary inference.\n"
        "Sign flip for fixed top-3 identity reflects different "
        "supplier sets (time-varying vs pre-period); see Methods.",
        fontsize=6.5, color=NOTE, style="italic", va="bottom", ha="left",
    )

    plt.subplots_adjust(top=0.88, bottom=0.22, left=0.28, right=0.95)

    print("X limits:", ax.get_xlim())
    print("Y limits:", ax.get_ylim())
    print("N rows drawn:", len(rows))
    assert rows[-1][4] == CAVEAT, "Last row must be CAVEAT color"
    assert rows[-1][6] is True, "Last row must be is_caveat=True"
    for label, beta, lo, hi, color, pval, is_caveat in rows:
        if not is_caveat:
            assert hi < 0.38, f"CI upper bound {hi} exceeds xlim for {label}"
    print("All checks passed.")

    ax.set_xlim(-0.08, 0.40)
    ax.set_xbound(lower=-0.08, upper=0.40)

    out_png = os.path.join(FIG_DIR, "figS1_identification.png")
    out_pdf = os.path.join(FIG_DIR, "figS1_identification.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Fig S1 patch 3 saved to outputs/figures/")


if __name__ == "__main__":
    main()
