#!/usr/bin/env python3
"""Figure 2 — Composition channel forest plot (IV mechanism + identification)."""

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

# ── Color system ──────────────────────────────────────────────────────────────
NAVY = "#1B3A5C"
STEEL = "#4A7BA7"
SLATE = "#8BA7BE"
SLATE_SIG = "#6B8FA8"
DARK = "#0F2336"
ZERO = "#2C2C2C"
NOTE = "#555555"

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


def apply_csv_override(label, row, path, spec):
    """Return row with CSV values if spec found; print override."""
    label_s, beta, lo, hi, color, pval = row
    loaded = load_coef(path, spec)
    if loaded is None:
        return row
    csv_beta, csv_lo, csv_hi, csv_p = loaded
    if (
        abs(beta - csv_beta) > 0.001
        or abs(lo - csv_lo) > 0.001
        or abs(hi - csv_hi) > 0.001
        or abs(pval - csv_p) > 0.001
    ):
        print(
            f"Override {label}: "
            f"β={csv_beta:.3f}, CI=[{csv_lo:.3f}, {csv_hi:.3f}], p={csv_p:.3f} "
            f"({os.path.basename(path)}:{spec})"
        )
    return label_s, csv_beta, csv_lo, csv_hi, color, csv_p


def main():
    round5_path = os.path.join(RESULTS_DIR, "round5", "round5_consolidated.csv")
    mechanism_path = os.path.join(RESULTS_DIR, "mechanism", "mechanism_consolidated.csv")
    spec_path = os.path.join(RESULTS_DIR, "spec_consolidated_table.csv")
    main_path = os.path.join(REG_DIR, "main_regression_results.csv")

    try:
        pd.read_csv(round5_path)
        print("Loaded round5 consolidated results")
    except FileNotFoundError:
        print("CSV not found — using hardcoded values")

    # Panel A: concentration outcomes on 0-1 index scale
    rows_a = [
        ("Top-three exposure share (IV)", 0.047, 0.029, 0.065, NAVY, 0.001),
        ("Top-three exposure share (Bartik)", 0.037, 0.003, 0.071, STEEL, 0.021),
    ]
    # Panel B: log-scale outcomes (TWFE for kcal; IV for import quantities)
    rows_b = [
        ("Log kcal per capita (TWFE)", 0.005, -0.001, 0.011, SLATE, 0.164),
        ("Log total imports (IV)", 0.129, 0.063, 0.195, STEEL, 0.001),
        ("Log top-3 imports (IV)", 0.476, 0.380, 0.572, DARK, 0.001),
    ]

    specs_a = [
        ("Top-three exposure share (IV)", mechanism_path, "S_IV_CONCENTRATION"),
        ("Top-three exposure share (Bartik)", mechanism_path, "S_BARTIK_CONCENTRATION"),
    ]
    specs_b = [
        ("Log kcal per capita (TWFE)", spec_path, "1B_linear"),
        ("Log total imports (IV)", round5_path, "S_DENOMINATOR"),
        ("Log top-3 imports (IV)", round5_path, "S_NUMERATOR"),
    ]

    rows_a = [
        apply_csv_override(lab, row, path, spec)
        for row, (lab, path, spec) in zip(rows_a, specs_a)
    ]
    rows_b = [
        apply_csv_override(lab, row, path, spec)
        for row, (lab, path, spec) in zip(rows_b, specs_b)
    ]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 3.4), gridspec_kw={"wspace": 0.72}
    )
    fig.patch.set_facecolor("white")

    y_pos = [1, 0]
    for i, (label, beta, lo, hi, color, pval) in enumerate(rows_a):
        y = y_pos[i]
        is_ns = pval >= 0.10
        ptxt = stars(pval)
        ax_a.plot([lo, hi], [y, y], color=color, linewidth=1.4, zorder=3)
        ax_a.plot(beta, y, "o", color=color, markersize=6 if not is_ns else 4.5, zorder=4)
        if label == "Log top-3 imports (IV)":
            ax_a.text(
                beta, y + 0.28, f"{beta:.3f}, {ptxt}",
                va="bottom", ha="center", fontsize=7.5, fontweight="bold", color=DARK,
            )
        else:
            ax_a.text(
                hi + 0.012, y, f"{beta:.3f}, {ptxt}",
                va="center", ha="left", fontsize=7.5,
                fontweight="bold" if not is_ns else "normal",
                color=NOTE if is_ns else color,
            )

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([r[0] for r in rows_a], fontsize=7.5, color="#1a1a1a")
    ax_a.set_ylim(-0.7, 1.7)
    ax_a.set_xlim(-0.02, 0.10)
    ax_a.set_xlabel(r"$\beta$ on 0–1 exposure index (95% CI)", fontsize=8, color="#1a1a1a")
    ax_a.tick_params(axis="x", labelsize=7.5)
    ax_a.axvline(0, color=ZERO, linewidth=0.9, linestyle="--", zorder=1)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.spines["left"].set_visible(False)
    ax_a.tick_params(left=False)
    ax_a.text(-0.16, 1.06, "A", transform=ax_a.transAxes,
              fontsize=11, fontweight="bold", color="#1a1a1a")
    ax_a.set_title(
        "Exposure channel\n(0--1 index scale)",
        fontsize=7.5, pad=10, color="#444444", loc="center",
    )

    y_pos_b = [2, 1, 0]
    for i, (label, beta, lo, hi, color, pval) in enumerate(rows_b):
        y = y_pos_b[i]
        is_ns = pval >= 0.10
        ptxt = stars(pval)
        ax_b.plot([lo, hi], [y, y], color=color, linewidth=1.4, zorder=3)
        ax_b.plot(beta, y, "o", color=color, markersize=6 if not is_ns else 4.5, zorder=4)
        ax_b.text(
            hi + 0.001, y, f"{beta:.3f}, {ptxt}",
            va="center", ha="left", fontsize=7.5,
            fontweight="bold" if not is_ns else "normal",
            color=NOTE if is_ns else color,
        )

    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels([r[0] for r in rows_b], fontsize=7.5, color="#1a1a1a")
    ax_b.set_ylim(-0.7, 2.7)
    ax_b.set_xlim(-0.06, 0.65)
    ax_b.set_xlabel(r"$\beta$ on log outcomes (95% CI)", fontsize=8, color="#1a1a1a")
    ax_b.tick_params(axis="x", labelsize=7.5)
    ax_b.axvline(0, color=ZERO, linewidth=0.9, linestyle="--", zorder=1)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(left=False)
    ax_b.text(-0.20, 1.06, "B", transform=ax_b.transAxes,
              fontsize=11, fontweight="bold", color="#1a1a1a")
    ax_b.set_title(
        "Quantity channel\n(log scale; kcal uses TWFE)",
        fontsize=7.5, pad=10, color="#444444", loc="center",
    )
    ax_b.text(
        0.98, -0.18,
        "IV import outcomes: partial F = 108",
        transform=ax_b.transAxes, fontsize=6.5, va="top", ha="right",
        color=NAVY,
        bbox=dict(
            facecolor="#F0F4F8", edgecolor="#AABCCC",
            boxstyle="round,pad=0.35", linewidth=0.7,
        ),
    )

    fig.suptitle(
        "Exporter floods raise flood-exposed import share"
        " without reducing caloric supply",
        fontsize=9.5, fontweight="bold", y=1.02, color="#1a1a1a",
    )

    plt.subplots_adjust(
        top=0.82,
        bottom=0.20,
        left=0.18,
        right=0.97,
        wspace=0.72,
    )

    print("Panel A xlim:", ax_a.get_xlim())
    print("Panel B xlim:", ax_b.get_xlim())
    print("Panel A ytick labels:", [r[0] for r in rows_a])
    print("Panel B ytick labels:", [r[0] for r in rows_b])
    assert rows_b[0][4] == SLATE, "kcal row B must be SLATE"
    print("Color checks passed.")

    out_png = os.path.join(FIG_DIR, "fig2_composition_channel.png")
    out_pdf = os.path.join(FIG_DIR, "fig2_composition_channel.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Fig 2 patched and saved to outputs/figures/")

    scripts_dir = os.path.join(BASE, "scripts", "figures")
    print("\nCurrent scripts/figures/ contents:")
    for f in sorted(os.listdir(scripts_dir)):
        print(f"  {f}")
    print("\nCurrent outputs/figures/ contents:")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
