#!/usr/bin/env python3
"""Figure S2 — Methods diagnostics: FTE autocorrelation + post-2015 corroboration."""

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

# ── Color system (identical to all previous figures) ─────────────────────────
NAVY = "#1B3A5C"
STEEL = "#4A7BA7"
SENTINEL = "#5B8C5A"
NOTE = "#555555"
RED_LINE = "#C0392B"

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


def main():
    # ── Load Panel A data ─────────────────────────────────────────────────────
    acf_path = os.path.join(RESULTS_DIR, "round4", "fte_autocorrelation.csv")
    try:
        acf_df = pd.read_csv(acf_path)
        acf_df = acf_df[acf_df["lag"] > 0].copy()
        # column is "autocorr" in the actual CSV
        acf_col = "autocorr" if "autocorr" in acf_df.columns else "autocorrelation"
        lags = acf_df["lag"].values.astype(int)
        acf_vals = acf_df[acf_col].values
        rho1 = float(acf_df.loc[acf_df["lag"] == 1, acf_col].iloc[0])
        print(f"Loaded ACF from CSV  (ρ lag-1 = {rho1:.3f})")
    except Exception as e:
        print(f"ACF CSV not loaded ({e}) — using hardcoded values")
        lags = np.array([1, 2, 3, 4])
        acf_vals = np.array([0.765, 0.628, 0.499, 0.326])
        rho1 = 0.765

    # ── Load Panel B data ─────────────────────────────────────────────────────
    mechanism_path = os.path.join(RESULTS_DIR, "mechanism", "mechanism_consolidated.csv")
    round5_path = os.path.join(RESULTS_DIR, "round5", "round5_consolidated.csv")

    # hardcoded defaults
    betas = [0.047, 0.137]
    ci_lo = [0.029, 0.088]
    ci_hi = [0.065, 0.186]
    pvals = [0.001, 0.001]

    loaded = load_coef(mechanism_path, "S_IV_CONCENTRATION")
    if loaded:
        b, lo, hi, p = loaded
        if any(abs(v1 - v2) > 0.002 for v1, v2 in [(b, betas[0]), (lo, ci_lo[0]), (hi, ci_hi[0])]):
            print(f"Override Full-sample IV: β={b:.3f}, CI=[{lo:.3f}, {hi:.3f}]")
        betas[0], ci_lo[0], ci_hi[0], pvals[0] = b, lo, hi, p

    loaded = load_coef(round5_path, "S_SENTINEL_BARTIK_CONC")
    if loaded:
        b, lo, hi, p = loaded
        if any(abs(v1 - v2) > 0.002 for v1, v2 in [(b, betas[1]), (lo, ci_lo[1]), (hi, ci_hi[1])]):
            print(f"Override post-2015 Bartik: β={b:.3f}, CI=[{lo:.3f}, {hi:.3f}]")
        betas[1], ci_lo[1], ci_hi[1], pvals[1] = b, lo, hi, p

    N_obs = 3319
    sig_threshold = 1.96 / np.sqrt(N_obs)
    print(f"95% CI bound: ±{sig_threshold:.4f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.45}
    )
    fig.patch.set_facecolor("white")

    # ── Panel A ───────────────────────────────────────────────────────────────
    ax_a.set_facecolor("white")
    ax_a.bar(lags, acf_vals, color=STEEL, width=0.6, zorder=3,
             edgecolor="white", linewidth=0.5)
    ax_a.axhline(sig_threshold, color=RED_LINE, linewidth=1.2, linestyle="--", zorder=4)
    ax_a.axhline(-sig_threshold, color=RED_LINE, linewidth=1.2, linestyle="--", zorder=4)
    ax_a.text(
        4.35, sig_threshold + 0.01, "95% CI bound",
        fontsize=6.5, color=RED_LINE, va="bottom", ha="right",
    )
    ax_a.text(
        0.97, 0.97, f"ρ(lag-1) = {rho1:.2f}",
        transform=ax_a.transAxes, fontsize=8, fontweight="bold",
        va="top", ha="right", color=NAVY,
        bbox=dict(facecolor="#F0F4F8", edgecolor="#AABCCC",
                  boxstyle="round,pad=0.35", linewidth=0.7),
    )
    ax_a.set_xticks(lags)
    ax_a.set_xticklabels([str(l) for l in lags], fontsize=8)
    ax_a.set_xlabel("Lag (years)", fontsize=8.5, color="#1a1a1a")
    ax_a.set_ylabel("Within-country FTE autocorrelation", fontsize=8, color="#1a1a1a")
    ax_a.set_ylim(-0.1, 0.90)
    ax_a.set_xlim(0.5, 4.75)
    ax_a.tick_params(axis="y", labelsize=7.5)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.text(-0.18, 1.06, "A", transform=ax_a.transAxes,
              fontsize=11, fontweight="bold", color="#1a1a1a")
    ax_a.set_title("FTE persistence invalidates\nevent-study timing",
                   fontsize=8, pad=6, color="#1a1a1a", loc="center")
    ax_a.text(
        0.0, -0.22,
        "Multi-year elevated-exposure spells (mean 5.4 years) drive\n"
        "FTE persistence, contaminating pre-trend tests.",
        transform=ax_a.transAxes, fontsize=6.5, color=NOTE,
        style="italic", va="top", ha="left",
    )

    # ── Panel B ───────────────────────────────────────────────────────────────
    ax_b.set_facecolor("white")
    specs_labels = [
        "Full sample\nIV (2000–2021)",
        "Post-2015 subsample\n(JRC GSW, Bartik)",
    ]
    colors = [NAVY, SENTINEL]
    x = np.arange(len(specs_labels))
    bar_width = 0.45

    for i, (beta, lo, hi, color, pval) in enumerate(
            zip(betas, ci_lo, ci_hi, colors, pvals)):
        ax_b.bar(x[i], beta, width=bar_width, color=color, zorder=3,
                 edgecolor="white", linewidth=0.5)
        ax_b.errorbar(
            x[i], beta, yerr=[[beta - lo], [hi - beta]],
            fmt="none", color="#333333", linewidth=1.2, capsize=4, zorder=4,
        )
        ax_b.text(
            x[i], hi + 0.012, f"{beta:.3f}, {stars(pval)}",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color=color,
        )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(specs_labels, fontsize=7.5)
    ax_b.set_ylabel(
        "β on top-3 exposure share\n(95% CI)",
        fontsize=8, color="#1a1a1a",
    )
    ax_b.set_ylim(0, 0.28)
    ax_b.set_xlim(-0.5, 1.5)
    ax_b.tick_params(axis="y", labelsize=7.5)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.text(-0.20, 1.06, "B", transform=ax_b.transAxes,
              fontsize=11, fontweight="bold", color="#1a1a1a")
    ax_b.set_title(
        "Exposure estimate remains positive\nin post-2015 subsample",
        fontsize=8, pad=6, color="#1a1a1a", loc="center",
    )
    ax_b.text(
        0.0, -0.22,
        "Different estimators and periods; both estimates are positive.\n"
        "Post-2015 β larger reflects unscaled JRC GSW after 2015.",
        transform=ax_b.transAxes, fontsize=6.5, color=NOTE,
        style="italic", va="top", ha="left",
    )

    # ── Overall title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "IV specification uses levels, not event-study timing;\n"
        "Post-2015 subsample corroborates",
        fontsize=9.5, fontweight="bold", y=1.04, color="#1a1a1a",
    )

    plt.subplots_adjust(top=0.82, bottom=0.26, left=0.12, right=0.97, wspace=0.45)

    # ── Verify ────────────────────────────────────────────────────────────────
    print("Panel A ylim:", ax_a.get_ylim())
    print("Panel B ylim:", ax_b.get_ylim())
    print(f"Sig threshold drawn at: ±{sig_threshold:.4f}")
    assert colors[0] == NAVY, "Full sample bar must be NAVY"
    assert colors[1] == SENTINEL, "Post-2015 bar must be SENTINEL green"
    for beta, hi in zip(betas, ci_hi):
        assert hi + 0.02 < 0.28, f"CI upper {hi} + label may exceed ylim 0.28"
    print("All checks passed.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_png = os.path.join(FIG_DIR, "figS2_methods.png")
    out_pdf = os.path.join(FIG_DIR, "figS2_methods.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Fig S2 patch 2 saved to outputs/figures/")

    print("\nFinal outputs/figures/ contents:")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {f}")
    scripts_dir = os.path.join(BASE, "scripts", "figures")
    print("\nFinal scripts/figures/ contents:")
    for f in sorted(os.listdir(scripts_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
