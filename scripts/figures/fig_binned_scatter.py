import os
import shutil
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy import stats

# ── Cleanup ───────────────────────────────────────────────────────────────────
for root, dirs, files in os.walk('scripts/figures/'):
    for f in files:
        if 'binned' in f and f != 'fig_binned_scatter.py':
            os.remove(os.path.join(root, f))
            print(f"Deleted: {os.path.join(root, f)}")
    for d in list(dirs):
        if d in ['revised', 'final']:
            shutil.rmtree(os.path.join(root, d))

for root, dirs, files in os.walk('outputs/figures/'):
    for f in files:
        if 'binned' in f and re.search(r'(_v\d+|revised|final)', f):
            os.remove(os.path.join(root, f))
    for d in list(dirs):
        if d in ['revised', 'final']:
            shutil.rmtree(os.path.join(root, d))

# ── Color system ──────────────────────────────────────────────────────────────
LIC_COL = '#D4622A'
LMC_COL = '#E8A838'
UMC_COL = '#4A7BA7'
HIC_COL = '#1B3A5C'
NAVY    = '#1B3A5C'
STEEL   = '#4A7BA7'
NOTE    = '#555555'
GRID    = '#EEEEEE'
ZERO    = '#2C2C2C'

income_colors = {
    'LIC': LIC_COL,
    'LMC': LMC_COL,
    'UMC': UMC_COL,
    'HIC': HIC_COL,
}

# ── Step 1: Load data ─────────────────────────────────────────────────────────
panel = pd.read_csv('outputs/regression/master_panel_round4.csv')
print("Panel columns:", panel.columns.tolist())
print("Panel shape:", panel.shape)

# Column detection — extend lookup to include 'fte_total'
fte_col = next(
    (c for c in panel.columns
     if c.lower() in ['fte', 'fte_total', 'flood_trade_exposure', 'mean_fte']),
    None)
conc_col = next(
    (c for c in panel.columns
     if 'top3' in c.lower() and 'conc' in c.lower()),
    None)
income_col = next(
    (c for c in panel.columns
     if c.lower() == 'income_level_id'),
    None) or next(
    (c for c in panel.columns
     if 'income' in c.lower()),
    None)
iso_col = next(
    (c for c in panel.columns
     if c.lower() in ['iso3', 'iso', 'country_code']),
    None)

print(f"FTE column:         {fte_col}")
print(f"Concentration col:  {conc_col}")
print(f"Income column:      {income_col}")
print(f"ISO column:         {iso_col}")

if not fte_col or not conc_col:
    raise ValueError(
        "Could not find FTE or concentration columns. "
        "Print panel.columns and adapt column names above."
    )

panel_clean = panel[[iso_col, fte_col, conc_col, income_col, 'year']].dropna().copy()
panel_clean = panel_clean.rename(columns={
    fte_col:    'fte',
    conc_col:   'concentration',
    income_col: 'income_group',
    iso_col:    'iso3',
})

# income_level_id is already 'LIC'/'LMC'/'UMC'/'HIC'
income_map = {
    'Low income': 'LIC',      'Low Income': 'LIC',      'L':  'LIC',
    'Lower middle income': 'LMC', 'Lower Middle Income': 'LMC', 'LM': 'LMC',
    'Upper middle income': 'UMC', 'Upper Middle Income': 'UMC', 'UM': 'UMC',
    'High income': 'HIC',     'High Income': 'HIC',     'H':  'HIC',
    'LIC': 'LIC', 'LMC': 'LMC', 'UMC': 'UMC', 'HIC': 'HIC',
}
panel_clean['income_label'] = panel_clean['income_group'].map(income_map).fillna('UMC')
panel_clean['income_color'] = panel_clean['income_label'].map(income_colors)

print(f"Clean panel: {len(panel_clean)} obs, {panel_clean['iso3'].nunique()} countries")
print("Income distribution:")
print(panel_clean['income_label'].value_counts())

# Remove extreme FTE outliers (top 1%)
fte_99 = panel_clean['fte'].quantile(0.99)
panel_clean = panel_clean[panel_clean['fte'] <= fte_99].copy()
print(f"After outlier removal: {len(panel_clean)} obs")
print(f"FTE range: {panel_clean['fte'].min():.4f} — {panel_clean['fte'].max():.4f}")

# ── Step 2: Compute binned statistics ─────────────────────────────────────────
N_BINS = 20

panel_clean['fte_bin'] = pd.cut(panel_clean['fte'], bins=N_BINS, labels=False)

bin_stats = panel_clean.groupby('fte_bin').agg(
    mean_fte   = ('fte', 'mean'),
    mean_conc  = ('concentration', 'mean'),
    std_conc   = ('concentration', 'std'),
    n_obs      = ('concentration', 'count'),
    median_fte = ('fte', 'median'),
).reset_index()

bin_stats['se_conc'] = bin_stats['std_conc'] / np.sqrt(bin_stats['n_obs'])
bin_stats['ci95']    = 1.96 * bin_stats['se_conc']

def majority_income(group):
    counts = group['income_label'].value_counts()
    return counts.index[0] if len(counts) > 0 else 'UMC'

income_by_bin = (
    panel_clean.groupby('fte_bin', group_keys=False)
               .apply(majority_income, include_groups=False)
               .reset_index()
)
income_by_bin.columns = ['fte_bin', 'majority_income']
bin_stats = bin_stats.merge(income_by_bin, on='fte_bin')
bin_stats['bin_color'] = bin_stats['majority_income'].map(income_colors)

print(f"\nBins computed: {len(bin_stats)}")
print(bin_stats[['mean_fte', 'mean_conc', 'n_obs', 'majority_income']].to_string())

# Patch 1: drop sparse high-FTE bins (N < 15 unreliable CIs)
bin_stats = bin_stats[bin_stats['n_obs'] >= 15].copy()
print(f"\nBins after N>=15 filter: {len(bin_stats)}")
print(bin_stats[['mean_fte', 'mean_conc', 'n_obs']].to_string())

# ── Step 3: Regression lines ──────────────────────────────────────────────────
fte_vals  = panel_clean['fte'].values
conc_vals = panel_clean['concentration'].values

ols_slope, ols_intercept, ols_r, ols_p, ols_se = stats.linregress(fte_vals, conc_vals)
print(f"\nOLS: β={ols_slope:.4f}, R²={ols_r**2:.3f}, p={ols_p:.4f}")

mean_fte_overall  = panel_clean['fte'].mean()
mean_conc_overall = panel_clean['concentration'].mean()
iv_slope          = 0.047
iv_intercept      = mean_conc_overall - iv_slope * mean_fte_overall
print(f"IV implied: β=0.047, intercept={iv_intercept:.4f}")

# X range based on remaining bins after N filter
x_line = np.linspace(
    bin_stats['mean_fte'].min(),
    bin_stats['mean_fte'].max() * 1.05,
    100,
)
ols_y  = ols_slope * x_line + ols_intercept
iv_y   = iv_slope  * x_line + iv_intercept

# IV 95% CI band
iv_lo, iv_hi = 0.029, 0.065
iv_y_lo = iv_lo * x_line + (mean_conc_overall - iv_lo * mean_fte_overall)
iv_y_hi = iv_hi * x_line + (mean_conc_overall - iv_hi * mean_fte_overall)

# ── Step 4: Figure layout ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(9.5, 7.8))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    width_ratios=[1, 1, 0.28],
    height_ratios=[0.28, 1, 1],
    hspace=0.05,
    wspace=0.05,
)

ax_main  = fig.add_subplot(gs[1:, :2])
ax_top   = fig.add_subplot(gs[0, :2], sharex=ax_main)
ax_right = fig.add_subplot(gs[1:, 2],  sharey=ax_main)

for ax in [ax_main, ax_top, ax_right]:
    ax.set_facecolor('white')

# ── Step 5: Main scatter ──────────────────────────────────────────────────────
# Patch 1: sqrt scaling, larger range (min=80, max=600)
def scale_bubble(n, min_size=80, max_size=600):
    n_min = bin_stats['n_obs'].min()
    n_max = bin_stats['n_obs'].max()
    if n_max == n_min:
        return min_size
    normalized = np.sqrt((n - n_min) / (n_max - n_min))
    return min_size + normalized * (max_size - min_size)

bubble_sizes = bin_stats['n_obs'].apply(scale_bubble).values

ax_main.scatter(
    bin_stats['mean_fte'], bin_stats['mean_conc'],
    s=bubble_sizes, c=bin_stats['bin_color'],
    alpha=0.85, edgecolors='white', linewidth=0.8, zorder=4,
)

for _, row in bin_stats.iterrows():
    ax_main.plot(
        [row['mean_fte'], row['mean_fte']],
        [row['mean_conc'] - row['ci95'], row['mean_conc'] + row['ci95']],
        color=row['bin_color'], linewidth=1.0, alpha=0.6, zorder=3,
    )

# IV CI band
ax_main.fill_between(
    x_line, iv_y_lo, iv_y_hi,
    color=NAVY, alpha=0.10, zorder=2,
)

# OLS line
ax_main.plot(
    x_line, ols_y,
    color=STEEL, linewidth=1.5, linestyle='--', zorder=5,
    label=f'OLS: β={ols_slope:.3f}',
)

# IV line (on top)
ax_main.plot(
    x_line, iv_y,
    color=NAVY, linewidth=2.0, linestyle='-', zorder=6,
    label='IV estimate: β=0.047',
)

ax_main.grid(True, color=GRID, linewidth=0.5, linestyle='-', zorder=1)
ax_main.set_xlabel('Mean FTE (thousand km$^2$/year)',
                   fontsize=11, color='#1a1a1a', labelpad=6)
ax_main.set_ylabel('Top-three flood-exposed import share',
                   fontsize=11, color='#1a1a1a', labelpad=6)
ax_main.tick_params(axis='both', labelsize=10)
ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)

# N annotation
ax_main.text(
    0.97, 0.04,
    f"N = {len(panel_clean):,} country-years\n"
    f"{panel_clean['iso3'].nunique()} countries, 2000–2021",
    transform=ax_main.transAxes,
    fontsize=9, color=NOTE, ha='right', va='bottom',
    bbox=dict(facecolor='white', edgecolor='#CCCCCC',
              boxstyle='round,pad=0.3', linewidth=0.7, alpha=0.9),
)

# Legend
income_patches = [
    mpatches.Patch(facecolor=LIC_COL, label='Low income (LIC)'),
    mpatches.Patch(facecolor=LMC_COL, label='Lower-middle (LMC)'),
    mpatches.Patch(facecolor=UMC_COL, label='Upper-middle (UMC)'),
    mpatches.Patch(facecolor=HIC_COL, label='High income (HIC)'),
]
line_handles = [
    mlines.Line2D([0], [0], color=NAVY, linewidth=2, linestyle='-',
                  label='IV slope: β=0.047 (partial F=108)'),
    mlines.Line2D([0], [0], color=STEEL, linewidth=1.5, linestyle='--',
                  label=f'OLS slope: β={ols_slope:.3f}'),
]
size_note = mlines.Line2D(
    [0], [0], marker='o', color='w',
    markerfacecolor='#AAAAAA', markersize=8,
    label='Bubble size = N obs in bin',
)
# Patch 2: legend compressed inside top-left (data-sparse area)
ax_main.legend(
    handles=income_patches + line_handles + [size_note],
    fontsize=8.5,
    loc='upper left',
    bbox_to_anchor=(0.01, 0.99),
    framealpha=0.92, edgecolor='#CCCCCC', frameon=True, ncol=1,
    handlelength=1.5, handleheight=0.8,
    borderpad=0.5, labelspacing=0.4,
)

# ── Step 6: Top marginal — FTE distribution ───────────────────────────────────
income_order  = ['HIC', 'UMC', 'LMC', 'LIC']
colors_order  = [HIC_COL, UMC_COL, LMC_COL, LIC_COL]

fte_bins_hist = np.linspace(
    panel_clean['fte'].min(),
    panel_clean['fte'].quantile(0.98),
    25,
)
bottom_vals = np.zeros(len(fte_bins_hist) - 1)

for income, color in zip(income_order, colors_order):
    sub    = panel_clean[panel_clean['income_label'] == income]['fte']
    counts, _ = np.histogram(sub, bins=fte_bins_hist)
    ax_top.bar(
        fte_bins_hist[:-1], counts,
        width=np.diff(fte_bins_hist), bottom=bottom_vals,
        color=color, alpha=0.80, align='edge', linewidth=0,
    )
    bottom_vals += counts

# Patch 3 (P2): linear scale clipped at 500; first bar annotation
ax_top.set_yscale('linear')
ax_top.set_ylim(0, 500)
ax_top.set_ylabel('Count\n(clipped at 500)', fontsize=9, color='#1a1a1a')
# Annotate the truncated first bar
ax_top.annotate(
    '2,100+',
    xy=(fte_bins_hist[0] + np.diff(fte_bins_hist)[0] * 0.5, 500),
    xytext=(fte_bins_hist[0] + np.diff(fte_bins_hist)[0] * 3.5, 420),
    fontsize=8, color=NOTE,
    arrowprops=dict(arrowstyle='->', color=NOTE, linewidth=0.8),
)
ax_top.tick_params(axis='y', labelsize=8.5)
ax_top.tick_params(axis='x', labelbottom=False)
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_top.set_facecolor('white')
ax_top.grid(True, color=GRID, linewidth=0.4, linestyle='-', axis='y', zorder=0)

# ── Step 7: Right marginal — concentration distribution ───────────────────────
conc_bins_hist = np.linspace(
    panel_clean['concentration'].min(),
    panel_clean['concentration'].quantile(0.99),
    25,
)
left_vals = np.zeros(len(conc_bins_hist) - 1)

for income, color in zip(income_order, colors_order):
    sub    = panel_clean[panel_clean['income_label'] == income]['concentration']
    counts, _ = np.histogram(sub, bins=conc_bins_hist)
    ax_right.barh(
        conc_bins_hist[:-1], counts,
        height=np.diff(conc_bins_hist), left=left_vals,
        color=color, alpha=0.80, align='edge', linewidth=0,
    )
    left_vals += counts

ax_right.set_xlabel('Count', fontsize=9, color='#1a1a1a')
ax_right.tick_params(axis='x', labelsize=8.5)
ax_right.tick_params(axis='y', labelleft=False)
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.set_facecolor('white')
ax_right.grid(True, color=GRID, linewidth=0.4, linestyle='-', axis='x', zorder=0)

# ── Step 8: Title and footnote ────────────────────────────────────────────────
fig.suptitle(
    'Raw relationship: flood trade exposure predicts\n'
    'flood-exposed import share rises with exporter FTE',
    fontsize=12, fontweight='bold', y=0.98, color='#1a1a1a',
)

fig.text(
    0.10, 0.01,
    'Each bubble = one FTE bin (20 equal-width bins). '
    'Bubble size proportional to N country-years in bin. '
    'Error bars = 95% CI of mean exposure.\n'
    'IV slope (β=0.047, p<0.001, partial F=108) exceeds OLS slope — '
    'consistent with attenuation bias, though other explanations are possible. '
    'Marginals show FTE and exposure distributions by income group.',
    fontsize=8, color=NOTE, style='italic', va='bottom', ha='left',
)

# ── Step 9: Layout ────────────────────────────────────────────────────────────
plt.subplots_adjust(top=0.91, bottom=0.11, left=0.10, right=0.97)

# Patch 2: tighten axes to data range
y_min_ax = max(0, bin_stats['mean_conc'].min() - bin_stats['ci95'].max() - 0.05)
y_max_ax = bin_stats['mean_conc'].max() + bin_stats['ci95'].max() + 0.08
ax_main.set_ylim(y_min_ax, y_max_ax)
x_min_ax = -bin_stats['mean_fte'].max() * 0.03
x_max_ax =  bin_stats['mean_fte'].max() * 1.08
ax_main.set_xlim(x_min_ax, x_max_ax)
print(f"Y axis set to: {y_min_ax:.3f} — {y_max_ax:.3f}")
print(f"X axis set to: {x_min_ax:.3f} — {x_max_ax:.3f}")

# ── Step 10: Verify ───────────────────────────────────────────────────────────
print("\nVerification:")
print(f"  N bins plotted: {len(bin_stats)}")
print(f"  OLS β: {ols_slope:.4f}, p={ols_p:.4f}")
print(f"  IV β (hardcoded): 0.047")
print(f"  IV > OLS: {0.047 > ols_slope} "
      f"(expected True — IV corrects attenuation)")
print(f"  X axis range: {ax_main.get_xlim()}")
print(f"  Y axis range: {ax_main.get_ylim()}")
print("All checks complete.")

# ── Step 11: Save ─────────────────────────────────────────────────────────────
plt.savefig(
    'outputs/figures/fig_binned_scatter.png',
    dpi=300, bbox_inches='tight', facecolor='white',
)
plt.savefig(
    'outputs/figures/fig_binned_scatter.pdf',
    bbox_inches='tight', facecolor='white',
)
plt.close()
print("\nBinned scatter saved to outputs/figures/")

print("\nCurrent outputs/figures/ contents:")
for f in sorted(os.listdir('outputs/figures/')):
    print(f"  {f}")
print("\nCurrent scripts/figures/ contents:")
for f in sorted(os.listdir('scripts/figures/')):
    print(f"  {f}")
