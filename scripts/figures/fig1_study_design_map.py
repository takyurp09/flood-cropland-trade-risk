#!/usr/bin/env python3
"""Figure 1 — Study design map (FCE exporters / FTE importers)."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
from cartopy.io.shapereader import natural_earth
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(BASE, "outputs", "figures")
FCE_DIR = os.path.join(BASE, "outputs", "fce")
REG_DIR = os.path.join(BASE, "outputs", "regression")

os.makedirs(FIG_DIR, exist_ok=True)

OCEAN_BG = "#D6E8F5"
LAND_NODATA = "#EEEEEE"
LAND_EDGE = "white"
SSA_ORANGE = "#D4622A"
LABEL_STROKE = "#1a1a1a"

ROBINSON = ccrs.Robinson()
GEO = ccrs.PlateCarree()

SSA_ISO = [
    "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF",
    "TCD", "COM", "COD", "COG", "CIV", "DJI", "GNQ", "ERI",
    "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO",
    "LBR", "MDG", "MWI", "MLI", "MRT", "MUS", "MOZ", "NAM",
    "NER", "NGA", "RWA", "STP", "SEN", "SLE", "SOM", "ZAF",
    "SSD", "SDN", "SWZ", "TZA", "TGO", "UGA", "ZMB", "ZWE",
]

PANEL_A_LABELS = [
    (-105, 42, "USA"), (-85, 68, "CAN"), (-50, -13, "BRA"),
    (70, 62, "RUS"), (80, 20, "IND"), (115, 33, "CHN"),
    (62, 33, "PAK"), (134, -26, "AUS"),
]

PANEL_B_LABELS_DEFAULT = {
    "HND": (-90, 13), "DOM": (-67, 20), "QAT": (44, 33),
    "NPL": (58, 24), "BTN": (108, 30),
}

PANEL_B_COUNTRY_CENTERS = {
    "HND": (-86.2, 14.6), "DOM": (-70.2, 18.7), "QAT": (51.2, 25.3),
    "NPL": (84.0, 28.0), "BTN": (90.4, 27.5),
    "BLZ": (-88.5, 17.2), "HTI": (-72.3, 18.9), "BHS": (-76.0, 24.3),
}

TEXT_EFFECTS = [pe.withStroke(linewidth=1.8, foreground="#111111")]
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def load_fce():
    for name in ("fce_final_panel_v2.csv", "fce_final_panel.csv"):
        path = os.path.join(FCE_DIR, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            col = "fce_central" if "fce_central" in df.columns else None
            if col is None:
                print("FCE columns:", df.columns.tolist(), file=sys.stderr)
                raise KeyError("fce_central not found")
            return df.groupby("iso3", as_index=False)[col].sum().rename(columns={col: "fce"})
    raise FileNotFoundError("No FCE panel in outputs/fce/")


def load_fte():
    path = os.path.join(REG_DIR, "master_panel_round4.csv")
    df = pd.read_csv(path)
    return df.groupby("iso3", as_index=False)["fte_total"].mean().rename(columns={"fte_total": "fte"})


def load_world(resolution="110m"):
    shpfile = natural_earth(resolution=resolution, category="cultural", name="admin_0_countries")
    world = gpd.read_file(shpfile)
    world["iso3"] = world["ADM0_A3"] if "ADM0_A3" in world.columns else world["ISO_A3"]
    world.loc[world["iso3"] == "SDS", "iso3"] = "SSD"
    world = world[world["iso3"].notna() & (world["iso3"] != "-99")].copy()
    world = world[world.geometry.representative_point().y > -60].copy()
    return world


def dissolve_ssa(world_50m, min_part_area=5.0):
    """Dissolve SSA mainland boundary.

    Each country geometry is pre-filtered to retain only parts whose
    area (in square degrees) exceeds min_part_area.  This strips
    island nations (CPV, COM, MUS, STP, SYC) and coastal island
    fragments entirely before the union, preventing broken-island
    artefacts on the final boundary ring.
    """
    ssa = world_50m[world_50m["iso3"].isin(SSA_ISO)].copy()
    ssa["geometry"] = ssa.geometry.buffer(0)

    mainland_parts = []
    for geom in ssa.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiPolygon):
            big = [p for p in geom.geoms if p.area >= min_part_area]
            if big:
                mainland_parts.extend(big)
        elif isinstance(geom, Polygon):
            if geom.area >= min_part_area:
                mainland_parts.append(geom)

    if not mainland_parts:
        raise ValueError("No SSA mainland polygons survived the area filter — "
                         "lower min_part_area.")

    dissolved = unary_union(mainland_parts)
    # After dissolving mainland-only parts, keep the single largest piece
    if isinstance(dissolved, MultiPolygon):
        dissolved = max(dissolved.geoms, key=lambda g: g.area)
    if not isinstance(dissolved, Polygon):
        raise TypeError(f"Unexpected SSA dissolve type: {dissolved.geom_type}")
    return np.array(dissolved.exterior.coords)


def plot_choropleth(ax, world, value_col, cmap, vmin, vmax):
    ax.set_facecolor(OCEAN_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    world.plot(ax=ax, transform=GEO, color=LAND_NODATA, edgecolor=LAND_EDGE, linewidth=0.3, zorder=1)
    mask = world[value_col].notna() & (world[value_col] > 0)
    if mask.any():
        world.loc[mask].plot(
            ax=ax, transform=GEO, column=value_col, cmap=cmap,
            vmin=vmin, vmax=vmax, edgecolor=LAND_EDGE, linewidth=0.3, zorder=2,
        )


def add_labels(ax, label_list, leader_threshold=None, country_centers=None, always_leader=None):
    geo_trans = GEO._as_mpl_transform(ax)
    always_leader = always_leader or set()
    for lon, lat, txt in label_list:
        draw_leader = txt in always_leader
        if not draw_leader and leader_threshold is not None and country_centers:
            for olon, olat, otxt in label_list:
                if otxt != txt and np.hypot(lon - olon, lat - olat) < leader_threshold:
                    draw_leader = True
                    break
        if draw_leader and country_centers and txt in country_centers:
            clon, clat = country_centers[txt]
            ax.annotate("", xy=(clon, clat), xytext=(lon, lat),
                        xycoords=geo_trans, textcoords=geo_trans,
                        arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.6), zorder=9)
        ax.text(lon, lat, txt, transform=GEO, fontsize=6.5, fontweight="bold",
                color="white", ha="center", va="center", zorder=10, path_effects=TEXT_EFFECTS)


def label_positions_for_top5(top5_isos):
    labels, used = [], []
    for iso in top5_isos:
        if iso in PANEL_B_LABELS_DEFAULT:
            lon, lat = PANEL_B_LABELS_DEFAULT[iso]
        else:
            centers = {"BLZ": (-88.5, 17.2), "HTI": (-72.3, 18.9), "BGD": (90.4, 23.7)}
            lon, lat = centers.get(iso, (0, 0))
            lon, lat = lon + 8, lat + 8
        for ulon, ulat, _ in used:
            if np.hypot(lon - ulon, lat - ulat) < 8:
                lon += 12
                lat += 6
        labels.append((lon, lat, iso))
        used.append((lon, lat, iso))
    return labels


def main():
    print("Loading data...")
    fce_df = load_fce()
    fte_df = load_fte()
    world = load_world("110m")
    world_50m = load_world("50m")

    world_fce = world.merge(fce_df, on="iso3", how="left")
    world_fte = world.merge(fte_df, on="iso3", how="left")
    world_fte["fte_plot"] = world_fte["fte"].fillna(0)

    fce_vmax = min(13000, float(world_fce["fce"].quantile(0.95)))
    fte_vmax = min(1.4, float(world_fte["fte_plot"].quantile(0.95)) * 1.05)
    ssa_coords = dissolve_ssa(world_50m)
    panel_b_labels = label_positions_for_top5(fte_df.nlargest(5, "fte")["iso3"].tolist())

    fig = plt.figure(figsize=(7.2, 2.85), facecolor="white")
    ax_a = fig.add_axes([0.01, 0.04, 0.46, 0.88], projection=ROBINSON)
    ax_b = fig.add_axes([0.51, 0.04, 0.46, 0.88], projection=ROBINSON)
    for ax in (ax_a, ax_b):
        ax.set_global()

    plot_choropleth(ax_a, world_fce, "fce", plt.cm.YlOrRd, 0, fce_vmax)
    add_labels(ax_a, PANEL_A_LABELS)

    plot_choropleth(ax_b, world_fte, "fte_plot", plt.cm.Blues, 0, fte_vmax)
    ax_b.add_geometries(
        [LineString(ssa_coords)], crs=GEO, facecolor="none",
        edgecolor=SSA_ORANGE, linewidth=1.3, linestyle=(0, (8, 4)), zorder=6,
    )
    add_labels(ax_b, panel_b_labels, leader_threshold=10,
               country_centers=PANEL_B_COUNTRY_CENTERS, always_leader={"NPL", "BTN"})
    ax_b.legend(
        handles=[Line2D([0], [0], color=SSA_ORANGE, linewidth=1.8, linestyle="--",
                         label="Sub-Saharan Africa")],
        loc="upper left", fontsize=7, framealpha=0.85, edgecolor="#cccccc", frameon=True,
    )

    pos_a = ax_a.get_position()
    subtitle_y = pos_a.y1 + 0.012
    fig.text(pos_a.x0, subtitle_y, "A", fontsize=10, fontweight="bold", va="bottom", ha="left", color="#1a1a1a")
    fig.text(pos_a.x0 + pos_a.width / 2, subtitle_y, "Flood-exposed cropland 2000–2021 (FCE)",
             fontsize=8, ha="center", va="bottom", color="#333333")
    pos_b = ax_b.get_position()
    fig.text(pos_b.x0, subtitle_y, "B", fontsize=10, fontweight="bold", va="bottom", ha="left", color="#1a1a1a")
    fig.text(pos_b.x0 + pos_b.width / 2, subtitle_y, "Flood trade exposure 2000–2021 (FTE)",
             fontsize=8, ha="center", va="bottom", color="#333333")
    fig.suptitle(
        "Satellite-detected flood cropland exposure (A) propagates\n"
        "through trade networks to food-importing countries (B)",
        fontsize=9.5, fontweight="bold", y=1.01, color=LABEL_STROKE,
    )

    for ax, cmap, vmax, ticks, ticklabels, label in [
        (ax_a, plt.cm.YlOrRd, fce_vmax,
         [0, 2500, 5000, 7500, 10000, 12500],
         ["0", "2,500", "5,000", "7,500", "10,000", "12,500"],
         "Cumulative FCE (km²)"),
        (ax_b, plt.cm.Blues, fte_vmax,
         [0, 0.25, 0.5, 0.75, 1.0, 1.25],
         ["0", "0.25", "0.5", "0.75", "1.0", "1.25"],
         "Mean annual FTE (thousand km$^2$/year)"),
    ]:
        cax = ax.inset_axes([0.09, 0.10, 0.82, 0.055])
        cax.set_facecolor((1, 1, 1, 0.92))
        cax.patch.set_edgecolor("#cccccc")
        sm = ScalarMappable(cmap=cmap, norm=Normalize(0, vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=6.5)
        cbar.set_label(label, fontsize=7.5, color="#333333", labelpad=4)

    out_png = os.path.join(FIG_DIR, "fig1_study_design_map.png")
    out_pdf = os.path.join(FIG_DIR, "fig1_study_design_map.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    print(f"Fig 1 saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
