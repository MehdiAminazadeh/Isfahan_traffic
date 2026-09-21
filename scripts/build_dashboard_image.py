from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "outputs" / "dashboard" / "artifact.json"
OUTPUT = ROOT / "outputs" / "dashboard" / "isfahan_dashboard_full.png"

BLUE = "#1565c0"
BLUE_LIGHT = "#90caf9"
ORANGE = "#ef6c00"
GOLD = "#d6a700"
OLIVE = "#718238"
PINK = "#c65a83"
INK = "#1e2933"
MUTED = "#64717d"
GRID = "#dfe4e8"
PAPER = "#f4f6f7"
COMMUNITY_COLORS = {
    "Community 1": BLUE,
    "Community 2": ORANGE,
    "Community 3": OLIVE,
    "Community 4": PINK,
}


def _datasets():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return {
        name: pd.DataFrame(rows)
        for name, rows in payload["snapshot"]["datasets"].items()
    }


def _panel(ax, number, title, description):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(
        "%s. %s" % (number, title), loc="left", fontsize=13, color=INK,
        fontweight="bold", pad=28,
    )
    ax.text(
        0, 1.015, description, transform=ax.transAxes, va="bottom", ha="left",
        fontsize=8.5, color=MUTED, wrap=True,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)


def _format_percent_axis(ax, axis="y"):
    from matplotlib.ticker import PercentFormatter

    formatter = PercentFormatter(1.0, decimals=0)
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def main():
    d = _datasets()
    headline = d["headline"].iloc[0]
    fig = plt.figure(figsize=(18, 70), facecolor=PAPER)
    grid = fig.add_gridspec(
        12, 2,
        height_ratios=[1.0, 1.0, 1.0, 1.1, 1.2, 1.1, 1.05, 1.0, 1.25, 1.0, 1.0, 0.9],
        hspace=0.62, wspace=0.22,
    )
    fig.subplots_adjust(top=0.965, bottom=0.025, left=0.07, right=0.965)
    fig.suptitle(
        "Isfahan Intersection Traffic — Full Dashboard Preview",
        x=0.06, y=0.997, ha="left", fontsize=24, color=INK, fontweight="bold",
    )
    fig.text(
        0.06, 0.991,
        (
            "October 2018–January 2021  |  Paired-site comparisons  |  "
            "Entries near the stop line, not exits or turning paths\n"
            "Headline: Feb–Mar 2020 %.1f%% (%d sites)  •  Apr–May %.1f%% (%d sites)  •  "
            "Jan 2021 vs Jan 2020 %.1f%% (%d sites)"
        )
        % (
            100 * headline["shock_change"], int(headline["shock_sites"]),
            100 * headline["recovery_change"], int(headline["recovery_sites"]),
            100 * headline["latest_yoy_change"], int(headline["latest_yoy_sites"]),
        ),
        ha="left", va="top", fontsize=10.5, color=MUTED,
    )

    ax = fig.add_subplot(grid[0, :])
    _panel(
        ax, 1, "Network traffic by month",
        "Median trusted intersection volume. November 2020 is missing and December 2020 is partial.",
    )
    city = d["city_monthly"].copy()
    city["month_date"] = pd.to_datetime(city["month_date"])
    ax.plot(
        city["month_date"].to_numpy(), city["median_intersection_volume"].to_numpy(dtype=float),
        color=BLUE, linewidth=2.2, marker="o", markersize=3,
    )
    ax.fill_between(
        city["month_date"].to_numpy(),
        city["p25_intersection_volume"].to_numpy(dtype=float),
        city["p75_intersection_volume"].to_numpy(dtype=float),
        color=BLUE_LIGHT, alpha=0.24, label="25th–75th percentile",
    )
    ax.set_ylabel("Vehicles / 15 min", color=MUTED)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(frameon=False, loc="lower left", fontsize=8)

    ax = fig.add_subplot(grid[1, 0])
    _panel(ax, 2, "Paired monthly change", "Median change for the same trusted intersections in adjacent months.")
    changes = d["paired_changes"].copy()
    colors = np.where(changes["median_change"] >= 0, BLUE, ORANGE)
    ax.bar(np.arange(len(changes)), changes["median_change"], color=colors, width=0.78)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(np.arange(len(changes))[::2])
    ax.set_xticklabels(changes["year_month"].iloc[::2], rotation=55, ha="right")
    _format_percent_axis(ax)

    ax = fig.add_subplot(grid[1, 1])
    _panel(ax, 3, "2019 month-of-year pattern", "Same 34 intersections in all twelve months; this is the clean pre-pandemic seasonal view.")
    season = d["seasonality"].copy()
    ax.bar(season["month_name"], season["median_volume"], color=BLUE)
    ax.set_ylabel("Vehicles / 15 min", color=MUTED)
    ax.set_ylim(0, season["median_volume"].max() * 1.18)

    ax = fig.add_subplot(grid[2, :])
    _panel(ax, 4, "Daily traffic index", "Seven-day median. A value of 100 matches the same month-and-weekday pattern in 2019.")
    daily = d["daily_trend"].copy()
    daily["date_label"] = pd.to_datetime(daily["date_label"])
    ax.plot(
        daily["date_label"].to_numpy(), daily["volume_index_7d"].to_numpy(dtype=float),
        color=BLUE, linewidth=1.5,
    )
    ax.axhline(100, color=INK, linewidth=0.9, linestyle="--", label="2019 baseline")
    ax.set_ylabel("Traffic index", color=MUTED)
    ax.legend(frameon=False, loc="lower left", fontsize=8)

    ax = fig.add_subplot(grid[3, 0])
    _panel(ax, 5, "Context-day differences", "Index-point difference after controlling for Gregorian month and weekday; association, not cause.")
    effects = d["context_effects"].sort_values("adjusted_index_difference_points")
    colors = [ORANGE if value < 0 else BLUE for value in effects["adjusted_index_difference_points"]]
    ax.barh(effects["factor_label"], effects["adjusted_index_difference_points"], color=colors)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("Adjusted index-point difference", color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.6)

    ax = fig.add_subplot(grid[3, 1])
    _panel(ax, 6, "Intersection shock and recovery", "Each point is one site: x = Feb–Mar 2020; y = Apr–May 2020.")
    sr = d["shock_recovery"]
    for community, group in sr.groupby("community_label"):
        ax.scatter(
            group["shock_change"].to_numpy(dtype=float),
            group["recovery_change"].to_numpy(dtype=float),
            s=np.clip(group["mean_volume"].to_numpy(dtype=float) / 5, 24, 190), alpha=0.78,
            color=COMMUNITY_COLORS.get(community, MUTED), edgecolor="white", linewidth=0.5,
            label=community,
        )
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.axvline(0, color=INK, linewidth=0.7)
    ax.set_xlabel("Feb–Mar change", color=MUTED)
    ax.set_ylabel("Apr–May change", color=MUTED)
    _format_percent_axis(ax, "x")
    _format_percent_axis(ax, "y")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = fig.add_subplot(grid[4, 0])
    _panel(ax, 7, "Largest February–March falls", "Twelve intersections with the lowest paired change.")
    shock = d["shock_rank"].sort_values("shock_change")
    labels = ["SCATS %d" % int(value) for value in shock["intersection_id"]]
    ax.barh(labels, shock["shock_change"], color=ORANGE)
    ax.invert_yaxis()
    _format_percent_axis(ax, "x")
    ax.grid(axis="x", color=GRID, linewidth=0.6)

    ax = fig.add_subplot(grid[4, 1])
    _panel(ax, 8, "January 2021 year-over-year extremes", "Eight largest falls and rises among sites trusted in both January months.")
    latest = d["latest_extremes"].sort_values("jan_2021_vs_2020")
    labels = ["SCATS %d" % int(value) for value in latest["intersection_id"]]
    colors = [ORANGE if value < 0 else BLUE for value in latest["jan_2021_vs_2020"]]
    ax.barh(labels, latest["jan_2021_vs_2020"], color=colors)
    ax.axvline(0, color=INK, linewidth=0.8)
    _format_percent_axis(ax, "x")
    ax.grid(axis="x", color=GRID, linewidth=0.6)

    ax = fig.add_subplot(grid[5, :])
    _panel(ax, 9, "High-volume intersection heatmap", "Each row is indexed to that site's 2019 median; 100 means its own baseline.")
    heat = d["month_heatmap"].set_index("year_month")
    matrix = heat.to_numpy(dtype=float).T
    norm = TwoSlopeNorm(vmin=np.nanpercentile(matrix, 4), vcenter=100, vmax=np.nanpercentile(matrix, 96))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_yticks(range(len(heat.columns)))
    ax.set_yticklabels([name.replace("i", "SCATS ") for name in heat.columns])
    ax.set_xticks(range(0, len(heat.index), 2))
    ax.set_xticklabels(heat.index[::2], rotation=45, ha="right")
    fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01, label="Traffic index")
    ax.grid(False)

    ax = fig.add_subplot(grid[6, :])
    _panel(ax, 10, "Weekly traffic pattern by hour", "Network mean volume. Friday is Iran's main weekend day.")
    week = d["hour_week_heatmap"].set_index("weekday_name")
    hour_columns = ["h%02d" % hour for hour in range(24)]
    image = ax.imshow(week[hour_columns].to_numpy(dtype=float), aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(week.index)))
    ax.set_yticklabels(week.index)
    ax.set_xticks(range(24))
    ax.set_xticklabels(["%02d" % hour for hour in range(24)])
    ax.set_xlabel("Hour", color=MUTED)
    fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01, label="Vehicles / 15 min")
    ax.grid(False)

    ax = fig.add_subplot(grid[7, :])
    _panel(ax, 11, "Intersection 1081 inbound approaches", "Entry-side detector volume by direction. These are not exit or turning counts.")
    approaches = d["approaches_1081"].copy()
    approaches["year_month"] = pd.to_datetime(approaches["year_month"] + "-01")
    direction_colors = {"South": BLUE, "West": ORANGE, "North": OLIVE, "East": PINK}
    for direction, group in approaches.groupby("approach"):
        ax.plot(
            group["year_month"].to_numpy(), group["mean_volume"].to_numpy(dtype=float),
            label=direction, linewidth=2, color=direction_colors[direction],
        )
    ax.set_ylabel("Vehicles / 15 min", color=MUTED)
    ax.legend(frameon=False, ncol=4, loc="upper center")

    ax = fig.add_subplot(grid[8, :])
    _panel(ax, 12, "Inbound direction mix at busy intersections", "Each bar totals 100%; only sites with four usable direction groups are included.")
    shares = d["approach_shares"].sort_values("mean_volume")
    y = np.arange(len(shares))
    left = np.zeros(len(shares))
    for field, label, color in [
        ("south_share", "South", BLUE), ("west_share", "West", ORANGE),
        ("north_share", "North", OLIVE), ("east_share", "East", PINK),
    ]:
        values = shares[field].to_numpy(dtype=float)
        ax.barh(y, values, left=left, label=label, color=color)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(["SCATS %d" % int(value) for value in shares["intersection_id"]])
    ax.set_xlim(0, 1)
    _format_percent_axis(ax, "x")
    ax.legend(frameon=False, ncol=4, loc="lower center")
    ax.grid(axis="x", color=GRID, linewidth=0.6)

    ax = fig.add_subplot(grid[9, 0])
    _panel(ax, 13, "Traffic volume and functional centrality", "Centrality means strong links to similar weekly profiles, not physical road centrality.")
    nodes = d["community_nodes"]
    for community, group in nodes.groupby("community_label"):
        ax.scatter(
            group["mean_volume"].to_numpy(dtype=float),
            group["weighted_degree"].to_numpy(dtype=float),
            s=np.clip(group["trusted_months"].to_numpy(dtype=float) * 4, 18, 120), alpha=0.75,
            color=COMMUNITY_COLORS.get(community, MUTED), label=community,
            edgecolor="white", linewidth=0.5,
        )
    ax.set_xlabel("Mean vehicles / 15 min", color=MUTED)
    ax.set_ylabel("Weighted degree", color=MUTED)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = fig.add_subplot(grid[9, 1])
    _panel(ax, 14, "Functional community sizes", "These are traffic-shape peer groups, not city districts.")
    communities = d["community_summary"].sort_values("community")
    ax.bar(
        communities["community_label"], communities["intersections"],
        color=[COMMUNITY_COLORS.get(value, BLUE) for value in communities["community_label"]],
    )
    ax.set_ylabel("Intersections", color=MUTED)

    ax = fig.add_subplot(grid[10, :])
    _panel(ax, 15, "Mapped intersection candidates", "Medium- and high-confidence OpenStreetMap candidates; municipal GIS review is still required.")
    points = d["map_points"]
    for community, group in points.groupby("community_label"):
        ax.scatter(
            group["longitude"].to_numpy(dtype=float), group["latitude"].to_numpy(dtype=float),
            s=np.clip(group["mean_volume"].to_numpy(dtype=float) / 2.5, 35, 300), alpha=0.72,
            color=COMMUNITY_COLORS.get(community, MUTED), label=community,
            edgecolor="white", linewidth=0.6,
        )
        for row in group.itertuples():
            ax.annotate(str(int(row.intersection_id)), (row.longitude, row.latitude), fontsize=6, color=INK, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Longitude", color=MUTED)
    ax.set_ylabel("Latitude", color=MUTED)
    ax.legend(frameon=False, ncol=4, fontsize=7)

    ax = fig.add_subplot(grid[11, 0])
    _panel(ax, 16, "Source-month temporal coverage", "Observed 15-minute timestamps divided by expected timestamps.")
    quality = d["source_quality"].copy()
    qcolors = [ORANGE if status != "Complete" else BLUE for status in quality["status"]]
    ax.bar(np.arange(len(quality)), quality["source_temporal_coverage"], color=qcolors)
    ax.set_xticks(np.arange(len(quality))[::2])
    ax.set_xticklabels(quality["source_month"].iloc[::2], rotation=55, ha="right")
    ax.set_ylim(0, 1.08)
    _format_percent_axis(ax)

    ax = fig.add_subplot(grid[11, 1])
    _panel(ax, 17, "Trusted-month coverage by intersection", "More trusted months give a stronger long-term trend.")
    trusted = d["trusted_months"]["trusted_months"].dropna()
    bins = np.arange(trusted.min() - 0.5, trusted.max() + 1.5, 2)
    ax.hist(trusted, bins=bins, color=BLUE, edgecolor="white")
    ax.set_xlabel("Trusted months", color=MUTED)
    ax.set_ylabel("Intersections", color=MUTED)

    fig.text(
        0.06, 0.002,
        (
            "Limits: November 2020 is missing; December 2020 is partial; 20.2% of profiled channel readings are unavailable. "
            "Map points need GIS review. Calendar, weather, event, and COVID comparisons are associations."
        ),
        fontsize=9, color=MUTED, ha="left", va="bottom",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, facecolor=PAPER, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("Dashboard image written to %s" % OUTPUT)


if __name__ == "__main__":
    main()
