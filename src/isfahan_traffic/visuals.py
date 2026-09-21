from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px

from .paths import PROJECT_ROOT, ensure_project_directories, load_settings
from .web_navigation import add_map_navigation


BLUE = "#2563EB"
GOLD = "#D4A017"
ORANGE = "#E67E22"
OLIVE = "#708238"
PINK = "#C05A8A"
INK = "#1F2937"
GRID = "#E5E7EB"


def _style_axis(axis, title: str, subtitle: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=14, color=INK, pad=22)
    axis.text(0, 1.02, subtitle, transform=axis.transAxes, fontsize=9, color="#6B7280", va="bottom")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_city_monthly(table_dir: Path, figure_dir: Path) -> None:
    data = pd.read_csv(table_dir / "city_monthly.csv")
    data["date"] = pd.to_datetime(data["year_month"] + "-01")
    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.fill_between(
        data["date"].to_numpy(),
        data["p25_intersection_volume"].to_numpy(dtype=float),
        data["p75_intersection_volume"].to_numpy(dtype=float),
        color=BLUE,
        alpha=0.14,
    )
    axis.plot(
        data["date"].to_numpy(),
        data["median_intersection_volume"].to_numpy(dtype=float),
        color=BLUE,
        linewidth=2.4,
        marker="o",
        markersize=3,
    )
    _style_axis(
        axis,
        "Citywide monthly traffic volume",
        "Median and interquartile range of trusted intersection-months; Gregorian calendar",
        "Month",
        "Coverage-normalized vehicles per 15 minutes",
    )
    _save(fig, figure_dir / "city_monthly_volume.png")


def plot_month_pattern(table_dir: Path, figure_dir: Path) -> None:
    data = pd.read_csv(table_dir / "month_of_year_profile.csv")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, axis = plt.subplots(figsize=(10, 5.2))
    axis.bar(
        data["month"].to_numpy(dtype=int),
        data["median_volume"].to_numpy(dtype=float),
        color=GOLD,
        edgecolor="#7C5A00",
        linewidth=0.7,
    )
    axis.set_xticks(range(1, 13))
    axis.set_xticklabels(month_labels)
    _style_axis(
        axis,
        "Typical traffic volume by month of year",
        "Median across trusted intersection-months in 2019, the only complete pre-pandemic calendar year",
        "Gregorian month",
        "Coverage-normalized vehicles per 15 minutes",
    )
    _save(fig, figure_dir / "month_of_year_pattern.png")


def plot_1081_hourly(table_dir: Path, figure_dir: Path) -> None:
    data = pd.read_csv(table_dir / "hourly_weekday_profile.csv")
    data = data[data["intersection_id"] == 1081]
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    colors = plt.cm.Blues(np.linspace(0.35, 0.95, 7))
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for weekday, color in zip(range(7), colors):
        subset = data[data["weekday"] == weekday].sort_values("hour")
        axis.plot(
            subset["hour"].to_numpy(dtype=int),
            subset["mean_volume"].to_numpy(dtype=float),
            color=color,
            linewidth=1.8,
            label=weekday_names[weekday],
        )
    axis.legend(ncol=4, frameon=False, fontsize=8, loc="upper left")
    axis.set_xticks(range(0, 24, 2))
    _style_axis(
        axis,
        "Intersection 1081 hourly profile by weekday",
        "Mean coverage-normalized entry volume across all available 15-minute observations",
        "Hour of day (local time)",
        "Vehicles per 15 minutes",
    )
    _save(fig, figure_dir / "intersection_1081_hourly_weekday.png")


def plot_1081_approaches(table_dir: Path, figure_dir: Path) -> None:
    data = pd.read_csv(table_dir / "intersection_1081_approach_monthly.csv")
    data["date"] = pd.to_datetime(data["year_month"] + "-01")
    colors = {"south": BLUE, "west": GOLD, "north": ORANGE, "east": OLIVE}
    fig, axis = plt.subplots(figsize=(11, 5.5))
    for approach, color in colors.items():
        axis.plot(
            data["date"].to_numpy(),
            data[approach + "_mean"].to_numpy(dtype=float),
            label=approach.title(),
            color=color,
            linewidth=2,
        )
    axis.legend(frameon=False, ncol=4, loc="upper left")
    _style_axis(
        axis,
        "Intersection 1081 entry volume by approach",
        "South 1-3, west 4-6, north 7-9, east 10-12; these are approach entries, not exits",
        "Month",
        "Vehicles per 15 minutes",
    )
    _save(fig, figure_dir / "intersection_1081_approaches.png")


def plot_functional_graph(table_dir: Path, figure_dir: Path) -> None:
    nodes = pd.read_csv(table_dir / "functional_graph_nodes.csv", encoding="utf-8-sig")
    edges = pd.read_csv(table_dir / "functional_graph_edges.csv")
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(int(row.intersection_id), community=int(row.community), weighted_degree=float(row.weighted_degree))
    for row in edges.itertuples(index=False):
        graph.add_edge(int(row.source), int(row.target), weight=float(row.profile_correlation))
    #the fixed seed keeps the layout stable between runs
    #these positions are for the chart and are not map coordinates
    positions = nx.spring_layout(graph, 0.45, None, None, 50, 1e-4, "weight", 1, None, 2, 42)
    communities = [graph.nodes[node]["community"] for node in graph.nodes]
    degrees = np.array([graph.nodes[node]["weighted_degree"] for node in graph.nodes], dtype=float)
    sizes = 80 + 520 * (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-9)
    widths = [0.4 + 1.8 * graph.edges[edge]["weight"] for edge in graph.edges]
    fig, axis = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(graph, positions, ax=axis, alpha=0.18, edge_color="#64748B", width=widths)
    nx.draw_networkx_nodes(graph, positions, ax=axis, node_color=communities, cmap=plt.cm.tab10, node_size=sizes, edgecolors="white", linewidths=0.7)
    top_nodes = nodes.nlargest(10, "weighted_degree")["intersection_id"].astype(int).tolist()
    nx.draw_networkx_labels(graph, positions, labels={node: str(node) for node in top_nodes}, ax=axis, font_size=8, font_color=INK)
    axis.text(
        0,
        1.055,
        "Functional similarity graph of Isfahan intersections",
        transform=axis.transAxes,
        fontsize=14,
        color=INK,
        va="bottom",
    )
    axis.text(
        0,
        1.025,
        "Edges connect similar 168-bin weekly traffic profiles; layout is not geographic",
        transform=axis.transAxes,
        fontsize=9,
        color="#6B7280",
        va="bottom",
    )
    axis.axis("off")
    _save(fig, figure_dir / "functional_similarity_graph.png")


def build_maps(table_dir: Path, figure_dir: Path) -> Dict[str, object]:
    geocoded_path = PROJECT_ROOT / "data" / "metadata" / "intersections_geocoded.csv"
    if not geocoded_path.exists():
        return {"located": 0, "eligible": 0, "note": "Geocoding output not available."}
    geocoded = pd.read_csv(geocoded_path, encoding="utf-8-sig")
    nodes = pd.read_csv(table_dir / "functional_graph_nodes.csv", encoding="utf-8-sig")
    join_columns = ["intersection_id", "latitude", "longitude", "geocode_method", "geocode_confidence"]
    if "name_fa" in geocoded:
        join_columns.append("name_fa")
    data = nodes.merge(geocoded[join_columns], how="left", on="intersection_id", suffixes=("", "_geo"))
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    reliable = data["geocode_method"].eq("direct_name") | (
        data["geocode_method"].eq("street_pair") & data["geocode_confidence"].eq("high")
    )
    located = data[reliable].dropna(subset=["latitude", "longitude"]).copy()
    if located.empty:
        return {"located": 0, "eligible": len(data), "note": "No intersections geocoded inside the Isfahan bounding box."}
    located["label"] = located["intersection_id"].astype(str)
    located["name_fa"] = located.get("name_fa", "").fillna("").astype(str)
    located["community"] = located["community"].fillna(0).astype(int).astype(str)
    located["mean_volume"] = located["mean_volume"].fillna(0)

    fig, axis = plt.subplots(figsize=(9, 8))
    scatter = axis.scatter(
        located["longitude"].to_numpy(dtype=float),
        located["latitude"].to_numpy(dtype=float),
        c=located["community"].astype(int),
        s=45 + 180 * located["mean_volume"] / max(located["mean_volume"].max(), 1),
        cmap=plt.cm.tab10,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.7,
    )
    for row in located.nlargest(min(12, len(located)), "weighted_degree").itertuples(index=False):
        axis.text(row.longitude, row.latitude, str(row.intersection_id), fontsize=7, color=INK)
    _style_axis(
        axis,
        "Located SCATS intersections in Isfahan",
        "Point size is mean traffic volume; color is functional community; coordinates require GIS review",
        "Longitude",
        "Latitude",
    )
    _save(fig, figure_dir / "isfahan_intersections_map.png")

    interactive = px.scatter_mapbox(
        located,
        lat="latitude",
        lon="longitude",
        size="mean_volume",
        color="community",
        hover_name="label",
        hover_data={
            "name_fa": True,
            "mean_volume": ":.1f",
            "dtw_cluster": True,
            "geocode_method": True,
            "geocode_confidence": True,
            "latitude": ":.5f",
            "longitude": ":.5f",
        },
        zoom=10.5,
        center={"lat": float(located["latitude"].median()), "lon": float(located["longitude"].median())},
        mapbox_style="open-street-map",
        title="Isfahan SCATS intersections: volume and functional community",
        size_max=24,
    )
    interactive.update_layout(margin={"r": 0, "t": 45, "l": 0, "b": 0})
    #include Plotly so the file opens without a local server
    #the background map still needs internet access
    interactive.write_html(
        str(PROJECT_ROOT / "outputs" / "isfahan_intersections_map.html"),
        include_plotlyjs=True,
        full_html=True,
    )
    add_map_navigation(PROJECT_ROOT / "outputs" / "isfahan_intersections_map.html")
    return {
        "located": int(len(located)),
        "eligible": int(len(data)),
        "coverage": float(len(located) / len(data)) if len(data) else 0.0,
        "note": "OpenStreetMap/Nominatim coordinates are cached candidates and should be verified in municipal GIS or QGIS.",
    }


def build_visuals() -> None:
    ensure_project_directories()
    table_dir = PROJECT_ROOT / "outputs" / "tables"
    figure_dir = PROJECT_ROOT / "outputs" / "figures"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    plot_city_monthly(table_dir, figure_dir)
    plot_month_pattern(table_dir, figure_dir)
    plot_1081_hourly(table_dir, figure_dir)
    plot_1081_approaches(table_dir, figure_dir)
    plot_functional_graph(table_dir, figure_dir)
    map_summary = build_maps(table_dir, figure_dir)
    (table_dir / "map_summary.json").write_text(json.dumps(map_summary, indent=2), encoding="utf-8")
    print("[visualize] wrote figures and map outputs", flush=True)
