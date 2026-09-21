from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .paths import PROJECT_ROOT, ensure_project_directories, load_settings


APPROACHES = ("south", "west", "north", "east")


def _read_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip", parse_dates=["timestamp"])
    frame["date"] = frame["timestamp"].dt.floor("D")
    frame["weekday"] = frame["timestamp"].dt.dayofweek
    frame["hour"] = frame["timestamp"].dt.hour
    frame["month"] = frame["timestamp"].dt.month
    frame["year"] = frame["timestamp"].dt.year
    frame["year_month"] = frame["timestamp"].dt.strftime("%Y-%m")
    return frame


def _monthly_intersection(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["volume_normalized"].notna()].copy()
    grouped = valid.groupby(["year_month", "intersection_id"], as_index=False).agg(
        valid_observations=("volume_normalized", "size"),
        mean_15min_volume=("volume_normalized", "mean"),
        median_15min_volume=("volume_normalized", "median"),
        total_volume_estimate=("volume_normalized", "sum"),
        mean_detector_coverage=("detector_coverage", "mean"),
    )
    all_counts = frame.groupby(["year_month", "intersection_id"], as_index=False).agg(
        source_observations=("timestamp", "size")
    )
    grouped = all_counts.merge(grouped, how="left", on=["year_month", "intersection_id"])
    timestamps = frame[["year_month", "timestamp"]].drop_duplicates()
    expected = (
        timestamps.groupby("year_month")["timestamp"]
        .agg(lambda values: values.dt.days_in_month.iloc[0] * 96)
        .rename("expected_full_month_observations")
        .reset_index()
    )
    grouped = grouped.merge(expected, on="year_month", how="left")
    grouped["temporal_coverage"] = (
        grouped["valid_observations"] / grouped["expected_full_month_observations"]
    )
    return grouped


def _daily_intersection(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["volume_normalized"].notna()].copy()
    daily = valid.groupby(["date", "intersection_id"], as_index=False).agg(
        valid_observations=("volume_normalized", "size"),
        total_volume_estimate=("volume_normalized", "sum"),
        mean_15min_volume=("volume_normalized", "mean"),
        mean_detector_coverage=("detector_coverage", "mean"),
    )
    daily["temporal_coverage"] = daily["valid_observations"] / 96.0
    return daily


def _hourly_weekday_sums(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["volume_normalized"].notna()]
    return valid.groupby(["intersection_id", "weekday", "hour"], as_index=False).agg(
        volume_sum=("volume_normalized", "sum"),
        observations=("volume_normalized", "size"),
    )


def _approach_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations = {}
    for approach in APPROACHES:
        aggregations[approach + "_mean"] = (approach + "_volume", "mean")
        aggregations[approach + "_observations"] = (approach + "_volume", "count")
        aggregations[approach + "_coverage"] = (approach + "_coverage", "mean")
    return frame.groupby(["year_month", "intersection_id"], as_index=False).agg(**aggregations)


def _dtw_distance(left: np.ndarray, right: np.ndarray, window: int = 3) -> float:
    #allow small shifts in peak timing when comparing daily patterns
    #the narrow window limits how far the peaks can move
    length_left = len(left)
    length_right = len(right)
    band = max(window, abs(length_left - length_right))
    matrix = np.full((length_left + 1, length_right + 1), np.inf)
    matrix[0, 0] = 0.0
    for i in range(1, length_left + 1):
        for j in range(max(1, i - band), min(length_right, i + band) + 1):
            cost = abs(float(left[i - 1]) - float(right[j - 1]))
            matrix[i, j] = cost + min(matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1])
    return float(matrix[length_left, length_right] / (length_left + length_right))


def _fit_precomputed_clusters(distance_matrix: np.ndarray, clusters: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(
            n_clusters=clusters, metric="precomputed", linkage="average"
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=clusters, affinity="precomputed", linkage="average"
        )
    return model.fit_predict(distance_matrix)


def build_dtw_clusters(hourly: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    daily_shape = hourly.groupby(["intersection_id", "hour"], as_index=False).agg(
        volume_sum=("volume_sum", "sum"), observations=("observations", "sum")
    )
    daily_shape["mean_volume"] = daily_shape["volume_sum"] / daily_shape["observations"]
    pivot = daily_shape.pivot(index="intersection_id", columns="hour", values="mean_volume")
    pivot = pivot[pivot.notna().sum(axis=1) >= 20]
    #fill missing hours from the same junction
    #standardize each profile to compare its shape rather than its volume
    pivot = pivot.apply(lambda row: row.fillna(row.median()), axis=1)
    values = pivot.to_numpy(dtype=float)
    row_means = values.mean(axis=1, keepdims=True)
    row_stds = values.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1.0
    standardized = (values - row_means) / row_stds
    count = len(pivot)
    distances = np.zeros((count, count), dtype=float)
    for left in range(count):
        for right in range(left + 1, count):
            distance = _dtw_distance(standardized[left], standardized[right])
            distances[left, right] = distance
            distances[right, left] = distance

    candidates = []
    best_labels = np.zeros(count, dtype=int)
    best_score = -1.0
    best_clusters = 1
    for clusters in range(2, min(6, count - 1) + 1):
        labels = _fit_precomputed_clusters(distances, clusters)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(distances, labels, metric="precomputed"))
        candidates.append({"clusters": clusters, "silhouette": score})
        if score > best_score:
            best_score = score
            best_labels = labels
            best_clusters = clusters

    output = pd.DataFrame(
        {
            "intersection_id": pivot.index.astype(int),
            "dtw_cluster": best_labels.astype(int) + 1,
        }
    )
    metadata = {
        "selected_clusters": int(best_clusters),
        "silhouette": best_score,
        "candidates": candidates,
        "intersections_clustered": int(count),
    }
    return output, metadata


def build_functional_graph(
    hourly: pd.DataFrame, neighbors: int, minimum_correlation: float
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    profile = hourly.copy()
    profile["bin"] = profile["weekday"] * 24 + profile["hour"]
    pivot = profile.pivot(index="intersection_id", columns="bin", values="mean_volume")
    pivot = pivot[pivot.notna().sum(axis=1) >= 126]
    pivot = pivot.apply(lambda row: row.fillna(row.median()), axis=1)
    values = pivot.to_numpy(dtype=float)
    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, keepdims=True)
    keep = stds[:, 0] > 0
    values = values[keep]
    node_ids = pivot.index.to_numpy(dtype=int)[keep]
    standardized = (values - means[keep]) / stds[keep]
    correlations = np.corrcoef(standardized)

    edge_weights: Dict[Tuple[int, int], float] = {}
    for index, intersection_id in enumerate(node_ids):
        ordering = np.argsort(correlations[index])[::-1]
        added = 0
        for candidate_index in ordering:
            if candidate_index == index:
                continue
            correlation = float(correlations[index, candidate_index])
            if correlation < minimum_correlation:
                continue
            other_id = int(node_ids[candidate_index])
            key = tuple(sorted((int(intersection_id), other_id)))
            edge_weights[key] = max(edge_weights.get(key, -1.0), correlation)
            added += 1
            if added >= neighbors:
                break

    graph = nx.Graph()
    graph.add_nodes_from(int(value) for value in node_ids)
    for (source, target), correlation in edge_weights.items():
        graph.add_edge(source, target, weight=correlation, distance=max(1e-6, 1.0 - correlation))

    if graph.number_of_edges() > 0:
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
    else:
        communities = [frozenset([node]) for node in graph.nodes]
    community_by_node = {}
    for community_id, community in enumerate(communities, start=1):
        for node in community:
            community_by_node[int(node)] = community_id

    degree = nx.degree_centrality(graph)
    #pass the seed by position to avoid a decorator compatibility issue
    #older NetworkX versions can fail when the seed is passed by name
    betweenness = nx.betweenness_centrality(graph, None, True, "distance", False, 42)
    weighted_degree = dict(graph.degree(weight="weight"))
    try:
        eigenvector = nx.eigenvector_centrality(graph, weight="weight", max_iter=2000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {node: np.nan for node in graph.nodes}

    nodes = pd.DataFrame(
        [
            {
                "intersection_id": int(node),
                "community": community_by_node.get(int(node)),
                "degree_centrality": degree.get(node, 0.0),
                "weighted_degree": weighted_degree.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "eigenvector_centrality": eigenvector.get(node, np.nan),
            }
            for node in graph.nodes
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "source": int(source),
                "target": int(target),
                "profile_correlation": data["weight"],
                "profile_distance": data["distance"],
            }
            for source, target, data in graph.edges(data=True)
        ]
    )
    metadata = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "communities": len(communities),
        "connected_components": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
        "interpretation": "Edges represent similar 168-bin weekly traffic profiles, not physical roads.",
    }
    return nodes, edges, metadata


def _consecutive_month_changes(city_monthly: pd.DataFrame) -> pd.DataFrame:
    frame = city_monthly.sort_values("year_month").copy()
    periods = pd.PeriodIndex(frame["year_month"], freq="M")
    frame["previous_period"] = (periods - 1).astype(str)
    frame["is_consecutive"] = frame["previous_period"].eq(frame["year_month"].shift(1))
    frame["mom_change"] = frame["median_intersection_volume"].pct_change()
    return frame[frame["is_consecutive"] & frame["mom_change"].notna()]


def _paired_month_changes(trusted: pd.DataFrame) -> pd.DataFrame:
    #compare the same junctions in both months so changes in site coverage
    #do not appear as changes in traffic
    periods = sorted(pd.PeriodIndex(trusted["year_month"].unique(), freq="M"))
    rows = []
    for current_period in periods:
        previous_period = current_period - 1
        if previous_period not in periods:
            continue
        previous = trusted[trusted["year_month"] == str(previous_period)][
            ["intersection_id", "mean_15min_volume"]
        ].rename(columns={"mean_15min_volume": "previous_volume"})
        current = trusted[trusted["year_month"] == str(current_period)][
            ["intersection_id", "mean_15min_volume"]
        ].rename(columns={"mean_15min_volume": "current_volume"})
        paired = previous.merge(current, on="intersection_id", how="inner")
        paired = paired[paired["previous_volume"] > 0]
        if paired.empty:
            continue
        paired["change"] = paired["current_volume"] / paired["previous_volume"] - 1.0
        rows.append(
            {
                "year_month": str(current_period),
                "previous_month": str(previous_period),
                "paired_intersections": int(len(paired)),
                "median_change": float(paired["change"].median()),
                "mean_change": float(paired["change"].mean()),
                "p25_change": float(paired["change"].quantile(0.25)),
                "p75_change": float(paired["change"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def analyze_all() -> None:
    ensure_project_directories()
    settings = load_settings()
    partition_dir = PROJECT_ROOT / "data" / "processed" / "15min"
    partitions = sorted(partition_dir.glob("*.csv.gz"))
    if not partitions:
        raise FileNotFoundError("No processed partitions found. Run the ingest stage first.")

    monthly_parts = []
    daily_parts = []
    hourly_parts = []
    approach_parts = []
    for index, path in enumerate(partitions, start=1):
        print("[analyze] {0}/{1}: {2}".format(index, len(partitions), path.name), flush=True)
        frame = _read_partition(path)
        #some exports include a closing timestamp from the next month
        #keep only records for the month named in the file
        frame = frame[frame["year_month"] == path.name[:7]].copy()
        monthly_parts.append(_monthly_intersection(frame))
        daily_parts.append(_daily_intersection(frame))
        hourly_parts.append(_hourly_weekday_sums(frame))
        approach_parts.append(_approach_monthly(frame))

    monthly = pd.concat(monthly_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)
    hourly_sums = pd.concat(hourly_parts, ignore_index=True)
    hourly = hourly_sums.groupby(["intersection_id", "weekday", "hour"], as_index=False).agg(
        volume_sum=("volume_sum", "sum"), observations=("observations", "sum")
    )
    hourly["mean_volume"] = hourly["volume_sum"] / hourly["observations"]
    approach_monthly = pd.concat(approach_parts, ignore_index=True)

    trusted = monthly[
        (monthly["temporal_coverage"] >= float(settings["trusted_temporal_coverage"]))
        & (monthly["mean_detector_coverage"] >= float(settings["trusted_detector_coverage"]))
    ].copy()
    city_monthly = trusted.groupby("year_month", as_index=False).agg(
        median_intersection_volume=("mean_15min_volume", "median"),
        mean_intersection_volume=("mean_15min_volume", "mean"),
        p25_intersection_volume=("mean_15min_volume", lambda values: values.quantile(0.25)),
        p75_intersection_volume=("mean_15min_volume", lambda values: values.quantile(0.75)),
        intersections=("intersection_id", "nunique"),
    )
    trusted["month"] = trusted["year_month"].str[-2:].astype(int)
    trusted["year"] = trusted["year_month"].str[:4].astype(int)
    #use the same sites throughout 2019 to keep changes in site coverage
    #out of the seasonal baseline for the only complete year before the pandemic
    baseline_2019 = trusted[trusted["year"] == 2019].copy()
    common_2019_ids = (
        baseline_2019.groupby("intersection_id")["month"].nunique().loc[lambda values: values == 12].index
    )
    baseline_2019 = baseline_2019[baseline_2019["intersection_id"].isin(common_2019_ids)]
    month_of_year = baseline_2019.groupby("month", as_index=False).agg(
        mean_volume=("mean_15min_volume", "mean"),
        median_volume=("mean_15min_volume", "median"),
        standard_deviation=("mean_15min_volume", "std"),
        intersection_months=("intersection_id", "size"),
    )
    month_year = trusted.groupby(["year", "month"], as_index=False).agg(
        median_volume=("mean_15min_volume", "median"),
        intersections=("intersection_id", "nunique"),
    )

    dtw_clusters, dtw_metadata = build_dtw_clusters(hourly)
    graph_nodes, graph_edges, graph_metadata = build_functional_graph(
        hourly,
        int(settings["functional_graph_neighbors"]),
        float(settings["functional_graph_min_correlation"]),
    )
    graph_nodes = graph_nodes.merge(dtw_clusters, how="left", on="intersection_id")
    volume_summary = trusted.groupby("intersection_id", as_index=False).agg(
        mean_volume=("mean_15min_volume", "mean"),
        trusted_months=("year_month", "nunique"),
    )
    graph_nodes = graph_nodes.merge(volume_summary, how="left", on="intersection_id")

    metadata_path = PROJECT_ROOT / "data" / "metadata" / "intersections.csv"
    intersection_metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    name_columns = [column for column in ["intersection_id", "name_fa", "gis_id"] if column in intersection_metadata]
    monthly = monthly.merge(intersection_metadata[name_columns], how="left", on="intersection_id")
    graph_nodes = graph_nodes.merge(intersection_metadata[name_columns], how="left", on="intersection_id")

    table_dir = PROJECT_ROOT / "outputs" / "tables"
    monthly.to_csv(table_dir / "intersection_monthly.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(table_dir / "intersection_daily.csv", index=False)
    hourly.to_csv(table_dir / "hourly_weekday_profile.csv", index=False)
    approach_monthly.to_csv(table_dir / "approach_monthly.csv", index=False)
    approach_monthly[approach_monthly["intersection_id"] == 1081].to_csv(
        table_dir / "intersection_1081_approach_monthly.csv", index=False
    )
    city_monthly.to_csv(table_dir / "city_monthly.csv", index=False)
    month_of_year.to_csv(table_dir / "month_of_year_profile.csv", index=False)
    month_year.to_csv(table_dir / "month_of_year_by_year.csv", index=False)
    graph_nodes.to_csv(table_dir / "functional_graph_nodes.csv", index=False, encoding="utf-8-sig")
    graph_edges.to_csv(table_dir / "functional_graph_edges.csv", index=False)
    dtw_clusters.to_csv(table_dir / "dtw_clusters.csv", index=False)
    paired_changes = _paired_month_changes(trusted)
    paired_changes.to_csv(table_dir / "paired_month_changes.csv", index=False)

    source_quality_path = table_dir / "source_month_quality.csv"
    source_quality = pd.read_csv(source_quality_path) if source_quality_path.exists() else pd.DataFrame()
    channel_quality = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "channel_quality.csv")
    if not source_quality.empty:
        observed_periods = pd.PeriodIndex(source_quality["source_month"], freq="M")
        complete_period_range = pd.period_range(observed_periods.min(), observed_periods.max(), freq="M")
        missing_source_months = [str(period) for period in complete_period_range if period not in set(observed_periods)]
    else:
        missing_source_months = []
    changes = _consecutive_month_changes(city_monthly)
    largest_decline = (
        paired_changes.loc[paired_changes["median_change"].idxmin()].to_dict()
        if not paired_changes.empty
        else {}
    )
    largest_increase = (
        paired_changes.loc[paired_changes["median_change"].idxmax()].to_dict()
        if not paired_changes.empty
        else {}
    )
    peak_pattern = month_of_year.loc[month_of_year["median_volume"].idxmax()].to_dict()
    low_pattern = month_of_year.loc[month_of_year["median_volume"].idxmin()].to_dict()
    central = (
        graph_nodes.sort_values("weighted_degree", ascending=False).iloc[0].to_dict()
        if not graph_nodes.empty
        else {}
    )
    volume_rank = volume_summary.sort_values("mean_volume", ascending=False).reset_index(drop=True)
    rank_1081 = (
        int(volume_rank.index[volume_rank["intersection_id"] == 1081][0]) + 1
        if 1081 in set(volume_rank["intersection_id"])
        else None
    )
    approach_1081 = approach_monthly[approach_monthly["intersection_id"] == 1081]
    approach_means = {
        approach: float(approach_1081[approach + "_mean"].mean())
        for approach in APPROACHES
        if not approach_1081.empty and approach_1081[approach + "_mean"].notna().any()
    }
    dominant_approach = max(approach_means, key=approach_means.get) if approach_means else None
    total_approach = sum(approach_means.values())
    graph_ranked = graph_nodes.sort_values("weighted_degree", ascending=False).reset_index(drop=True)
    graph_rank_1081 = (
        int(graph_ranked.index[graph_ranked["intersection_id"] == 1081][0]) + 1
        if 1081 in set(graph_ranked["intersection_id"])
        else None
    )
    monthly_1081 = monthly[monthly["intersection_id"] == 1081].set_index("year_month")
    def intersection_change(previous: str, current: str):
        if previous not in monthly_1081.index or current not in monthly_1081.index:
            return None
        previous_value = float(monthly_1081.loc[previous, "mean_15min_volume"])
        current_value = float(monthly_1081.loc[current, "mean_15min_volume"])
        return current_value / previous_value - 1.0 if previous_value else None

    operational_detectors = pd.read_csv(
        PROJECT_ROOT / "data" / "metadata" / "detector_channels.csv"
    )
    operational_detectors = operational_detectors[
        operational_detectors["operational"].astype(str).str.lower().isin(["true", "1"])
    ]

    summary = {
        "data_window": {
            "first_month": str(city_monthly["year_month"].min()),
            "last_month": str(city_monthly["year_month"].max()),
            "raw_intersection_records": int(source_quality["intersection_records"].sum()) if not source_quality.empty else None,
            "intersections_with_raw_data": int(monthly["intersection_id"].nunique()),
            "trusted_intersection_months": int(len(trusted)),
            "partial_source_months": source_quality.loc[
                source_quality["source_temporal_coverage"] < 0.8, "source_month"
            ].astype(str).tolist() if not source_quality.empty else [],
            "missing_source_months": missing_source_months,
        },
        "quality": {
            "channel_readings_profiled": int(channel_quality["observations"].sum()),
            "invalid_channel_readings": int(channel_quality["invalid"].sum()),
            "invalid_channel_rate": float(channel_quality["invalid"].sum() / channel_quality["observations"].sum()),
            "operational_detector_invalid_rate": float(
                operational_detectors["invalid"].sum() / operational_detectors["observations"].sum()
            ),
        },
        "time_series": {
            "largest_consecutive_month_decline": largest_decline,
            "largest_consecutive_month_increase": largest_increase,
            "highest_typical_month": peak_pattern,
            "lowest_typical_month": low_pattern,
            "month_of_year_baseline": "2019 (only complete pre-pandemic calendar year)",
            "month_of_year_common_panel_intersections": int(len(common_2019_ids)),
            "seasonal_peak_to_trough": float(
                peak_pattern["median_volume"] / low_pattern["median_volume"] - 1.0
            ),
        },
        "intersection_1081": {
            "volume_rank": rank_1081,
            "ranked_intersections": int(len(volume_rank)),
            "functional_similarity_centrality_rank": graph_rank_1081,
            "mean_approach_volumes": approach_means,
            "dominant_entry_approach": dominant_approach,
            "dominant_approach_share": approach_means.get(dominant_approach, 0.0) / total_approach if total_approach else None,
            "north_south_corridor_share": (
                (approach_means.get("north", 0.0) + approach_means.get("south", 0.0)) / total_approach
                if total_approach
                else None
            ),
            "feb_to_mar_2020_change": intersection_change("2020-02", "2020-03"),
            "apr_to_may_2020_change": intersection_change("2020-04", "2020-05"),
        },
        "functional_graph": {**graph_metadata, "highest_weighted_degree_node": central},
        "dtw_clustering": dtw_metadata,
        "method_note": "Volume is coverage-normalized across operational detector channels; functional graph edges encode weekly-profile similarity, not physical adjacency or vehicle transfers.",
    }
    (table_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("[analyze] wrote analytical tables and summary", flush=True)
