from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .paths import PROJECT_ROOT, ensure_project_directories, load_settings


TABLES = PROJECT_ROOT / "outputs" / "tables"
METADATA = PROJECT_ROOT / "data" / "metadata"
DASHBOARD_DIR = PROJECT_ROOT / "outputs" / "dashboard"


def _records(frame: pd.DataFrame) -> List[Dict[str, object]]:
    #convert missing values to null before building the JSON records
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    tables_used: Sequence[str],
    filters: Sequence[str],
    definitions: Sequence[str],
) -> Dict[str, object]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "DuckDB-compatible SQL over CSV outputs",
            "sql": "SELECT * FROM read_csv_auto('%s', header = true);" % path,
            "description": description,
            "tables_used": list(tables_used),
            "filters": list(filters),
            "metric_definitions": list(definitions),
            "language": "SQL",
        },
    }


def _intersection_labels(monthly: pd.DataFrame) -> pd.DataFrame:
    names = (
        monthly.sort_values("year_month")
        .groupby("intersection_id", as_index=False)["name_fa"]
        .agg(lambda values: next((str(v) for v in values if pd.notna(v) and str(v).strip()), ""))
    )
    names["intersection_label"] = "SCATS " + names["intersection_id"].astype(int).astype(str)
    has_name = names["name_fa"].fillna("").str.strip().ne("")
    names.loc[has_name, "intersection_label"] += " — " + names.loc[has_name, "name_fa"]
    return names


def _paired_change(
    trusted: pd.DataFrame, earlier: str, later: str, value_name: str
) -> pd.DataFrame:
    left = trusted.loc[
        trusted["year_month"].eq(earlier), ["intersection_id", "mean_15min_volume"]
    ].rename(columns={"mean_15min_volume": "earlier_volume"})
    right = trusted.loc[
        trusted["year_month"].eq(later), ["intersection_id", "mean_15min_volume"]
    ].rename(columns={"mean_15min_volume": "later_volume"})
    paired = left.merge(right, on="intersection_id", how="inner", validate="one_to_one")
    paired[value_name] = paired["later_volume"] / paired["earlier_volume"] - 1.0
    paired["earlier_month"] = earlier
    paired["later_month"] = later
    return paired


def _build_daily_trend(city_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily = city_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.month
    baseline = (
        daily.loc[
            daily["date"].dt.year.eq(2019) & daily["city_median_15min_volume"].notna()
        ]
        .groupby(["month", "weekday"], as_index=False)["city_median_15min_volume"]
        .median()
        .rename(columns={"city_median_15min_volume": "baseline_2019_volume"})
    )
    daily = daily.merge(baseline, on=["month", "weekday"], how="left", validate="many_to_one")
    daily["volume_index_2019"] = (
        100.0 * daily["city_median_15min_volume"] / daily["baseline_2019_volume"]
    )
    daily["volume_index_7d"] = daily["volume_index_2019"].rolling(7, min_periods=4).median()
    daily["context_note"] = ""
    holiday = daily["official_holiday_name_en"].fillna("").str.strip()
    daily.loc[holiday.ne(""), "context_note"] = holiday[holiday.ne("")]
    event = daily["manual_event_names"].fillna("").str.strip()
    daily.loc[event.ne(""), "context_note"] = event[event.ne("")]
    daily["traffic_status"] = np.where(
        daily["city_median_15min_volume"].notna(), "Observed", "Missing / insufficient coverage"
    )
    daily["date_label"] = daily["date"].dt.strftime("%Y-%m-%d")

    observed = daily[daily["volume_index_2019"].notna()].copy()
    observed["deviation_points"] = observed["volume_index_2019"] - 100.0
    anomalies = pd.concat(
        [observed.nsmallest(12, "deviation_points"), observed.nlargest(8, "deviation_points")],
        ignore_index=True,
    ).drop_duplicates("date")
    anomalies["direction"] = np.where(anomalies["deviation_points"] < 0, "Low", "High")
    anomalies = anomalies.sort_values("deviation_points")
    return daily, anomalies


def _build_approach_shares(approach: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({"intersection_id": sorted(approach["intersection_id"].unique())})
    for direction in ("south", "west", "north", "east"):
        work = approach.loc[
            approach[direction + "_coverage"].ge(0.8)
            & approach[direction + "_mean"].notna()
            & approach[direction + "_observations"].gt(0),
            ["intersection_id", direction + "_mean", direction + "_observations"],
        ].copy()
        work["weighted"] = work[direction + "_mean"] * work[direction + "_observations"]
        grouped = work.groupby("intersection_id", as_index=False).agg(
            weighted=("weighted", "sum"), observations=(direction + "_observations", "sum")
        )
        grouped[direction + "_mean"] = grouped["weighted"] / grouped["observations"]
        output = output.merge(
            grouped[["intersection_id", direction + "_mean"]], on="intersection_id", how="left"
        )
    directions = [direction + "_mean" for direction in ("south", "west", "north", "east")]
    output["approach_total"] = output[directions].sum(axis=1, min_count=4)
    for direction in ("south", "west", "north", "east"):
        output[direction + "_share"] = output[direction + "_mean"] / output["approach_total"]
    output = output.merge(
        graph[["intersection_id", "mean_volume", "community"]], on="intersection_id", how="inner"
    )
    output = output.dropna(subset=[direction + "_share" for direction in ("south", "west", "north", "east")])
    return output.nlargest(12, "mean_volume")


def _build_hour_week_heatmap(profile: pd.DataFrame) -> pd.DataFrame:
    city = profile.groupby(["weekday", "hour"], as_index=False).agg(
        volume_sum=("volume_sum", "sum"), observations=("observations", "sum")
    )
    city["mean_volume"] = city["volume_sum"] / city["observations"]
    pivot = city.pivot(index="weekday", columns="hour", values="mean_volume").reindex(range(7))
    pivot.columns = ["h%02d" % hour for hour in pivot.columns]
    pivot = pivot.reset_index()
    names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    pivot["weekday_name"] = pivot["weekday"].map(names)
    return pivot


def _build_month_heatmap(trusted: pd.DataFrame, labels: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    baseline = trusted.loc[trusted["year_month"].between("2019-01", "2019-12")].groupby(
        "intersection_id", as_index=False
    ).agg(baseline_volume=("mean_15min_volume", "median"), baseline_months=("year_month", "nunique"))
    candidates = (
        trusted.groupby("intersection_id", as_index=False)
        .agg(mean_volume=("mean_15min_volume", "mean"), trusted_months=("year_month", "nunique"))
        .merge(baseline, on="intersection_id", how="inner")
    )
    selected = candidates.loc[
        candidates["trusted_months"].ge(18) & candidates["baseline_months"].ge(8)
    ].nlargest(12, "mean_volume")
    ids = selected["intersection_id"].astype(int).tolist()
    work = trusted.loc[
        trusted["intersection_id"].isin(ids)
        & trusted["year_month"].between("2019-01", "2020-10")
    ].merge(baseline[["intersection_id", "baseline_volume"]], on="intersection_id", how="left")
    work["volume_index"] = 100.0 * work["mean_15min_volume"] / work["baseline_volume"]
    pivot = work.pivot(index="year_month", columns="intersection_id", values="volume_index")
    pivot = pivot.reindex(pd.period_range("2019-01", "2020-10", freq="M").astype(str))
    pivot.index.name = "year_month"
    pivot.columns = ["i%d" % int(value) for value in pivot.columns]
    return pivot.reset_index(), ids


def _community_summary(graph: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    named = graph.merge(labels[["intersection_id", "intersection_label"]], on="intersection_id", how="left")
    top_index = named.groupby("community")["weighted_degree"].idxmax()
    top = named.loc[top_index, ["community", "intersection_label"]].rename(
        columns={"intersection_label": "most_central_intersection"}
    )
    summary = named.groupby("community", as_index=False).agg(
        intersections=("intersection_id", "nunique"),
        median_volume=("mean_volume", "median"),
        mean_weighted_degree=("weighted_degree", "mean"),
        median_trusted_months=("trusted_months", "median"),
    )
    summary = summary.merge(top, on="community", how="left")
    summary["community_label"] = "Community " + summary["community"].astype(int).astype(str)
    return summary


def _add_missing_source_month(quality: pd.DataFrame) -> pd.DataFrame:
    quality = quality.copy()
    quality["status"] = "Complete"
    quality.loc[quality["source_month"].eq("2020-12"), "status"] = "Partial"
    if not quality["source_month"].eq("2020-11").any():
        missing = {column: np.nan for column in quality.columns}
        missing.update(
            {
                "source_month": "2020-11",
                "timestamps_observed": 0,
                "timestamps_expected_full_month": 2880,
                "source_temporal_coverage": 0.0,
                "intersection_records": 0,
                "records_with_normalized_volume": 0,
                "intersections": 0,
                "status": "Missing",
            }
        )
        quality = pd.concat([quality, pd.DataFrame([missing])], ignore_index=True)
    return quality.sort_values("source_month")


def build_dashboard_datasets() -> Tuple[Dict[str, pd.DataFrame], Dict[str, object]]:
    settings = load_settings()
    monthly = pd.read_csv(TABLES / "intersection_monthly.csv")
    daily = pd.read_csv(TABLES / "city_daily_with_context.csv", low_memory=False)
    profile = pd.read_csv(TABLES / "hourly_weekday_profile.csv")
    approach = pd.read_csv(TABLES / "approach_monthly.csv")
    approach_1081 = pd.read_csv(TABLES / "intersection_1081_approach_monthly.csv")
    graph = pd.read_csv(TABLES / "functional_graph_nodes.csv")
    edges = pd.read_csv(TABLES / "functional_graph_edges.csv")
    quality = pd.read_csv(TABLES / "source_month_quality.csv")
    context_effects = pd.read_csv(TABLES / "traffic_context_associations.csv")
    seasonality = pd.read_csv(TABLES / "month_of_year_profile.csv")
    geocoded = pd.read_csv(METADATA / "intersections_geocoded.csv")
    summary = json.loads((TABLES / "analysis_summary.json").read_text(encoding="utf-8"))

    labels = _intersection_labels(monthly)
    trusted = monthly.loc[
        monthly["temporal_coverage"].ge(float(settings["trusted_temporal_coverage"]))
        & monthly["mean_detector_coverage"].ge(float(settings["trusted_detector_coverage"]))
    ].copy()
    if trusted.duplicated(["year_month", "intersection_id"]).any():
        raise ValueError("trusted monthly table is not unique by month and intersection")

    shock = _paired_change(trusted, "2020-02", "2020-03", "shock_change")
    recovery = _paired_change(trusted, "2020-04", "2020-05", "recovery_change")
    latest = _paired_change(trusted, "2020-01", "2021-01", "jan_2021_vs_2020")
    shock_recovery = (
        shock[["intersection_id", "shock_change"]]
        .merge(recovery[["intersection_id", "recovery_change"]], on="intersection_id", how="inner")
        .merge(labels, on="intersection_id", how="left")
        .merge(graph[["intersection_id", "community", "mean_volume", "trusted_months"]], on="intersection_id", how="left")
    )
    shock_recovery["community_label"] = "Community " + shock_recovery["community"].astype(int).astype(str)

    pre = trusted.loc[trusted["year_month"].between("2019-01", "2020-02")].groupby(
        "intersection_id", as_index=False
    ).agg(pre_pandemic_volume=("mean_15min_volume", "median"), pre_months=("year_month", "nunique"))
    post = trusted.loc[trusted["year_month"].between("2020-03", "2020-10")].groupby(
        "intersection_id", as_index=False
    ).agg(mar_oct_2020_volume=("mean_15min_volume", "median"), mar_oct_months=("year_month", "nunique"))
    volatility = trusted.groupby("intersection_id", as_index=False).agg(
        trusted_months_calc=("year_month", "nunique"),
        monthly_mean_volume=("mean_15min_volume", "mean"),
        monthly_std_volume=("mean_15min_volume", "std"),
    )
    volatility["monthly_volume_cv"] = volatility["monthly_std_volume"] / volatility["monthly_mean_volume"]
    ordered = trusted.sort_values(["intersection_id", "year_month"]).copy()
    ordered["period_start"] = pd.to_datetime(ordered["year_month"] + "-01")
    ordered["previous_period_start"] = ordered.groupby("intersection_id")["period_start"].shift(1)
    ordered["previous_volume"] = ordered.groupby("intersection_id")["mean_15min_volume"].shift(1)
    ordered["is_consecutive"] = ordered["period_start"].eq(
        ordered["previous_period_start"] + pd.offsets.MonthBegin(1)
    )
    ordered["consecutive_change"] = np.where(
        ordered["is_consecutive"],
        ordered["mean_15min_volume"] / ordered["previous_volume"] - 1.0,
        np.nan,
    )
    regime = ordered.groupby("intersection_id", as_index=False).agg(
        max_abs_consecutive_month_change=(
            "consecutive_change", lambda values: values.abs().max()
        )
    )
    regime["possible_regime_shift"] = np.where(
        regime["max_abs_consecutive_month_change"].ge(0.5), "Review", "No large shift"
    )
    intersection_summary = (
        graph.merge(labels, on="intersection_id", how="left")
        .merge(pre, on="intersection_id", how="left")
        .merge(post, on="intersection_id", how="left")
        .merge(volatility, on="intersection_id", how="left")
        .merge(regime, on="intersection_id", how="left")
        .merge(shock[["intersection_id", "shock_change"]], on="intersection_id", how="left")
        .merge(recovery[["intersection_id", "recovery_change"]], on="intersection_id", how="left")
        .merge(latest[["intersection_id", "jan_2021_vs_2020"]], on="intersection_id", how="left")
    )
    intersection_summary["mar_oct_vs_pre"] = (
        intersection_summary["mar_oct_2020_volume"] / intersection_summary["pre_pandemic_volume"] - 1.0
    )
    intersection_summary["volume_rank"] = intersection_summary["mean_volume"].rank(
        ascending=False, method="min"
    )
    intersection_summary["centrality_rank"] = intersection_summary["weighted_degree"].rank(
        ascending=False, method="min"
    )
    intersection_summary["community_label"] = (
        "Community " + intersection_summary["community"].astype(int).astype(str)
    )

    shock_rank = shock.merge(labels, on="intersection_id", how="left").nsmallest(12, "shock_change")
    latest_named = latest.merge(labels, on="intersection_id", how="left")
    latest_extremes = pd.concat(
        [latest_named.nsmallest(8, "jan_2021_vs_2020"), latest_named.nlargest(8, "jan_2021_vs_2020")],
        ignore_index=True,
    ).drop_duplicates("intersection_id").sort_values("jan_2021_vs_2020")

    city_monthly = pd.read_csv(TABLES / "city_monthly.csv")
    all_months = pd.DataFrame(
        {"year_month": pd.period_range("2018-10", "2021-01", freq="M").astype(str)}
    )
    city_monthly = all_months.merge(city_monthly, on="year_month", how="left", validate="one_to_one")
    city_monthly["month_date"] = city_monthly["year_month"] + "-01"
    city_monthly["source_status"] = "Complete"
    city_monthly.loc[city_monthly["year_month"].eq("2020-11"), "source_status"] = "Missing"
    city_monthly.loc[city_monthly["year_month"].eq("2020-12"), "source_status"] = "Partial"

    paired_changes = pd.read_csv(TABLES / "paired_month_changes.csv")
    paired_changes["period_label"] = paired_changes["previous_month"] + " → " + paired_changes["year_month"]
    paired_changes["month_date"] = paired_changes["year_month"] + "-01"

    daily_trend, anomalies = _build_daily_trend(daily)
    month_heatmap, heatmap_ids = _build_month_heatmap(trusted, labels)
    hour_week = _build_hour_week_heatmap(profile)
    approach_shares = _build_approach_shares(approach, graph).merge(
        labels[["intersection_id", "intersection_label"]], on="intersection_id", how="left"
    )

    approach_rows = []
    for row in approach_1081.to_dict(orient="records"):
        for direction in ("south", "west", "north", "east"):
            approach_rows.append(
                {
                    "year_month": row["year_month"],
                    "approach": direction.title(),
                    "mean_volume": row[direction + "_mean"],
                    "coverage": row[direction + "_coverage"],
                    "observations": row[direction + "_observations"],
                }
            )
    approaches_1081 = pd.DataFrame(approach_rows)

    community_nodes = graph.merge(labels, on="intersection_id", how="left")
    community_nodes["community_label"] = "Community " + community_nodes["community"].astype(int).astype(str)
    communities = _community_summary(graph, labels)
    reliable_map = geocoded.loc[
        geocoded["geocode_confidence"].isin(["medium", "high"])
        & geocoded["latitude"].notna()
        & geocoded["longitude"].notna()
    ]
    map_points = community_nodes.merge(
        reliable_map[["intersection_id", "latitude", "longitude", "geocode_confidence"]],
        on="intersection_id",
        how="inner",
        validate="one_to_one",
    )

    quality_full = _add_missing_source_month(quality)
    quality_full["month_date"] = quality_full["source_month"] + "-01"
    context_effects = context_effects.copy()
    #adjusting for weekday already removes the Friday pattern
    #show it in the weekly heatmap instead
    context_effects = context_effects.loc[
        ~context_effects["context_factor"].eq("friday_weekend")
    ].copy()
    context_effects["factor_label"] = context_effects["context_factor"].str.replace("_", " ").str.title()
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    seasonality["month_name"] = seasonality["month"].map(month_names)
    seasonality["common_panel_intersections"] = summary["time_series"][
        "month_of_year_common_panel_intersections"
    ]

    headline = pd.DataFrame(
        [
            {
                "latest_yoy_change": latest["jan_2021_vs_2020"].median(),
                "latest_yoy_sites": len(latest),
                "shock_change": shock["shock_change"].median(),
                "shock_sites": len(shock),
                "recovery_change": recovery["recovery_change"].median(),
                "recovery_sites": len(recovery),
                "observed_traffic_days": int(daily["traffic_data_available"].sum()),
                "calendar_days": int(len(daily)),
                "invalid_channel_rate": summary["quality"]["invalid_channel_rate"],
                "trusted_site_months": int(len(trusted)),
                "graph_nodes": int(len(graph)),
                "graph_edges": int(len(edges)),
                "graph_communities": int(graph["community"].nunique()),
                "mapped_sites": int(len(map_points)),
                "map_coverage": len(map_points) / len(graph),
            }
        ]
    )

    checks = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "monthly_rows": int(len(monthly)),
        "trusted_monthly_rows": int(len(trusted)),
        "monthly_duplicate_keys": int(monthly.duplicated(["year_month", "intersection_id"]).sum()),
        "daily_calendar_rows": int(len(daily)),
        "daily_duplicate_dates": int(pd.to_datetime(daily["date"]).duplicated().sum()),
        "graph_nodes": int(len(graph)),
        "graph_duplicate_nodes": int(graph["intersection_id"].duplicated().sum()),
        "graph_edges": int(len(edges)),
        "shock_paired_sites": int(len(shock)),
        "recovery_paired_sites": int(len(recovery)),
        "latest_yoy_paired_sites": int(len(latest)),
        "possible_regime_shift_sites": int(regime["possible_regime_shift"].eq("Review").sum()),
        "reliable_mapped_graph_nodes": int(len(map_points)),
        "month_heatmap_intersections": heatmap_ids,
        "missing_source_months": ["2020-11"],
        "partial_source_months": ["2020-12"],
        "passed": bool(
            monthly.duplicated(["year_month", "intersection_id"]).sum() == 0
            and pd.to_datetime(daily["date"]).duplicated().sum() == 0
            and graph["intersection_id"].duplicated().sum() == 0
            and len(shock) >= 30
            and len(recovery) >= 30
            and len(latest) >= 25
        ),
        "confidence": "Share with caveats",
        "caveats": [
            "November 2020 is missing and December 2020 is partial.",
            "The set of trusted intersections changes by month, so paired-site changes are used for headline comparisons.",
            "Functional graph edges show similar weekly patterns, not roads or vehicle movements.",
            "Map coordinates are OpenStreetMap candidates and need municipal GIS review.",
            "Calendar, weather, and COVID results are associations, not proof of cause.",
        ],
    }

    datasets = {
        "headline": headline,
        "city_monthly": city_monthly,
        "paired_changes": paired_changes,
        "daily_trend": daily_trend[[
            "date_label", "volume_index_2019", "volume_index_7d", "city_median_15min_volume",
            "trusted_intersection_count", "context_note", "traffic_status",
        ]],
        "context_effects": context_effects,
        "seasonality": seasonality,
        "shock_recovery": shock_recovery,
        "shock_rank": shock_rank,
        "latest_extremes": latest_extremes,
        "month_heatmap": month_heatmap,
        "hour_week_heatmap": hour_week,
        "approaches_1081": approaches_1081,
        "approach_shares": approach_shares,
        "community_nodes": community_nodes,
        "community_summary": communities,
        "map_points": map_points,
        "source_quality": quality_full,
        "trusted_months": graph[["intersection_id", "trusted_months"]],
        "intersection_summary": intersection_summary,
        "anomaly_dates": anomalies[[
            "date_label", "direction", "deviation_points", "volume_index_2019",
            "city_median_15min_volume", "trusted_intersection_count", "context_note",
            "precipitation_sum_mm", "covid_stringency_index",
        ]],
    }
    return datasets, checks


def _chart(
    chart_id: str,
    title: str,
    subtitle: str,
    description: str,
    chart_type: str,
    dataset: str,
    source_id: str,
    intent: str,
    question: str,
    rationale: str,
    encodings: Dict[str, object],
    layout: str = "half",
    **extra: object,
) -> Dict[str, object]:
    chart: Dict[str, object] = {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "headerMarkdown": description,
        "showDescription": True,
        "type": chart_type,
        "dataset": dataset,
        "sourceId": source_id,
        "intent": intent,
        "question": question,
        "rationale": rationale,
        "encodings": encodings,
        "layout": layout,
    }
    chart.update(extra)
    return chart


def build_dashboard_artifact() -> Path:
    ensure_project_directories()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    datasets, checks = build_dashboard_datasets()
    summary = json.loads((TABLES / "analysis_summary.json").read_text(encoding="utf-8"))
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    datasets["intersection_summary"].to_csv(
        TABLES / "dashboard_intersection_change_summary.csv", index=False
    )
    datasets["daily_trend"].to_csv(TABLES / "dashboard_daily_network_index.csv", index=False)
    (TABLES / "dashboard_validation.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    shock_recovery = datasets["shock_recovery"]
    community_nodes = datasets["community_nodes"]
    hour_week = datasets["hour_week_heatmap"].set_index("weekday_name")
    hour_fields = ["h%02d" % hour for hour in range(24)]
    hour_long = hour_week[hour_fields].stack()
    peak_weekday, peak_field = hour_long.idxmax()
    friday_mean = float(hour_week.loc["Fri", hour_fields].mean())
    other_day_mean = float(hour_week.drop(index="Fri")[hour_fields].to_numpy().mean())
    context_lookup = datasets["context_effects"].set_index("context_factor")
    insights = {
        "shock_sites_declining": int((datasets["intersection_summary"]["shock_change"].dropna() < 0).sum()),
        "shock_sites_total": int(datasets["intersection_summary"]["shock_change"].notna().sum()),
        "recovery_sites_rising": int((datasets["intersection_summary"]["recovery_change"].dropna() > 0).sum()),
        "recovery_sites_total": int(datasets["intersection_summary"]["recovery_change"].notna().sum()),
        "shock_recovery_correlation": float(
            shock_recovery["shock_change"].corr(shock_recovery["recovery_change"])
        ),
        "volume_centrality_spearman": float(
            community_nodes["mean_volume"].corr(community_nodes["weighted_degree"], method="spearman")
        ),
        "weekly_peak_weekday": peak_weekday,
        "weekly_peak_hour": int(peak_field[1:]),
        "weekly_peak_mean_volume": float(hour_long.max()),
        "friday_mean_vs_other_days": friday_mean / other_day_mean - 1.0,
        "nowruz_adjusted_difference_points": float(
            context_lookup.loc["nowruz_window", "adjusted_index_difference_points"]
        ),
        "holiday_adjusted_difference_points": float(
            context_lookup.loc["official_holiday", "adjusted_index_difference_points"]
        ),
        "wet_weather_adjusted_difference_points": float(
            context_lookup.loc["wet_weather", "adjusted_index_difference_points"]
        ),
        "regime_shift_review_sites": int(checks["possible_regime_shift_sites"]),
        "intersection_1081": summary["intersection_1081"],
    }
    (TABLES / "dashboard_insights.json").write_text(
        json.dumps(insights, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    traffic_source = _source(
        "traffic_analysis",
        "Trusted SCATS traffic analysis",
        "outputs/tables/intersection_monthly.csv",
        "Monthly and paired-intersection traffic measures built from coverage-normalized SCATS counts.",
        ["intersection_monthly.csv", "city_monthly.csv", "paired_month_changes.csv"],
        [
            "Temporal coverage is at least 80%.",
            "Mean operational detector coverage is at least 80%.",
            "Headline changes compare the same intersections in both periods.",
        ],
        [
            "Mean 15-minute volume is the coverage-normalized intersection count averaged across valid 15-minute records.",
            "Paired change is later-month volume divided by earlier-month volume minus one, calculated per intersection; the headline is the median site change.",
            "The city monthly line is the median of trusted intersection means for that month.",
        ],
    )
    daily_source = _source(
        "daily_context",
        "Daily traffic and external context",
        "outputs/tables/city_daily_with_context.csv",
        "Daily city traffic joined one-to-one to Iranian calendar, ERA5 weather, documented events, and national COVID policy fields.",
        ["city_daily_with_context.csv", "traffic_context_associations.csv"],
        ["Daily city traffic uses intersections with at least 80% temporal coverage."],
        [
            "2019 volume index equals daily city median volume divided by the matching 2019 month-and-weekday median, times 100.",
            "Context effect is the difference in adjusted volume index between flagged and comparison days.",
        ],
    )
    graph_source = _source(
        "graph_analysis",
        "Functional traffic graph",
        "outputs/tables/functional_graph_nodes.csv",
        "A graph where intersections are connected when their 168-bin weekly traffic profiles are similar.",
        ["functional_graph_nodes.csv", "functional_graph_edges.csv", "dtw_clusters.csv"],
        ["Graph-eligible intersections have enough hourly-weekday observations for stable profiles."],
        [
            "Weighted degree is the sum of profile-correlation edge weights touching an intersection.",
            "Community is a group of intersections with more similar weekly profiles inside the group.",
            "DTW cluster groups intersections by the shape of their 24-hour profile, allowing small peak-time shifts.",
        ],
    )
    weekly_source = _source(
        "weekly_profile",
        "Hourly-weekday traffic profile",
        "outputs/tables/hourly_weekday_profile.csv",
        "Intersection traffic summed and counted for each weekday and hour, then aggregated to a network mean.",
        ["hourly_weekday_profile.csv"],
        ["Only valid coverage-normalized 15-minute volumes are included."],
        [
            "Hour-of-week mean volume is summed valid volume divided by summed valid observations for that weekday and hour.",
        ],
    )
    approach_source = _source(
        "approach_analysis",
        "Inbound approach detector groups",
        "outputs/tables/approach_monthly.csv",
        "Monthly south, west, north, and east entry-side detector volumes derived from the intersection diagrams.",
        ["approach_monthly.csv", "intersection_1081_approach_monthly.csv"],
        ["Approach shares use direction-months with at least 80% detector coverage."],
        [
            "Approach volume is an entry-side detector count near the stop line; it is not a measured exit or turning movement.",
            "Approach share is one direction's mean divided by the sum of the four direction means.",
        ],
    )
    map_source = _source(
        "map_candidates",
        "Reviewed OpenStreetMap coordinate candidates",
        "data/metadata/intersections_geocoded.csv",
        "Medium- and high-confidence geocode candidates matched to graph intersections.",
        ["intersections_geocoded.csv", "functional_graph_nodes.csv"],
        ["Only medium- and high-confidence candidates with coordinates are shown."],
        ["Map coverage is reliable coordinate candidates divided by functional graph nodes."],
    )
    quality_source = _source(
        "quality_audit",
        "SCATS source and detector quality audit",
        "outputs/tables/source_month_quality.csv",
        "Source-month completeness and channel-level unavailable-reading checks.",
        ["source_month_quality.csv", "analysis_summary.json"],
        ["SCATS values 2046, 2047, and NA are unavailable; zero stays a valid count."],
        [
            "Source temporal coverage is observed 15-minute timestamps divided by expected calendar-month timestamps.",
            "Trusted months meet both 80% temporal and detector coverage thresholds.",
        ],
    )
    sources = [
        traffic_source, daily_source, weekly_source, graph_source,
        approach_source, map_source, quality_source,
    ]

    h = datasets["headline"].iloc[0]
    shock_med = float(h["shock_change"])
    recovery_med = float(h["recovery_change"])
    latest_med = float(h["latest_yoy_change"])
    heatmap_ids = checks["month_heatmap_intersections"]
    heatmap_series = [
        {"field": "i%d" % int(intersection_id), "label": "SCATS %d" % int(intersection_id)}
        for intersection_id in heatmap_ids
    ]
    hour_series = [
        {"field": "h%02d" % hour, "label": "%02d:00" % hour} for hour in range(24)
    ]

    charts = [
        _chart(
            "city_monthly",
            "Network traffic by month",
            "Median trusted intersection volume, Oct 2018–Jan 2021; vehicles per 15 minutes",
            "This line shows the typical intersection each month. A gap is data that should not be treated as a normal month: November 2020 is missing and December 2020 is partial.",
            "line", "city_monthly", "traffic_analysis", "trend",
            "How did typical intersection traffic move over the full archive?",
            "A line chart shows the continuous monthly path and makes the March 2020 break easy to see.",
            {
                "x": {"field": "month_date", "type": "temporal", "label": "Month"},
                "y": {"field": "median_intersection_volume", "type": "quantitative", "label": "Vehicles / 15 min"},
                "tooltip": [
                    {"field": "intersections", "type": "quantitative", "label": "Trusted intersections", "format": "number"},
                    {"field": "p25_intersection_volume", "type": "quantitative", "label": "25th percentile", "format": "number"},
                    {"field": "p75_intersection_volume", "type": "quantitative", "label": "75th percentile", "format": "number"},
                    {"field": "source_status", "type": "nominal", "label": "Source status"},
                ],
            },
            layout="full", valueFormat="number",
        ),
        _chart(
            "paired_changes",
            "Paired month-to-month change",
            "Median change for the same trusted intersections in adjacent months",
            "Each bar compares the same sites in both months. This avoids a false change caused only by intersections entering or leaving the trusted panel.",
            "bar", "paired_changes", "traffic_analysis", "comparison",
            "Which month-to-month movements were large after holding the site set constant?",
            "Signed bars make the size and direction of each paired movement clear.",
            {
                "x": {"field": "month_date", "type": "temporal", "label": "Later month"},
                "y": {"field": "median_change", "type": "quantitative", "label": "Median paired change", "format": "percent"},
                "tooltip": [
                    {"field": "period_label", "type": "nominal", "label": "Comparison"},
                    {"field": "paired_intersections", "type": "quantitative", "label": "Paired sites", "format": "number"},
                    {"field": "p25_change", "type": "quantitative", "label": "25th percentile", "format": "percent"},
                    {"field": "p75_change", "type": "quantitative", "label": "75th percentile", "format": "percent"},
                ],
            },
            referenceLines=[{"axis": "y", "value": 0, "label": "No change", "color": "neutral"}],
            valueFormat="percent",
        ),
        _chart(
            "seasonality_2019",
            "Month-of-year pattern in 2019",
            "Median volume for the same 34 intersections in all twelve months",
            "This is the cleanest pre-pandemic seasonality view because it keeps the site set fixed. January is the highest typical month and November is the lowest in this common panel.",
            "bar", "seasonality", "traffic_analysis", "comparison",
            "How much did normal traffic vary by month before the pandemic?",
            "A common-panel monthly bar chart separates seasonality from changes in which sites report.",
            {
                "x": {"field": "month_name", "type": "ordinal", "label": "Month"},
                "y": {"field": "median_volume", "type": "quantitative", "label": "Median vehicles / 15 min"},
                "tooltip": [
                    {"field": "mean_volume", "type": "quantitative", "label": "Mean volume", "format": "number"},
                    {"field": "standard_deviation", "type": "quantitative", "label": "Standard deviation", "format": "number"},
                    {"field": "common_panel_intersections", "type": "quantitative", "label": "Common-panel sites", "format": "number"},
                ],
            },
            valueFormat="number",
        ),
        _chart(
            "daily_index",
            "Daily traffic index",
            "Seven-day median; 100 means the matching month-and-weekday level in 2019",
            "The index removes normal month and weekday differences using 2019 as the baseline. It still describes traffic; it does not prove why traffic changed.",
            "line", "daily_trend", "daily_context", "trend",
            "When did daily traffic move away from its normal 2019 pattern?",
            "A smoothed daily line shows the timing of shocks while keeping the missing-data gap visible.",
            {
                "x": {"field": "date_label", "type": "temporal", "label": "Date"},
                "y": {"field": "volume_index_7d", "type": "quantitative", "label": "Traffic index"},
                "tooltip": [
                    {"field": "volume_index_2019", "type": "quantitative", "label": "Daily index", "format": "number"},
                    {"field": "city_median_15min_volume", "type": "quantitative", "label": "Vehicles / 15 min", "format": "number"},
                    {"field": "trusted_intersection_count", "type": "quantitative", "label": "Trusted sites", "format": "number"},
                    {"field": "context_note", "type": "nominal", "label": "Calendar / event note"},
                ],
            },
            layout="full", valueFormat="number",
            referenceLines=[{"axis": "y", "value": 100, "label": "2019 baseline", "color": "neutral"}],
        ),
        _chart(
            "context_effects",
            "Traffic difference on context days",
            "Index-point difference after controlling for Gregorian month and weekday",
            "Negative values mean traffic was lower on flagged days than on comparison days. Weather, holidays, events, and COVID policy can overlap, so these are associations, not causal effects.",
            "horizontalBar", "context_effects", "daily_context", "comparison",
            "Which external contexts line up with higher or lower traffic?",
            "A ranked signed bar chart compares factors on the same adjusted index scale.",
            {
                "x": {"field": "factor_label", "type": "nominal", "label": "Context factor"},
                "y": {"field": "adjusted_index_difference_points", "type": "quantitative", "label": "Index-point difference"},
                "tooltip": [
                    {"field": "flagged_days_with_traffic", "type": "quantitative", "label": "Flagged days", "format": "number"},
                    {"field": "comparison_days_with_traffic", "type": "quantitative", "label": "Comparison days", "format": "number"},
                    {"field": "flagged_mean_adjusted_volume_index", "type": "quantitative", "label": "Flagged-day index", "format": "number"},
                ],
            },
            referenceLines=[{"axis": "y", "value": 0, "label": "No difference", "color": "neutral"}],
        ),
        _chart(
            "shock_recovery",
            "Intersection shock and recovery",
            "Each point is one intersection; x = Feb–Mar 2020, y = Apr–May 2020",
            "Points farther left had a larger March fall. Points higher up had a larger May recovery. Color shows the functional community, which is based on weekly traffic shape.",
            "scatter", "shock_recovery", "traffic_analysis", "relationship",
            "Did the intersections with the largest March fall also have the largest May recovery?",
            "A scatter plot shows site-level spread, exceptions, and the relationship between the two changes.",
            {
                "x": {"field": "shock_change", "type": "quantitative", "label": "Feb–Mar change", "format": "percent"},
                "y": {"field": "recovery_change", "type": "quantitative", "label": "Apr–May change", "format": "percent"},
                "color": {"field": "community_label", "type": "nominal", "label": "Functional community"},
                "size": {"field": "mean_volume", "type": "quantitative", "label": "Mean volume"},
                "label": {"field": "intersection_label", "type": "nominal", "label": "Intersection"},
                "tooltip": [
                    {"field": "intersection_label", "type": "nominal", "label": "Intersection"},
                    {"field": "mean_volume", "type": "quantitative", "label": "Mean volume", "format": "number"},
                    {"field": "trusted_months", "type": "quantitative", "label": "Trusted months", "format": "number"},
                ],
            },
            layout="full",
            combinationRationale="Color separates functional communities; point size shows normal traffic scale without changing the two change axes.",
            referenceLines=[
                {"axis": "x", "value": 0, "label": "No March change", "color": "neutral"},
                {"axis": "y", "value": 0, "label": "No May change", "color": "neutral"},
            ],
        ),
        _chart(
            "shock_rank",
            "Largest February–March 2020 falls",
            "Twelve intersections with the lowest paired change",
            "This ranking shows where the March fall was strongest. It is limited to intersections trusted in both February and March.",
            "horizontalBar", "shock_rank", "traffic_analysis", "comparison",
            "Which intersections had the largest March 2020 decline?",
            "A sorted horizontal bar makes long intersection names and negative changes easy to compare.",
            {
                "x": {"field": "intersection_label", "type": "nominal", "label": "Intersection"},
                "y": {"field": "shock_change", "type": "quantitative", "label": "Feb–Mar change", "format": "percent"},
                "tooltip": [
                    {"field": "earlier_volume", "type": "quantitative", "label": "February volume", "format": "number"},
                    {"field": "later_volume", "type": "quantitative", "label": "March volume", "format": "number"},
                ],
            },
            valueFormat="percent",
        ),
        _chart(
            "latest_extremes",
            "January 2021 change from January 2020",
            "Eight largest falls and eight largest rises among paired trusted sites",
            "Using the same calendar month reduces seasonality. The chart still compares only two points, so use the detail table for the longer history.",
            "horizontalBar", "latest_extremes", "traffic_analysis", "comparison",
            "Which intersections were most different in January 2021 than one year earlier?",
            "A signed extremes chart keeps both improving and declining sites visible.",
            {
                "x": {"field": "intersection_label", "type": "nominal", "label": "Intersection"},
                "y": {"field": "jan_2021_vs_2020", "type": "quantitative", "label": "Jan 2021 vs Jan 2020", "format": "percent"},
                "tooltip": [
                    {"field": "earlier_volume", "type": "quantitative", "label": "Jan 2020 volume", "format": "number"},
                    {"field": "later_volume", "type": "quantitative", "label": "Jan 2021 volume", "format": "number"},
                ],
            },
            valueFormat="percent", referenceLines=[{"axis": "y", "value": 0, "label": "No change", "color": "neutral"}],
        ),
        _chart(
            "month_heatmap",
            "Traffic index for high-volume intersections",
            "Jan 2019–Oct 2020; 100 = each intersection's median 2019 volume",
            "The heatmap compares each intersection with its own 2019 baseline. Darker or lighter cells show relative change, not absolute traffic size.",
            "heatmap", "month_heatmap", "traffic_analysis", "relationship",
            "Was the 2020 traffic shock broad across large intersections or limited to a few sites?",
            "A month-by-intersection heatmap shows common timing and site exceptions in one view.",
            {
                "x": {"field": "year_month", "type": "ordinal", "label": "Month"},
                "y": {"fields": [item["field"] for item in heatmap_series], "type": "quantitative", "label": "Traffic index"},
            },
            layout="full", valueFormat="number",
            combinationRationale="Each series is one intersection on the same self-indexed scale, so heatmap intensity is comparable across rows.",
        ),
        _chart(
            "hour_week_heatmap",
            "Weekly traffic pattern by hour",
            "Network mean volume by weekday and hour; vehicles per 15 minutes",
            "This heatmap shows the normal weekly rhythm. Friday is Iran's main weekend day; bright cells show the busiest hours.",
            "heatmap", "hour_week_heatmap", "weekly_profile", "relationship",
            "At what hours and weekdays is traffic normally highest?",
            "A weekday-by-hour heatmap is the clearest view of a dense 7 by 24 pattern.",
            {
                "x": {"field": "weekday_name", "type": "ordinal", "label": "Weekday"},
                "y": {"fields": [item["field"] for item in hour_series], "type": "quantitative", "label": "Hour"},
            },
            layout="full", valueFormat="number",
            combinationRationale="All 24 hour fields use the same vehicle-count unit and form the second heatmap dimension.",
        ),
        _chart(
            "approaches_1081",
            "Intersection 1081 inbound approaches",
            "Monthly mean entry-side detector volume by direction",
            "North and south carry most of the detected entry volume. These are not exit counts and do not show turning destinations.",
            "line", "approaches_1081", "approach_analysis", "trend",
            "How did each inbound approach at Intersection 1081 change over time?",
            "Four lines keep direction-level movement visible without adding incompatible measures.",
            {
                "x": {"field": "year_month", "type": "temporal", "label": "Month"},
                "y": {"field": "mean_volume", "type": "quantitative", "label": "Vehicles / 15 min"},
                "color": {"field": "approach", "type": "nominal", "label": "Inbound approach"},
                "tooltip": [
                    {"field": "coverage", "type": "quantitative", "label": "Detector coverage", "format": "percent"},
                    {"field": "observations", "type": "quantitative", "label": "Observations", "format": "number"},
                ],
            },
            layout="full", combinationRationale="Color identifies the four inbound directions on one common vehicle-count scale.",
        ),
        _chart(
            "approach_shares",
            "Inbound direction mix at busy intersections",
            "Twelve high-volume sites with four usable approach groups",
            "Each bar totals 100%. The segments show how much detected entry volume comes from south, west, north, and east.",
            "horizontalStackedBar100", "approach_shares", "approach_analysis", "composition",
            "How balanced are the four inbound directions at high-volume intersections?",
            "A 100% stacked bar compares direction mix while removing differences in total site volume.",
            {
                "x": {"field": "intersection_label", "type": "nominal", "label": "Intersection"},
                "y": {"fields": ["south_share", "west_share", "north_share", "east_share"], "type": "quantitative", "label": "Share"},
            },
            layout="full",
            valueFormat="percent",
            combinationRationale="The four direction shares use the same denominator and add to 100% for each intersection.",
        ),
        _chart(
            "centrality_scatter",
            "Traffic volume and functional centrality",
            "Each point is an intersection; centrality is similarity-network weighted degree",
            "A central point has many strong links to sites with similar weekly patterns. It is not necessarily a physically central or high-volume road junction.",
            "scatter", "community_nodes", "graph_analysis", "relationship",
            "Are high-volume intersections also central in the weekly-pattern graph?",
            "A scatter plot shows whether traffic scale and functional centrality move together and where exceptions sit.",
            {
                "x": {"field": "mean_volume", "type": "quantitative", "label": "Mean vehicles / 15 min"},
                "y": {"field": "weighted_degree", "type": "quantitative", "label": "Weighted degree"},
                "color": {"field": "community_label", "type": "nominal", "label": "Functional community"},
                "size": {"field": "trusted_months", "type": "quantitative", "label": "Trusted months"},
                "label": {"field": "intersection_label", "type": "nominal", "label": "Intersection"},
                "tooltip": [
                    {"field": "betweenness_centrality", "type": "quantitative", "label": "Betweenness", "format": "number"},
                    {"field": "dtw_cluster", "type": "quantitative", "label": "DTW cluster", "format": "number"},
                ],
            },
            layout="full", combinationRationale="Color shows community and point size shows data coverage; both help interpret the volume-centrality relationship.",
        ),
        _chart(
            "community_sizes",
            "Functional community sizes",
            "Number of intersections in each weekly-pattern community",
            "Communities group similar traffic rhythms. They are operational peer groups, not administrative districts.",
            "bar", "community_summary", "graph_analysis", "comparison",
            "How is the functional graph split across communities?",
            "A simple bar chart shows the size of each mutually exclusive group.",
            {
                "x": {"field": "community_label", "type": "nominal", "label": "Community"},
                "y": {"field": "intersections", "type": "quantitative", "label": "Intersections"},
                "tooltip": [
                    {"field": "median_volume", "type": "quantitative", "label": "Median volume", "format": "number"},
                    {"field": "most_central_intersection", "type": "nominal", "label": "Most central site"},
                ],
            },
            valueFormat="number",
        ),
        _chart(
            "spatial_scatter",
            "Mapped intersection candidates",
            "Medium- and high-confidence OpenStreetMap candidates; longitude and latitude",
            "This is a coordinate plot, not a road map. Use it to see the spread of mapped candidates; municipal GIS should confirm the locations before planning work.",
            "scatter", "map_points", "map_candidates", "relationship",
            "Where are the reliably geocoded functional intersections located across Isfahan?",
            "A longitude-latitude scatter preserves the spatial pattern without using an unverified basemap.",
            {
                "x": {"field": "longitude", "type": "quantitative", "label": "Longitude"},
                "y": {"field": "latitude", "type": "quantitative", "label": "Latitude"},
                "color": {"field": "community_label", "type": "nominal", "label": "Functional community"},
                "size": {"field": "mean_volume", "type": "quantitative", "label": "Mean traffic volume"},
                "label": {"field": "intersection_label", "type": "nominal", "label": "Intersection"},
                "tooltip": [
                    {"field": "geocode_confidence", "type": "nominal", "label": "Geocode confidence"},
                    {"field": "weighted_degree", "type": "quantitative", "label": "Weighted degree", "format": "number"},
                ],
            },
            layout="full", combinationRationale="Color shows functional community and point size shows traffic scale on the coordinate plane.",
        ),
        _chart(
            "source_coverage",
            "Source-month temporal coverage",
            "Observed 15-minute timestamps divided by expected calendar-month timestamps",
            "A value near 100% means the source month is nearly complete. November 2020 is missing; December 2020 has only a small part of the month.",
            "bar", "source_quality", "quality_audit", "status",
            "Which source months are complete enough to support comparison?",
            "Monthly bars make missing and partial archive periods impossible to overlook.",
            {
                "x": {"field": "month_date", "type": "temporal", "label": "Month"},
                "y": {"field": "source_temporal_coverage", "type": "quantitative", "label": "Temporal coverage", "format": "percent"},
                "tooltip": [
                    {"field": "status", "type": "nominal", "label": "Status"},
                    {"field": "intersections", "type": "quantitative", "label": "Intersections", "format": "number"},
                    {"field": "intersection_records", "type": "quantitative", "label": "Records", "format": "number"},
                ],
            },
            layout="full", valueFormat="percent",
        ),
        _chart(
            "trusted_months",
            "Trusted-month coverage by intersection",
            "Distribution across 74 graph intersections",
            "More trusted months means a more stable long-term comparison. A short history does not make a site useless, but its trend is less certain.",
            "histogram", "trusted_months", "graph_analysis", "distribution",
            "How much usable monthly history does each intersection have?",
            "A histogram shows whether coverage is broad or concentrated in a small set of sites.",
            {
                "x": {"field": "trusted_months", "type": "quantitative", "label": "Trusted months"},
                "y": {"field": "intersection_id", "type": "quantitative", "aggregate": "count", "label": "Intersections"},
            },
            valueFormat="number",
        ),
    ]

    cards = [
        {
            "id": "latest_card", "dataset": "headline", "sourceId": "traffic_analysis",
            "description": "Median paired-site change from January 2020 to January 2021; the same calendar month reduces seasonality.",
            "metrics": [
                {"label": "Jan 2021 vs Jan 2020", "field": "latest_yoy_change", "format": "percent", "signed": True},
                {"label": "Paired sites", "field": "latest_yoy_sites", "format": "number"},
            ],
        },
        {
            "id": "shock_card", "dataset": "headline", "sourceId": "traffic_analysis",
            "description": "Largest network-wide paired monthly fall in the archive.",
            "metrics": [
                {"label": "Feb–Mar 2020", "field": "shock_change", "format": "percent", "signed": True},
                {"label": "Paired sites", "field": "shock_sites", "format": "number"},
            ],
        },
        {
            "id": "recovery_card", "dataset": "headline", "sourceId": "traffic_analysis",
            "description": "Largest paired monthly increase after the March–April low period.",
            "metrics": [
                {"label": "Apr–May 2020", "field": "recovery_change", "format": "percent", "signed": True},
                {"label": "Paired sites", "field": "recovery_sites", "format": "number"},
            ],
        },
        {
            "id": "days_card", "dataset": "headline", "sourceId": "daily_context",
            "description": "Days with a usable city traffic median inside the 854-day context calendar.",
            "metrics": [
                {"label": "Traffic days", "field": "observed_traffic_days", "format": "number"},
                {"label": "Calendar days", "field": "calendar_days", "format": "number"},
            ],
        },
        {
            "id": "quality_card", "dataset": "headline", "sourceId": "quality_audit",
            "description": "Unavailable channel readings are kept missing; trusted site-months meet both 80% thresholds.",
            "metrics": [
                {"label": "Unavailable readings", "field": "invalid_channel_rate", "format": "percent"},
                {"label": "Trusted site-months", "field": "trusted_site_months", "format": "number"},
            ],
        },
        {
            "id": "graph_card", "dataset": "headline", "sourceId": "graph_analysis",
            "description": "Functional graph built from similarity in weekly traffic profiles.",
            "metrics": [
                {"label": "Graph intersections", "field": "graph_nodes", "format": "number"},
                {"label": "Similarity edges", "field": "graph_edges", "format": "number"},
                {"label": "Communities", "field": "graph_communities", "format": "number"},
            ],
        },
        {
            "id": "map_card", "dataset": "headline", "sourceId": "map_candidates",
            "description": "Graph intersections with medium- or high-confidence coordinate candidates.",
            "metrics": [
                {"label": "Mapped candidates", "field": "mapped_sites", "format": "number"},
                {"label": "Graph coverage", "field": "map_coverage", "format": "percent"},
            ],
        },
    ]

    tables = [
        {
            "id": "intersection_detail", "title": "Intersection change and graph detail",
            "subtitle": "All 74 graph intersections; use exact values for lookup and follow-up",
            "headerMarkdown": "Missing change values mean that the intersection did not meet the trusted threshold in both required months.",
            "showDescription": True, "dataset": "intersection_summary", "sourceId": "traffic_analysis",
            "defaultSort": {"field": "shock_change", "direction": "asc"}, "density": "dense", "layout": "full",
            "columns": [
                {"field": "intersection_id", "label": "SCATS ID", "format": "number"},
                {"field": "intersection_label", "label": "Intersection", "type": "text"},
                {"field": "community", "label": "Community", "format": "number"},
                {"field": "mean_volume", "label": "Mean volume", "format": "number"},
                {"field": "shock_change", "label": "Feb–Mar 2020", "format": "percent", "movement": True},
                {"field": "recovery_change", "label": "Apr–May 2020", "format": "percent", "movement": True},
                {"field": "jan_2021_vs_2020", "label": "Jan 2021 YoY", "format": "percent", "movement": True},
                {"field": "mar_oct_vs_pre", "label": "Mar–Oct vs pre", "format": "percent", "movement": True},
                {"field": "monthly_volume_cv", "label": "Monthly CV", "format": "number"},
                {"field": "max_abs_consecutive_month_change", "label": "Largest monthly swing", "format": "percent"},
                {"field": "possible_regime_shift", "label": "Regime-shift review", "type": "text"},
                {"field": "weighted_degree", "label": "Weighted degree", "format": "number"},
                {"field": "trusted_months", "label": "Trusted months", "format": "number"},
            ],
        },
        {
            "id": "anomaly_detail", "title": "Most unusual network days",
            "subtitle": "Twelve lowest and eight highest daily indexes versus matching 2019 month-weekday baselines",
            "headerMarkdown": "An unusual day is a starting point for investigation. The calendar and weather note may explain timing, but it does not prove cause.",
            "showDescription": True, "dataset": "anomaly_dates", "sourceId": "daily_context",
            "defaultSort": {"field": "deviation_points", "direction": "asc"}, "density": "dense", "layout": "full",
            "columns": [
                {"field": "date_label", "label": "Date", "type": "date"},
                {"field": "direction", "label": "Direction", "type": "text"},
                {"field": "deviation_points", "label": "Index-point deviation", "format": "number", "movement": True},
                {"field": "city_median_15min_volume", "label": "Vehicles / 15 min", "format": "number"},
                {"field": "trusted_intersection_count", "label": "Trusted sites", "format": "number"},
                {"field": "context_note", "label": "Calendar / event note", "type": "text"},
                {"field": "precipitation_sum_mm", "label": "Rain/snow, mm", "format": "number"},
                {"field": "covid_stringency_index", "label": "COVID stringency", "format": "number"},
            ],
        },
        {
            "id": "community_detail", "title": "Functional community summary",
            "subtitle": "Traffic-shape groups from the functional similarity graph",
            "headerMarkdown": "The most central site has the highest weighted degree inside that community.",
            "showDescription": True, "dataset": "community_summary", "sourceId": "graph_analysis",
            "defaultSort": {"field": "community", "direction": "asc"}, "density": "spacious",
            "columns": [
                {"field": "community", "label": "Community", "format": "number"},
                {"field": "intersections", "label": "Intersections", "format": "number"},
                {"field": "median_volume", "label": "Median volume", "format": "number"},
                {"field": "mean_weighted_degree", "label": "Mean weighted degree", "format": "number"},
                {"field": "median_trusted_months", "label": "Median trusted months", "format": "number"},
                {"field": "most_central_intersection", "label": "Most central intersection", "type": "text"},
            ],
        },
        {
            "id": "source_quality_detail", "title": "Source-month quality audit",
            "subtitle": "Every archive month, including the missing month added as an explicit row",
            "headerMarkdown": "Use this table before reading any sharp monthly change. Low source coverage can look like a traffic change when it is only missing data.",
            "showDescription": True, "dataset": "source_quality", "sourceId": "quality_audit",
            "defaultSort": {"field": "source_month", "direction": "asc"}, "density": "dense", "layout": "full",
            "columns": [
                {"field": "source_month", "label": "Month", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "source_temporal_coverage", "label": "Temporal coverage", "format": "percent"},
                {"field": "intersections", "label": "Intersections", "format": "number"},
                {"field": "intersection_records", "label": "Raw records", "format": "number"},
            ],
        },
    ]

    blocks: List[Dict[str, object]] = [
        {"id": "title", "type": "markdown", "body": "# Isfahan Intersection Traffic Dashboard"},
        {
            "id": "how_to_read", "type": "markdown",
            "sourceId": "traffic_analysis",
            "body": (
                "This dashboard tracks **75 SCATS intersections** from October 2018 to January 2021. "
                "The main unit is the coverage-normalized number of vehicles in a 15-minute interval. "
                "Use the figures from top to bottom: network change, intersection differences, weekly shape, graph groups, map candidates, then data quality. "
                "All changes are descriptive."
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards]},
        {
            "id": "network_findings", "type": "markdown", "sourceId": "traffic_analysis",
            "body": (
                "## 1. Network change\n\n"
                "The largest paired fall was **%.1f%% from February to March 2020** across **%d intersections**. "
                "The largest rise was **%.1f%% from April to May 2020** across **%d intersections**. "
                "For the longer comparison, January 2021 was **%+.1f%%** versus January 2020 across **%d paired sites**."
                % (
                    100 * shock_med, int(h["shock_sites"]), 100 * recovery_med,
                    int(h["recovery_sites"]), 100 * latest_med, int(h["latest_yoy_sites"]),
                )
            ),
        },
        {"id": "city_monthly_block", "type": "chart", "chartId": "city_monthly", "layout": "full"},
        {"id": "paired_changes_block", "type": "chart", "chartId": "paired_changes", "layout": "half"},
        {"id": "seasonality_block", "type": "chart", "chartId": "seasonality_2019", "layout": "half"},
        {"id": "context_effects_block", "type": "chart", "chartId": "context_effects", "layout": "half"},
        {"id": "daily_index_block", "type": "chart", "chartId": "daily_index", "layout": "full"},
        {
            "id": "intersection_section", "type": "markdown",
            "body": "## 2. How individual intersections changed\n\nThe next figures keep the intersection as the unit. This shows whether a city-wide movement was broad or driven by a few sites.",
        },
        {
            "id": "intersection_findings", "type": "markdown", "sourceId": "traffic_analysis",
            "body": (
                "All **%d of %d paired sites fell** from February to March 2020, and all **%d of %d paired sites rose** from April to May. "
                "The correlation between the size of the fall and the later recovery is **%.2f**, which is close to zero. "
                "A large fall did not reliably predict a large recovery. **%d sites** have at least one consecutive monthly swing above 50%% and are marked for review before the change is treated as operational growth or decline."
                % (
                    insights["shock_sites_declining"], insights["shock_sites_total"],
                    insights["recovery_sites_rising"], insights["recovery_sites_total"],
                    insights["shock_recovery_correlation"], insights["regime_shift_review_sites"],
                )
            ),
        },
        {"id": "shock_recovery_block", "type": "chart", "chartId": "shock_recovery", "layout": "full"},
        {"id": "shock_rank_block", "type": "chart", "chartId": "shock_rank", "layout": "half"},
        {"id": "latest_extremes_block", "type": "chart", "chartId": "latest_extremes", "layout": "half"},
        {"id": "month_heatmap_block", "type": "chart", "chartId": "month_heatmap", "layout": "full"},
        {"id": "intersection_table_block", "type": "table", "tableId": "intersection_detail", "layout": "full"},
        {
            "id": "shape_section", "type": "markdown",
            "body": "## 3. Weekly shape and inbound approaches\n\nVolume says how busy a site is. Shape says when it is busy and which entry direction carries the traffic.",
        },
        {
            "id": "weekly_findings", "type": "markdown", "sourceId": "weekly_profile",
            "body": (
                "The highest network hour-of-week average is **%s at %02d:00**, at about **%.0f vehicles per 15 minutes** across the aggregated intersection profile. "
                "Friday's average hourly profile is **%.1f%% lower** than the average of the other six days."
                % (
                    insights["weekly_peak_weekday"], insights["weekly_peak_hour"],
                    insights["weekly_peak_mean_volume"], 100 * abs(insights["friday_mean_vs_other_days"]),
                )
            ),
        },
        {
            "id": "approach_1081_findings", "type": "markdown", "sourceId": "approach_analysis",
            "body": (
                "Intersection 1081 fell **%.1f%%** from February to March 2020 and rose **%.1f%%** from April to May. "
                "North and south approaches make up **%.1f%%** of its four-direction mean entry volume."
                % (
                    100 * insights["intersection_1081"]["feb_to_mar_2020_change"],
                    100 * insights["intersection_1081"]["apr_to_may_2020_change"],
                    100 * insights["intersection_1081"]["north_south_corridor_share"],
                )
            ),
        },
        {"id": "hour_week_block", "type": "chart", "chartId": "hour_week_heatmap", "layout": "full"},
        {"id": "approach_1081_block", "type": "chart", "chartId": "approaches_1081", "layout": "full"},
        {"id": "approach_shares_block", "type": "chart", "chartId": "approach_shares", "layout": "full"},
        {
            "id": "graph_section", "type": "markdown", "sourceId": "graph_analysis",
            "body": (
                "## 4. Functional graph\n\n"
                "The graph contains **%d intersections**, **%d similarity edges**, and **%d communities**. "
                "An edge means two weekly traffic profiles are similar. It does not mean the roads touch."
                % (len(datasets["community_nodes"]), summary["functional_graph"]["edges"], len(datasets["community_summary"]))
            ),
        },
        {
            "id": "graph_findings", "type": "markdown", "sourceId": "graph_analysis",
            "body": (
                "Traffic volume and functional centrality have a Spearman correlation of **%.2f**, which is close to zero. "
                "Busy intersections and pattern-central intersections are different operational groups. Intersection 1081 is a good example: it ranks **%d of %d by volume** but **%d by functional centrality**."
                % (
                    insights["volume_centrality_spearman"],
                    insights["intersection_1081"]["volume_rank"],
                    insights["intersection_1081"]["ranked_intersections"],
                    insights["intersection_1081"]["functional_similarity_centrality_rank"],
                )
            ),
        },
        {"id": "centrality_block", "type": "chart", "chartId": "centrality_scatter", "layout": "full"},
        {"id": "community_size_block", "type": "chart", "chartId": "community_sizes", "layout": "half"},
        {"id": "community_table_block", "type": "table", "tableId": "community_detail", "layout": "half"},
        {
            "id": "map_section", "type": "markdown",
            "body": "## 5. Spatial view\n\nOnly medium- and high-confidence coordinate candidates are shown. The separate interactive Isfahan map remains available in the project output folder.",
        },
        {"id": "spatial_block", "type": "chart", "chartId": "spatial_scatter", "layout": "full"},
        {
            "id": "quality_section", "type": "markdown", "sourceId": "quality_audit",
            "body": (
                "## 6. Data quality and unusual days\n\n"
                "About **%.1f%% of profiled channel readings are unavailable**. They remain missing and are never randomly filled. "
                "November 2020 is missing and December 2020 is partial."
                % (100 * summary["quality"]["invalid_channel_rate"])
            ),
        },
        {
            "id": "context_findings", "type": "markdown", "sourceId": "daily_context",
            "body": (
                "Nowruz days average **%.1f index points lower** than matched comparison days, official holidays average **%.1f points lower**, and wet-weather days average **%.1f points lower**. "
                "Friday is not included in this adjusted-factor chart because Friday is itself a weekday; its lower profile is shown in the weekly heatmap."
                % (
                    abs(insights["nowruz_adjusted_difference_points"]),
                    abs(insights["holiday_adjusted_difference_points"]),
                    abs(insights["wet_weather_adjusted_difference_points"]),
                )
            ),
        },
        {"id": "coverage_block", "type": "chart", "chartId": "source_coverage", "layout": "full"},
        {"id": "trusted_months_block", "type": "chart", "chartId": "trusted_months", "layout": "half"},
        {"id": "source_quality_table_block", "type": "table", "tableId": "source_quality_detail", "layout": "half"},
        {"id": "anomaly_table_block", "type": "table", "tableId": "anomaly_detail", "layout": "full"},
        {
            "id": "limits", "type": "markdown",
            "body": (
                "## Limits\n\n"
                "- Detector groups measure **entries near the stop line**, not exits or turning paths.\n"
                "- Functional edges describe similar weekly patterns, not road links.\n"
                "- Map points need municipal GIS review.\n"
                "- Weather, calendar, event, and COVID comparisons are associations, not causal estimates.\n"
                "- A missing or partial source month is not interpreted as a real traffic fall."
            ),
        },
    ]

    artifact = {
        "surface": "dashboard",
        "manifest": {
            "version": 1,
            "surface": "dashboard",
            "title": "Isfahan Intersection Traffic Dashboard",
            "description": "Deep traffic-change, weekly-pattern, approach, graph, map, context, and quality analysis for Isfahan SCATS intersections.",
            "generatedAt": generated_at,
            "filters": [
                {
                    "id": "community_filter", "label": "Functional community",
                    "dataset": "community_nodes", "field": "community_label", "includeAll": True,
                    "targets": [
                        {"dataset": "community_nodes", "field": "community_label"},
                        {"dataset": "map_points", "field": "community_label"},
                        {"dataset": "shock_recovery", "field": "community_label"},
                        {"dataset": "intersection_summary", "field": "community_label"},
                    ],
                }
            ],
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {name: _records(frame) for name, frame in datasets.items()},
            "accessIssues": [],
        },
        "sources": sources,
        "package_info": {
            "originUrl": "artifact://isfahan-intersection-traffic-dashboard",
            "controls": {"edit": False, "refresh": False},
        },
    }
    artifact_path = DASHBOARD_DIR / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "Dashboard artifact: %d charts, %d tables, %d datasets; validation=%s"
        % (len(charts), len(tables), len(datasets), "PASS" if checks["passed"] else "CHECK")
    )
    return artifact_path
