from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "outputs" / "report"


def records(frame: pd.DataFrame):
    #round trip through JSON so pandas missing values become null
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def source(source_id, label, path, description, code, tables_used, filters, definitions):
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "Python/pandas",
            "sql": code,
            "description": description,
            "tables_used": tables_used,
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


def main():
    summary = json.loads((TABLES / "analysis_summary.json").read_text(encoding="utf-8"))
    map_summary = json.loads((TABLES / "map_summary.json").read_text(encoding="utf-8"))

    city = pd.read_csv(TABLES / "city_monthly.csv")
    city["source_month_status"] = "complete"
    city.loc[city["year_month"] == "2020-12", "source_month_status"] = "partial"

    season = pd.read_csv(TABLES / "month_of_year_profile.csv")
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    season["month_name"] = season["month"].map(month_names)
    season["baseline_year"] = 2019
    season["common_panel_intersections"] = summary["time_series"]["month_of_year_common_panel_intersections"]

    approach_wide = pd.read_csv(TABLES / "intersection_1081_approach_monthly.csv")
    approach_rows = []
    for row in approach_wide.to_dict(orient="records"):
        for direction in ("south", "west", "north", "east"):
            approach_rows.append({
                "year_month": row["year_month"],
                "intersection_id": int(row["intersection_id"]),
                "approach": direction.title(),
                "mean_volume": row[direction + "_mean"],
                "observations": row[direction + "_observations"],
                "coverage": row[direction + "_coverage"],
            })
    approaches = pd.DataFrame(approach_rows)

    graph_nodes = pd.read_csv(TABLES / "functional_graph_nodes.csv")
    graph_nodes["intersection_id"] = graph_nodes["intersection_id"].astype(int)
    graph_nodes["display_name"] = graph_nodes["name_fa"].fillna("").astype(str)
    graph_nodes.loc[graph_nodes["display_name"].eq(""), "display_name"] = "SCATS " + graph_nodes["intersection_id"].astype(str)
    top_nodes = graph_nodes.sort_values("weighted_degree", ascending=False).head(12)[[
        "intersection_id", "display_name", "community", "weighted_degree",
        "betweenness_centrality", "mean_volume", "trusted_months",
    ]]
    communities = graph_nodes.groupby("community", as_index=False).agg(
        intersections=("intersection_id", "count"),
        mean_volume=("mean_volume", "mean"),
        mean_weighted_degree=("weighted_degree", "mean"),
        median_trusted_months=("trusted_months", "median"),
    )

    event_changes = pd.DataFrame([
        {
            "period": "Feb to Mar 2020",
            "paired_intersections": summary["time_series"]["largest_consecutive_month_decline"]["paired_intersections"],
            "median_change": summary["time_series"]["largest_consecutive_month_decline"]["median_change"],
            "mean_change": summary["time_series"]["largest_consecutive_month_decline"]["mean_change"],
            "p25_change": summary["time_series"]["largest_consecutive_month_decline"]["p25_change"],
            "p75_change": summary["time_series"]["largest_consecutive_month_decline"]["p75_change"],
        },
        {
            "period": "Apr to May 2020",
            "paired_intersections": summary["time_series"]["largest_consecutive_month_increase"]["paired_intersections"],
            "median_change": summary["time_series"]["largest_consecutive_month_increase"]["median_change"],
            "mean_change": summary["time_series"]["largest_consecutive_month_increase"]["mean_change"],
            "p25_change": summary["time_series"]["largest_consecutive_month_increase"]["p25_change"],
            "p75_change": summary["time_series"]["largest_consecutive_month_increase"]["p75_change"],
        },
    ])

    quality = pd.read_csv(TABLES / "source_month_quality.csv")
    quality["status"] = "complete"
    quality.loc[quality["source_month"] == "2020-12", "status"] = "partial"
    #add missing November to the audit so the gap stays visible
    missing_row = pd.DataFrame([{
        "source_file": None,
        "source_month": "2020-11",
        "first_timestamp": None,
        "last_timestamp": None,
        "timestamps_observed": 0,
        "timestamps_expected_full_month": 2880,
        "source_temporal_coverage": 0.0,
        "intersection_records": 0,
        "records_with_normalized_volume": 0,
        "intersections": 0,
        "output_file": None,
        "status": "missing",
    }])
    quality = pd.concat([quality, missing_row], ignore_index=True).sort_values("source_month")

    headlines = pd.DataFrame([{
        "raw_records": summary["data_window"]["raw_intersection_records"],
        "raw_intersections": summary["data_window"]["intersections_with_raw_data"],
        "invalid_channel_rate": summary["quality"]["invalid_channel_rate"],
        "march_change": summary["time_series"]["largest_consecutive_month_decline"]["median_change"],
        "march_pair_count": summary["time_series"]["largest_consecutive_month_decline"]["paired_intersections"],
        "mapped_locations": map_summary["located"],
        "map_coverage": map_summary["coverage"],
    }])

    analysis_source = source(
        "analysis_outputs",
        "Reviewed traffic-analysis tables",
        "outputs/tables/",
        "Reviewed monthly, seasonal, approach, and functional-network outputs produced by the project pipeline.",
        "SELECT * FROM read_csv_auto('outputs/tables/<table>.csv', header = true);",
        [
            "city_monthly.csv", "month_of_year_profile.csv",
            "intersection_1081_approach_monthly.csv", "functional_graph_nodes.csv",
        ],
        [
            "Trusted intersection-months meet configured detector and temporal coverage thresholds.",
            "Seasonality uses the 34-intersection common panel present in every month of 2019.",
        ],
        [
            "Coverage-normalized intersection volume = observed valid detector sum / valid operational detector share.",
            "Functional similarity edge weight = similarity of standardized 168-bin hour-of-week profiles.",
            "Monthly network statistic = median across trusted intersections, limiting domination by the busiest sites.",
        ],
    )
    quality_source = source(
        "quality_outputs",
        "SCATS parser and source-month audit",
        "outputs/tables/source_month_quality.csv",
        "File-level temporal coverage and detector availability audit from the raw SCATS archives.",
        "SELECT * FROM read_csv_auto('outputs/tables/source_month_quality.csv', header = true);",
        ["source_month_quality.csv", "analysis_summary.json"],
        ["Sentinel values 2046, 2047, and NA are unavailable; zero remains a valid count."],
        [
            "Unavailable-channel rate = unavailable channel readings / all profiled channel readings.",
            "Source temporal coverage = distinct observed 15-minute timestamps / timestamps expected for that calendar month.",
        ],
    )
    map_source = source(
        "map_outputs",
        "Cached OpenStreetMap/Nominatim candidates",
        "data/metadata/geocode_cache.csv",
        "Candidate intersection coordinates obtained from named direct matches or high-confidence street-pair matches.",
        "SELECT * FROM read_csv_auto('data/metadata/geocode_cache.csv', header = true);",
        ["intersection registry workbook", "geocode_cache.csv"],
        ["Only direct named matches and high-confidence street-pair matches are plotted."],
        ["Map coverage = reliable candidate locations / graph-eligible intersections."],
    )

    title = "Isfahan Intersection Traffic: Functional Graph Mining and Time-Series Analysis"
    manifest_sources = [analysis_source, quality_source, map_source]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "A reproducible technical analysis of Isfahan SCATS detector volumes from October 2018 through January 2021.",
            "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "filters": [],
            "cards": [
                {
                    "id": "records_card",
                    "description": "Parsed intersection-by-15-minute records across the available source archive.",
                    "dataset": "headlines",
                    "sourceId": "quality_outputs",
                    "metrics": [{"label": "Raw records", "field": "raw_records", "format": "number"}],
                },
                {
                    "id": "quality_card",
                    "description": "Share of all profiled channel readings marked unavailable by SCATS sentinels or NA.",
                    "dataset": "headlines",
                    "sourceId": "quality_outputs",
                    "metrics": [{"label": "Unavailable readings", "field": "invalid_channel_rate", "format": "percent"}],
                },
                {
                    "id": "march_card",
                    "description": "Median paired-intersection change from February to March 2020; descriptive, not causal.",
                    "dataset": "headlines",
                    "sourceId": "analysis_outputs",
                    "metrics": [
                        {"label": "Feb-Mar 2020", "field": "march_change", "format": "percent", "signed": True},
                        {"label": "Paired sites", "field": "march_pair_count", "format": "number"},
                    ],
                },
                {
                    "id": "map_card",
                    "description": "High-confidence coordinate candidates ready for validation against municipal GIS.",
                    "dataset": "headlines",
                    "sourceId": "map_outputs",
                    "metrics": [
                        {"label": "Mapped candidates", "field": "mapped_locations", "format": "number"},
                        {"label": "Coverage", "field": "map_coverage", "format": "percent"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "city_monthly_chart",
                    "title": "Typical intersection volume by month",
                    "subtitle": "Median across trusted intersections; October 2018 to January 2021",
                    "headerMarkdown": "Values are **vehicles per 15-minute interval**, normalized for available operational detectors.",
                    "type": "line",
                    "dataset": "city_monthly",
                    "sourceId": "analysis_outputs",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "year_month", "type": "temporal", "label": "Month"},
                        "y": {"field": "median_intersection_volume", "type": "quantitative", "label": "Median vehicles / 15 min"},
                        "tooltip": [
                            {"field": "intersections", "type": "quantitative", "label": "Trusted intersections", "format": "number"},
                            {"field": "p25_intersection_volume", "type": "quantitative", "label": "25th percentile", "format": "number"},
                            {"field": "p75_intersection_volume", "type": "quantitative", "label": "75th percentile", "format": "number"},
                            {"field": "source_month_status", "type": "nominal", "label": "Source status"},
                        ],
                    },
                },
                {
                    "id": "seasonality_chart",
                    "title": "2019 month-of-year baseline",
                    "subtitle": "Same 34 intersections in all 12 months",
                    "headerMarkdown": "A common panel isolates within-panel seasonality from changes in which intersections report.",
                    "type": "bar",
                    "dataset": "seasonality",
                    "sourceId": "analysis_outputs",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "month_name", "type": "nominal", "label": "Month"},
                        "y": {"field": "median_volume", "type": "quantitative", "label": "Median vehicles / 15 min"},
                        "tooltip": [
                            {"field": "mean_volume", "type": "quantitative", "label": "Mean", "format": "number"},
                            {"field": "standard_deviation", "type": "quantitative", "label": "Std. dev.", "format": "number"},
                            {"field": "common_panel_intersections", "type": "quantitative", "label": "Intersections", "format": "number"},
                        ],
                    },
                },
                {
                    "id": "approach_1081_chart",
                    "title": "Intersection 1081 by inbound approach",
                    "subtitle": "Detector-channel groups inferred from the junction diagram and checked against workbook formulas",
                    "headerMarkdown": "North and south approaches dominate. These are **entry-side detector counts**, not destination-side exit counts.",
                    "type": "line",
                    "dataset": "approaches_1081",
                    "sourceId": "analysis_outputs",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "year_month", "type": "temporal", "label": "Month"},
                        "y": {"field": "mean_volume", "type": "quantitative", "label": "Mean vehicles / 15 min"},
                        "color": {"field": "approach", "type": "nominal", "label": "Inbound approach"},
                        "tooltip": [
                            {"field": "coverage", "type": "quantitative", "label": "Detector coverage", "format": "percent"},
                            {"field": "observations", "type": "quantitative", "label": "15-min observations", "format": "number"},
                        ],
                    },
                },
            ],
            "tables": [
                {
                    "id": "event_changes_table",
                    "title": "Largest paired month-to-month movements",
                    "subtitle": "Changes are computed only for the same trusted intersections in adjacent months.",
                    "dataset": "event_changes",
                    "sourceId": "analysis_outputs",
                    "columns": [
                        {"field": "period", "label": "Period", "type": "text"},
                        {"field": "paired_intersections", "label": "Paired sites", "format": "number"},
                        {"field": "median_change", "label": "Median change", "format": "percent", "movement": True},
                        {"field": "p25_change", "label": "25th percentile", "format": "percent", "movement": True},
                        {"field": "p75_change", "label": "75th percentile", "format": "percent", "movement": True},
                    ],
                },
                {
                    "id": "top_nodes_table",
                    "title": "Most central intersections in the functional-similarity graph",
                    "subtitle": "Centrality reflects similar weekly traffic signatures, not physical road adjacency.",
                    "dataset": "top_nodes",
                    "sourceId": "analysis_outputs",
                    "defaultSort": {"field": "weighted_degree", "direction": "desc"},
                    "columns": [
                        {"field": "intersection_id", "label": "SCATS ID", "format": "number"},
                        {"field": "display_name", "label": "Intersection", "type": "text"},
                        {"field": "community", "label": "Community", "format": "number"},
                        {"field": "weighted_degree", "label": "Weighted degree", "format": "number"},
                        {"field": "betweenness_centrality", "label": "Betweenness", "format": "number"},
                        {"field": "mean_volume", "label": "Mean vehicles / 15 min", "format": "number"},
                    ],
                },
                {
                    "id": "quality_table",
                    "title": "Source-month coverage audit",
                    "subtitle": "November 2020 is missing; December 2020 is partial.",
                    "dataset": "source_quality",
                    "sourceId": "quality_outputs",
                    "defaultSort": {"field": "source_month", "direction": "asc"},
                    "columns": [
                        {"field": "source_month", "label": "Month", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                        {"field": "source_temporal_coverage", "label": "Temporal coverage", "format": "percent"},
                        {"field": "intersections", "label": "Intersections", "format": "number"},
                        {"field": "intersection_records", "label": "Raw records", "format": "number"},
                    ],
                },
            ],
            "sources": manifest_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# " + title},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "analysis_outputs",
                    "body": (
                        "## Technical Summary\n\n"
                        "The archive supports a functional traffic network and monthly operating profile, but not a vehicle-routing graph. "
                        "The strongest system-wide movement is a **25.8% median decline from February to March 2020** across 41 paired intersections, followed by a **24.4% median increase from April to May** across 43 sites. "
                        "For Intersection 1081, north and south inbound approaches carry **68.9%** of the four-approach mean volume. "
                        "A 74-node weekly-profile similarity graph forms four operational communities; Intersection 1081 ranks fifth by weighted similarity centrality despite ranking 61st of 74 by average volume."
                    ),
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["records_card", "quality_card", "march_card", "map_card"]},
                {
                    "id": "key_findings",
                    "type": "markdown",
                    "sourceId": "analysis_outputs",
                    "body": (
                        "## Key Findings\n\n"
                        "The results point to three distinct stories: a sharp early-2020 system shift, modest pre-pandemic seasonality, and corridor-specific imbalance at Intersection 1081. "
                        "All volume comparisons below use coverage-normalized counts and trusted intersection-months."
                    ),
                },
                {"id": "city_monthly_block", "type": "chart", "chartId": "city_monthly_chart"},
                {
                    "id": "city_monthly_note",
                    "type": "markdown",
                    "sourceId": "analysis_outputs",
                    "body": (
                        "### System movement\n\n"
                        "The paired-site calculation avoids confusing changes in reporting coverage with real traffic change. "
                        "The February-to-March 2020 interquartile range was **-30.3% to -23.2%**, so the decline was broad rather than driven by a handful of intersections. "
                        "The rebound is also broad, with an April-to-May interquartile range of **+21.6% to +27.8%**. These are temporal associations; the dataset alone does not identify a cause."
                    ),
                },
                {"id": "event_changes_block", "type": "table", "tableId": "event_changes_table"},
                {"id": "seasonality_block", "type": "chart", "chartId": "seasonality_chart"},
                {
                    "id": "seasonality_note",
                    "type": "markdown",
                    "sourceId": "analysis_outputs",
                    "body": (
                        "### Calendar pattern\n\n"
                        "Within the 2019 common panel, January has the highest typical level (**546.5 vehicles per 15 minutes**) and November the lowest (**500.2**), a **9.3%** peak-to-trough spread. "
                        "This pattern is a baseline, not a universal seasonal law: only one complete pre-pandemic calendar year is available."
                    ),
                },
                {"id": "approach_block", "type": "chart", "chartId": "approach_1081_chart"},
                {
                    "id": "approach_note",
                    "type": "markdown",
                    "sourceId": "analysis_outputs",
                    "body": (
                        "### Intersection 1081\n\n"
                        "The verified channel grouping is south 1-3, west 4-6, north 7-9, and east 10-12. "
                        "Mean inbound counts are approximately south **150.0**, west **60.4**, north **152.3**, and east **76.0 vehicles per 15 minutes**. "
                        "That imbalance suggests that north-south timing plans and queue observations deserve priority, while the high functional-centrality rank makes 1081 a useful representative monitoring site."
                    ),
                },
                {
                    "id": "graph_findings",
                    "type": "markdown",
                    "sourceId": "analysis_outputs",
                    "body": (
                        "### Functional graph\n\n"
                        "The graph contains **74 nodes, 218 similarity edges, four communities, and one connected component**. "
                        "Edges connect intersections with similar standardized 168-bin hour-of-week profiles; they do not mean that roads connect the sites or that vehicles traveled between them. "
                        "The two-cluster DTW solution has silhouette 0.569 but mostly isolates three outliers (1017, 1031, and 1032), so the four graph communities are more useful for operational peer groups."
                    ),
                },
                {"id": "top_nodes_block", "type": "table", "tableId": "top_nodes_table"},
                {
                    "id": "scope_data_metrics",
                    "type": "markdown",
                    "sourceId": "quality_outputs",
                    "body": (
                        "## Scope, Data, and Metrics\n\n"
                        "The source window is October 2018 through January 2021 and contains **5,454,089** intersection records, **76,422** distinct 15-minute timestamps, and 75 raw intersection IDs. "
                        "Of 57,594,780 channel readings, **20.2%** are unavailable sentinels or NA. Zero is retained as a valid detector count. "
                        "The parser never uses random imputation; monthly volumes are normalized for the fraction of operational detectors observed."
                    ),
                },
                {"id": "quality_block", "type": "table", "tableId": "quality_table"},
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "analysis_outputs",
                    "body": (
                        "## Methodology\n\n"
                        "1. Parse the raw SCATS text defensively, keeping zero and mapping 2046, 2047, and NA to unavailable.\n"
                        "2. Detect numbered channel boxes in the junction diagrams and apply reviewed manual overrides for 1029, 1081, 1092, and 1109.\n"
                        "3. Aggregate valid detector counts to inbound approaches and coverage-normalized intersection volumes.\n"
                        "4. Build trusted monthly, weekday-hour, and 168-bin weekly profiles.\n"
                        "5. Create a k-nearest-neighbor functional graph from standardized weekly-profile similarity, then compute communities and centralities.\n"
                        "6. Cluster dynamic time-warping distances as a sensitivity check.\n"
                        "7. Geocode named intersections with cached, rate-limited OpenStreetMap/Nominatim candidates and plot only direct or high-confidence street-pair matches."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations, Uncertainty, and Robustness\n\n"
                        "- Detector channels measure inbound approaches near the stop line. Direct exits, turning movements, origin-destination flows, and physical road links require additional topology or movement data.\n"
                        "- November 2020 is absent and December 2020 is only 20.3% complete; neither should be used as an ordinary full-month comparison.\n"
                        "- The 2019 seasonal profile uses only 34 intersections with complete trusted coverage across all 12 months.\n"
                        "- Early-2020 changes are descriptive associations, not causal estimates. Weather, holidays, incidents, signal plans, and policy covariates were not joined.\n"
                        "- Only 30 of 74 graph-eligible sites have reliable geocoding candidates. All coordinates should be checked against municipal GIS before operational use.\n"
                        "- Automated channel-to-approach assignments outside the manually reviewed examples remain hypotheses until verified diagram by diagram."
                    ),
                },
                {
                    "id": "recommendations",
                    "type": "markdown",
                    "sourceId": "analysis_outputs",
                    "body": (
                        "## Recommended Next Steps\n\n"
                        "1. Validate the 30 mapped candidates in QGIS against the municipality's authoritative junction layer, then add coordinates for the remaining sites.\n"
                        "2. Manually review approach-channel mappings for the highest-volume and highest-centrality intersections before using them for signal timing.\n"
                        "3. Use the four functional communities as peer groups for anomaly alerts, detector health monitoring, and timing-plan comparisons.\n"
                        "4. At 1081, compare north-south green time, queues, and saturation with east-west performance; the count imbalance alone is not enough to prescribe a split.\n"
                        "5. Add holidays, weather, incidents, road works, and policy dates for interrupted time-series or causal analysis.\n"
                        "6. Obtain turn-count or trajectory data before building a physical flow or origin-destination graph."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further Questions\n\n"
                        "- Which SCATS channels are loops, stop-line detectors, or movement-specific lanes at each site?\n"
                        "- Can municipal GIS provide authoritative coordinates and road topology for all 75 IDs?\n"
                        "- Which timing-plan changes, holidays, closures, or enforcement interventions align with the detected shifts?\n"
                        "- Do the functional communities remain stable when profiles are split by season, weekday/weekend, or peak period?\n"
                        "- Can queue length, occupancy, travel time, or floating-car data be joined to distinguish demand from signal-control effects?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "ready",
            "datasets": {
                "headlines": records(headlines),
                "city_monthly": records(city),
                "seasonality": records(season),
                "approaches_1081": records(approaches),
                "event_changes": records(event_changes),
                "top_nodes": records(top_nodes),
                "community_summary": records(communities),
                "source_quality": records(quality),
            },
            "accessIssues": [],
        },
        "sources": manifest_sources,
        "package_info": {
            "originUrl": "artifact://isfahan-intersection-traffic-analysis",
            "controls": {"edit": False, "refresh": False},
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "artifact.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
