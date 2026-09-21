from __future__ import annotations

import json
from pathlib import Path

from plotly.offline import get_plotlyjs

from .paths import PROJECT_ROOT


ACCENT = "#217568"
NEGATIVE = "#bd634e"
COLORS = [ACCENT, "#c89445", "#627bb0", "#9b6e91"]


def _months(first: str, last: str) -> list[str]:
    year, month = map(int, first.split("-"))
    result = []
    while "%04d-%02d" % (year, month) <= last:
        result.append("%04d-%02d" % (year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return result


def _layout(x: str, y: str, **extra) -> dict:
    layout = {
        "autosize": True,
        "height": 430,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Arial, sans-serif", "size": 12, "color": "#31443e"},
        "margin": {"l": 62, "r": 24, "t": 28, "b": 70},
        "xaxis": {"title": {"text": x}, "automargin": True, "showgrid": False},
        "yaxis": {"title": {"text": y}, "automargin": True, "gridcolor": "#e4eae6", "zerolinecolor": "#c0ccc4"},
        "legend": {"orientation": "h", "y": 1.16, "x": 0},
        "hoverlabel": {"bgcolor": "#ffffff", "font": {"color": "#263d34"}},
        "colorway": COLORS,
    }
    layout.update(extra)
    return layout


def _chart(chart_id, section, title, summary, takeaway, read, limit, source, data, layout):
    return {
        "id": chart_id, "section": section, "title": title, "summary": summary,
        "takeaway": takeaway, "read": read, "limit": limit, "source": source,
        "figure": {"data": data, "layout": layout},
    }


def _developer_data(datasets: dict, validation: dict, settings: dict) -> dict:
    pipeline = [
        {
            "id": "metadata", "label": "1. Read records and detector metadata",
            "detail": "parser.py joins multi-line SCATS records. metadata.py matches the intersection registry, diagrams and channel history to build the detector map.",
            "inputs": ["SCATS monthly text exports", "intersection registry workbook", "junction diagrams"],
            "outputs": ["data/metadata/intersections.csv", "data/metadata/channel_quality.csv", "data/metadata/detector_channels.csv"],
        },
        {
            "id": "ingest", "label": "2. Build the 15-minute records",
            "detail": "ingest.py uses detector_channels.csv to group entry directions and adjust counts for the operational detectors that are available. Missing time intervals stay missing.",
            "inputs": ["SCATS monthly text exports", "data/metadata/detector_channels.csv", "config/settings.json"],
            "outputs": ["data/processed/15min/YYYY-MM.csv.gz", "outputs/tables/source_month_quality.csv"],
        },
        {
            "id": "analysis", "label": "3. Compare traffic and weekly patterns",
            "detail": "analysis.py checks monthly coverage, compares the same junctions between adjacent months and builds 168-hour weekly profiles. Similarity links and DTW groups compare pattern shape.",
            "inputs": ["data/processed/15min/YYYY-MM.csv.gz", "config/settings.json"],
            "outputs": ["outputs/tables/intersection_monthly.csv", "outputs/tables/paired_month_changes.csv", "outputs/tables/hourly_weekday_profile.csv", "outputs/tables/functional_graph_nodes.csv", "outputs/tables/functional_graph_edges.csv", "outputs/tables/analysis_summary.json"],
        },
        {
            "id": "context", "label": "4. Join calendar, weather and locations",
            "detail": "context.py joins dates to holidays, weather, policy and recorded events. geocode.py separately finds candidate locations for the map.",
            "inputs": ["outputs/tables/intersection_daily.csv", "data/external/raw/", "data/external/manual/", "data/metadata/intersections.csv"],
            "outputs": ["data/external/processed/isfahan_daily_context.csv", "outputs/tables/city_daily_with_context.csv", "outputs/tables/traffic_context_associations.csv", "data/metadata/intersections_geocoded.csv"],
        },
        {
            "id": "dashboard", "label": "5. Prepare the chart data and checks",
            "detail": "dashboard.py combines the reviewed tables into an artifact and records the checks below. report_ui.py selects eight views and embeds their data with Plotly in each HTML file.",
            "inputs": ["outputs/tables/", "data/metadata/intersections_geocoded.csv", "src/isfahan_traffic/web/traffic_explorer.html"],
            "outputs": ["outputs/dashboard/artifact.json", "outputs/tables/dashboard_validation.json", "outputs/tables/dashboard_insights.json", "outputs/dashboard/dashboard.html", "outputs/report/isfahan_traffic_analysis.html"],
        },
    ]
    rule_definitions = [
        ("minimum_detector_coverage", "Minimum detector share for a 15-minute count", "ingest.py leaves the normalized count missing below this share of operational detector readings", True),
        ("trusted_detector_coverage", "Detector coverage for a usable month", "analysis.py checks the average detector coverage before including an intersection-month", True),
        ("trusted_temporal_coverage", "Time coverage for a usable month", "analysis.py checks usable 15-minute counts against the full calendar month", True),
        ("functional_graph_neighbors", "Nearest pattern links per junction", "analysis.py considers up to this many strongest qualifying neighbors before merging links into an undirected graph", False),
        ("functional_graph_min_correlation", "Minimum weekly pattern correlation", "analysis.py keeps candidate links only when their profile correlation meets this threshold", False),
    ]
    rules = [
        {"name": label, "value": ("%.0f%%" % (settings[key] * 100) if percent else str(settings[key])), "detail": detail}
        for key, label, detail, percent in rule_definitions if key in settings
    ]
    rules.extend([
        {"name": "Missing detector codes", "value": "NA, 2046, 2047", "detail": "parser.py converts these values to missing readings"},
        {"name": "Count adjustment", "value": "valid count sum / available operational detector share", "detail": "ingest.py adjusts detector coverage within a recorded interval without filling missing intervals"},
        {"name": "Paired monthly change", "value": "later volume / earlier volume - 1", "detail": "analysis.py calculates this for each junction present in both months and reports the median change"},
        {"name": "Weekly profile", "value": "7 days × 24 hours = 168 bins", "detail": "analysis.py standardizes each junction's profile before comparing its shape with other junctions"},
    ])
    return {
        "pipeline": pipeline, "rules": rules,
        "datasets": [{"id": name, "count": len(rows), "fields": list(rows[0]) if rows else [], "preview": rows[:3]} for name, rows in datasets.items()],
        "validation": validation,
    }


def build_explorer_payload(artifact, summary, insights, validation=None, settings=None) -> dict:
    datasets = artifact["snapshot"]["datasets"]
    months = _months(summary["data_window"]["first_month"], summary["data_window"]["last_month"])
    dates = [month + "-01" for month in months]
    quality = {row["source_month"]: row for row in datasets["source_quality"]}
    status = {month: quality.get(month, {}).get("status", "Unknown") for month in months}
    headlines = datasets["headline"][0]
    charts = {}

    #keep missing calendar months visible after dashboard.py prepares the data
    monthly = {row["year_month"]: row for row in datasets["city_monthly"]}
    monthly_y = [monthly.get(month, {}).get("median_intersection_volume") for month in months]
    monthly_layout = _layout("Month", "Vehicles per 15 minutes")
    monthly_layout["xaxis"].update({"type": "date", "dtick": "M4", "tickformat": "%b<br>%Y"})
    monthly_layout["yaxis"]["rangemode"] = "tozero"
    monthly_layout["shapes"] = [{"type": "rect", "xref": "x", "yref": "paper", "x0": "2020-11-01", "x1": "2020-12-31", "y0": 0, "y1": 1, "fillcolor": "#eee3d4", "opacity": 0.55, "line": {"width": 0}, "layer": "below"}]
    charts["monthly"] = _chart(
        "monthly", "city", "Monthly traffic", "Follow the main changes across the whole archive.",
        "Traffic fell in March 2020 and rose again in May. The gap near the end is missing or incomplete data.",
        "Each point is the middle value across junctions with enough usable data that month. Hover to see how many junctions were included.",
        "The junctions included can change each month. Use Changes between months to compare the same junctions. Counts show entries near the stop line, not all trips in the city.",
        "city_monthly.csv · median of junction means after the monthly coverage checks",
        [{"type": "scatter", "mode": "lines+markers", "name": "Typical junction", "x": dates, "y": monthly_y, "connectgaps": False,
          "line": {"color": ACCENT, "width": 3}, "marker": {"size": 6},
          "customdata": [[monthly.get(month, {}).get("intersections"), status[month]] for month in months],
          "hovertemplate": "%{x|%b %Y}<br>%{y:.1f} vehicles / 15 min<br>%{customdata[0]:.0f} usable junctions<br>Source: %{customdata[1]}<extra></extra>"}], monthly_layout,
    )

    changes = {row["year_month"]: row for row in datasets["paired_changes"]}
    change_y = [changes.get(month, {}).get("median_change") for month in months]
    change_layout = _layout("Later month", "Change from the previous month")
    change_layout["xaxis"].update({"type": "date", "dtick": "M4", "tickformat": "%b<br>%Y"})
    change_layout["yaxis"]["tickformat"] = ".0%"
    charts["changes"] = _chart(
        "changes", "city", "Changes between months", "Compare the same junctions before and after each change.",
        "The largest fall was %.1f%% from February to March 2020. The largest rise was %.1f%% from April to May." % (abs(headlines["shock_change"]) * 100, headlines["recovery_change"] * 100),
        "Bars below zero show a fall; bars above zero show a rise. Each bar is the middle change among junctions with usable data in both months.",
        "Different bars can include different junctions. A blank month means a comparison is unavailable, not that traffic stayed the same.",
        "paired_month_changes.csv · adjacent months matched by intersection_id",
        [{"type": "bar", "x": dates, "y": change_y,
          "marker": {"color": [NEGATIVE if value is not None and value < 0 else ACCENT for value in change_y]},
          "customdata": [[changes.get(month, {}).get("period_label"), changes.get(month, {}).get("paired_intersections")] for month in months],
          "hovertemplate": "%{customdata[0]}<br>%{y:.1%} median change<br>%{customdata[1]:.0f} matched junctions<extra></extra>"}], change_layout,
    )

    weekly = sorted(datasets["hour_week_heatmap"], key=lambda row: row["weekday"])
    weekday_names = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"}
    weekly_layout = _layout("Hour of day", "")
    weekly_layout["xaxis"].update({"tickmode": "array", "tickvals": list(range(0, 24, 3)), "ticktext": ["%02d:00" % hour for hour in range(0, 24, 3)]})
    weekly_layout["yaxis"].update({"autorange": "reversed", "showgrid": False})
    charts["weekly"] = _chart(
        "weekly", "patterns", "Busy days and hours", "See how traffic changes through the week.",
        "%s at %02d:00 has the highest average in this data. Friday's average is %.1f%% below the other six days." % (weekday_names.get(insights["weekly_peak_weekday"], insights["weekly_peak_weekday"]), insights["weekly_peak_hour"], abs(insights["friday_mean_vs_other_days"]) * 100),
        "Darker cells mean more vehicles. Each cell groups the available 15-minute counts for one hour and weekday.",
        "These averages cover the available records across the archive. They do not describe every junction or every week. The value is still vehicles per 15 minutes, not an hourly total.",
        "hourly_weekday_profile.csv · counts pooled by weekday and hour using their observation counts",
        [{"type": "heatmap", "x": list(range(24)), "y": [weekday_names[row["weekday_name"]] for row in weekly],
          "z": [[row.get("h%02d" % hour) for hour in range(24)] for row in weekly],
          "colorscale": [[0, "#f0f4eb"], [0.5, "#87b6a1"], [1, "#1d675b"]], "xgap": 2, "ygap": 2,
          "colorbar": {"title": {"text": "Vehicles<br>/ 15 min"}, "thickness": 12, "len": 0.85},
          "hovertemplate": "%{y}, %{x}:00<br>%{z:.1f} vehicles / 15 min<extra></extra>"}], weekly_layout,
    )

    seasons = sorted(datasets["seasonality"], key=lambda row: row["month"])
    peak = max(seasons, key=lambda row: row["median_volume"])
    low = min(seasons, key=lambda row: row["median_volume"])
    panel_count = summary["time_series"]["month_of_year_common_panel_intersections"]
    season_layout = _layout("Month in 2019", "Vehicles per 15 minutes")
    season_layout["yaxis"]["rangemode"] = "tozero"
    charts["seasonality"] = _chart(
        "seasonality", "patterns", "Months in 2019", "Keep the same junctions throughout the year.",
        "Among the same %d junctions, %s had the highest typical traffic and %s had the lowest. The difference was about %.1f%% of the lower value." % (panel_count, peak["month_name"], low["month_name"], (peak["median_volume"] / low["median_volume"] - 1) * 100),
        "Each bar shows the middle value across junctions that passed the coverage checks in all twelve months.",
        "This covers one year: 2019, the only complete year before the pandemic. It is not a long-term seasonal forecast.",
        "month_of_year_profile.csv · common panel present in all twelve months of 2019",
        [{"type": "bar", "x": [row["month_name"] for row in seasons], "y": [row["median_volume"] for row in seasons],
          "marker": {"color": [ACCENT if row["month"] == peak["month"] else "#93bbae" for row in seasons]},
          "customdata": [row["common_panel_intersections"] for row in seasons],
          "hovertemplate": "%{x} 2019<br>%{y:.1f} vehicles / 15 min<br>%{customdata:.0f} junctions in every month<extra></extra>"}], season_layout,
    )

    sites = datasets["shock_recovery"]
    site_layout = _layout("February to March 2020", "April to May 2020")
    site_layout["xaxis"]["tickformat"] = ".0%"
    site_layout["yaxis"]["tickformat"] = ".0%"
    site_layout["showlegend"] = False
    site_traces = []
    for highlighted in (False, True):
        selected = [row for row in sites if (row["intersection_id"] == 1081) == highlighted]
        site_traces.append({"type": "scatter", "mode": "markers", "name": "Junction 1081" if highlighted else "Junctions",
                            "x": [row["shock_change"] for row in selected], "y": [row["recovery_change"] for row in selected],
                            "marker": {"color": NEGATIVE if highlighted else ACCENT, "size": 13 if highlighted else 9, "opacity": 0.85, "line": {"width": 1, "color": "#ffffff"}},
                            "customdata": [[row["intersection_label"], row["trusted_months"]] for row in selected],
                            "hovertemplate": "%{customdata[0]}<br>Feb–Mar: %{x:.1%}<br>Apr–May: %{y:.1%}<br>%{customdata[1]} usable months<extra></extra>"})
    charts["sites"] = _chart(
        "sites", "intersections", "The fall and the recovery", "Find how individual junctions changed in 2020.",
        "A bigger March fall did not reliably mean a bigger May recovery. The relationship was weak across the %d junctions shown." % len(sites),
        "Each dot is a junction. Further left means a larger March fall; higher up means a larger May rise. Junction 1081 is shown in orange.",
        "The March comparison alone includes %d junctions and the May comparison includes %d. This chart uses the %d that appear in both. It does not prove why traffic changed." % (headlines["shock_sites"], headlines["recovery_sites"], len(sites)),
        "intersection_monthly.csv · matched February/March and April/May 2020 comparisons",
        site_traces, site_layout,
    )

    approaches = datasets["approaches_1081"]
    approach_traces = []
    #break the lines at missing months and show partial months as separate points
    for direction, color in zip(("South", "West", "North", "East"), COLORS):
        rows = {row["year_month"]: row for row in approaches if row["approach"] == direction}
        for partial in (False, True):
            values = [rows.get(month, {}).get("mean_volume") if (status[month] == "Partial") == partial and status[month] != "Missing" else None for month in months]
            approach_traces.append({
                "type": "scatter", "mode": "markers" if partial else "lines+markers", "name": direction + (" · partial month" if partial else ""),
                "legendgroup": direction, "showlegend": not partial, "x": dates, "y": values, "connectgaps": False,
                "line": {"color": color, "width": 2}, "marker": {"color": color, "size": 10 if partial else 4, "symbol": "diamond-open" if partial else "circle"},
                "customdata": [[rows.get(month, {}).get("coverage"), rows.get(month, {}).get("observations"), status[month]] for month in months],
                "hovertemplate": direction + " · %{x|%b %Y}<br>%{y:.1f} vehicles / 15 min<br>Detector coverage: %{customdata[0]:.1%}<br>%{customdata[1]} observations<br>Source: %{customdata[2]}<extra></extra>",
            })
    approach_layout = _layout("Month", "Vehicles per 15 minutes")
    approach_layout["xaxis"].update({"type": "date", "dtick": "M4", "tickformat": "%b<br>%Y"})
    approach_layout["yaxis"]["rangemode"] = "tozero"
    charts["approaches"] = _chart(
        "approaches", "intersections", "Where traffic enters junction 1081", "Follow the four entry directions over time.",
        "North and south account for about %.1f%% of the combined average entry counts at junction 1081." % (insights["intersection_1081"]["north_south_corridor_share"] * 100),
        "Each color is an entry direction. Open diamonds show December 2020, which only covers part of the month. The missing November is a gap.",
        "These are entry counts. They do not show where vehicles turn or leave. High detector coverage in December does not mean the whole month was recorded.",
        "intersection_1081_approach_monthly.csv · entry-side detector groups from the junction diagram",
        approach_traces, approach_layout,
    )

    nodes = datasets["community_nodes"]
    graph_traces = []
    group_layout = _layout("Average vehicles per 15 minutes", "Strength of pattern links")
    group_layout["showlegend"] = False
    for highlighted in (False, True):
        rows = [row for row in nodes if (row["intersection_id"] == 1081) == highlighted]
        graph_traces.append({"type": "scatter", "mode": "markers+text" if highlighted else "markers",
                             "name": "Junction 1081" if highlighted else "Junctions", "x": [row["mean_volume"] for row in rows], "y": [row["weighted_degree"] for row in rows],
                             "text": ["1081" for row in rows] if highlighted else [], "textposition": "top center",
                             "marker": {"color": NEGATIVE if highlighted else ACCENT, "size": 13 if highlighted else 8, "opacity": 0.85, "line": {"width": 1, "color": "#ffffff"}},
                             "customdata": [[row["intersection_label"], row["trusted_months"], row["community"]] for row in rows],
                             "hovertemplate": "%{customdata[0]}<br>%{x:.1f} vehicles / 15 min<br>Pattern-link strength: %{y:.2f}<br>%{customdata[1]} usable months<br>Pattern group %{customdata[2]}<extra></extra>"})
    focus = insights["intersection_1081"]
    charts["groups"] = _chart(
        "groups", "intersections", "Busy junctions and similar patterns", "Compare how busy a junction is with how closely its weekly pattern matches others.",
        "Junction 1081 ranks %d of %d by traffic volume, but %d by the strength of its pattern links." % (focus["volume_rank"], focus["ranked_intersections"], focus["functional_similarity_centrality_rank"]),
        "Further right means more traffic. Higher up means stronger combined links to junctions with similar weekly traffic patterns. Junction 1081 is shown in orange.",
        "These links compare traffic patterns. They are not road connections, and a high value does not mean a junction is physically central in the city.",
        "functional_graph_nodes.csv · weighted degree from correlations between standardized weekly profiles",
        graph_traces, group_layout,
    )

    coverage_layout = _layout("Source month", "Available timestamps")
    coverage_layout["showlegend"] = False
    coverage_layout["xaxis"].update({"type": "date", "dtick": "M4", "tickformat": "%b<br>%Y"})
    coverage_layout["yaxis"].update({"tickformat": ".0%", "range": [0, 1.06]})
    december = quality.get("2020-12", {}).get("source_temporal_coverage")
    partial_text = "December contains about %.1f%% of the expected 15-minute timestamps." % (december * 100) if december is not None else "December is marked as partial in the source audit."
    charts["coverage"] = _chart(
        "coverage", "quality", "How much of each month is available", "Check the source gaps before comparing traffic.",
        "November 2020 has no source data. " + partial_text,
        "A full bar means timestamps cover nearly the whole month. Hover to see the source status and number of junctions in that file.",
        "This checks whether timestamps exist. A recorded timestamp can still contain missing detector readings, so detector coverage is checked separately.",
        "source_month_quality.csv · observed timestamps divided by expected calendar-month timestamps; the missing November row is added by dashboard.py",
        [{"type": "bar", "x": dates, "y": [quality.get(month, {}).get("source_temporal_coverage") for month in months],
          "marker": {"color": [NEGATIVE if status[month] in ("Missing", "Partial") else ACCENT for month in months]},
          "customdata": [[status[month], quality.get(month, {}).get("intersections"), quality.get(month, {}).get("timestamps_observed")] for month in months],
          "hovertemplate": "%{x|%b %Y}<br>%{y:.1%} timestamps available<br>%{customdata[0]}<br>%{customdata[1]} junctions<br>%{customdata[2]} timestamps<extra></extra>"},
         {"type": "scatter", "mode": "markers", "showlegend": False, "x": [month + "-01" for month in months if status[month] == "Missing"], "y": [0 for month in months if status[month] == "Missing"], "marker": {"color": NEGATIVE, "size": 11, "symbol": "x"}, "hovertemplate": "%{x|%b %Y}<br>Missing source month<extra></extra>"}], coverage_layout,
    )

    return {
        "title": "Isfahan traffic explorer",
        "period": "October 2018 – January 2021",
        "description": "Explore the main changes in traffic, then open a chart when you want a closer look.",
        "mapHref": "../isfahan_intersections_map.html",
        "metrics": [
            {"value": "−%.1f%%" % (abs(headlines["shock_change"]) * 100), "label": "February → March 2020", "detail": "%d matched junctions" % headlines["shock_sites"], "chartId": "changes"},
            {"value": "+%.1f%%" % (headlines["recovery_change"] * 100), "label": "April → May 2020", "detail": "%d matched junctions" % headlines["recovery_sites"], "chartId": "changes"},
            {"value": str(panel_count), "label": "Same junctions through 2019", "detail": "A full year for comparing months", "chartId": "seasonality"},
        ],
        "sections": [
            {"id": "city", "label": "Traffic over time", "title": "How traffic changed", "description": "Start with the monthly picture, then compare the same junctions between months.", "chartIds": ["monthly", "changes"]},
            {"id": "patterns", "label": "Busy days and months", "title": "When traffic is busy", "description": "Look at the weekly rhythm and the months in 2019.", "chartIds": ["weekly", "seasonality"]},
            {"id": "intersections", "label": "At each junction", "title": "A closer look at the junctions", "description": "Compare the fall and recovery, inspect junction 1081, or explore similar weekly patterns.", "chartIds": ["sites", "approaches", "groups"]},
            {"id": "quality", "label": "Data and limits", "title": "What the data can tell us", "description": "Check missing periods and the limits behind each comparison.", "chartIds": ["coverage"]},
        ],
        "charts": charts,
        "notes": [
            {"title": "Missing readings stay missing", "body": "About %.1f%% of checked detector readings are unavailable. Counts are adjusted for missing operational detectors within a recorded interval; missing time periods are not filled." % (summary["quality"]["invalid_channel_rate"] * 100)},
            {"title": "A gap is not a traffic fall", "body": "November 2020 is missing. December 2020 covers only part of the month, so it is excluded from the main monthly comparison."},
            {"title": "Enough coverage comes first", "body": "A junction-month enters the monthly comparisons only after passing both time and detector coverage checks. The developer view shows the configured thresholds."},
            {"title": "Calendar and weather can overlap", "body": "After allowing for month and weekday differences, Nowruz days average %.1f index points lower, official holidays %.1f lower, and wet-weather days %.1f lower. These comparisons do not prove what caused a change." % (abs(insights["nowruz_adjusted_difference_points"]), abs(insights["holiday_adjusted_difference_points"]), abs(insights["wet_weather_adjusted_difference_points"]))},
            {"title": "Entry counts and pattern links", "body": "Detectors count entries near the stop line, not turning paths. Links between similar weekly patterns are not physical road connections."},
            {"title": "Map locations still need checking", "body": "The map uses candidate locations from OpenStreetMap. Municipal GIS should confirm them before planning work."},
        ],
        "sites": [{"id": row["intersection_id"], "name": row["intersection_label"], "change": row.get("shock_change"), "months": row["trusted_months"]} for row in datasets["intersection_summary"]],
        "developer": _developer_data(datasets, validation or {}, settings or {}),
    }


def build_analysis_pages(root: Path = PROJECT_ROOT) -> list[Path]:
    root = Path(root)
    tables = root / "outputs" / "tables"
    read_json = lambda path: json.loads(path.read_text(encoding="utf-8"))
    payload = build_explorer_payload(
        read_json(root / "outputs" / "dashboard" / "artifact.json"),
        read_json(tables / "analysis_summary.json"),
        read_json(tables / "dashboard_insights.json"),
        read_json(tables / "dashboard_validation.json"),
        read_json(root / "config" / "settings.json"),
    )
    template = (Path(__file__).parent / "web" / "traffic_explorer.html").read_text(encoding="utf-8")
    for placeholder in ("__APP_DATA__", "__PLOTLY_JS__"):
        if template.count(placeholder) != 1:
            raise ValueError("HTML template must contain exactly one " + placeholder)
    #embed the data so each page works without a server or network request
    app_data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = template.replace("__PLOTLY_JS__", get_plotlyjs())
    html = html.replace("__APP_DATA__", app_data)
    paths = [
        root / "outputs" / "dashboard" / "dashboard.html",
        root / "outputs" / "dashboard" / "isfahan_intersection_dashboard.html",
        root / "outputs" / "report" / "isfahan_traffic_analysis.html",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    return paths
