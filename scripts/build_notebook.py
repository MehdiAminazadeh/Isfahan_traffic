from __future__ import annotations

import json
import asyncio
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    #the default Windows loop can hang or fail in some Anaconda builds
    #use the selector loop for notebook execution
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "notebook"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary = json.loads(
    (PROJECT_ROOT / "outputs" / "tables" / "analysis_summary.json").read_text(encoding="utf-8")
)

decline = summary["time_series"]["largest_consecutive_month_decline"]
increase = summary["time_series"]["largest_consecutive_month_increase"]
site = summary["intersection_1081"]
graph = summary["functional_graph"]
clustering = summary["dtw_clustering"]

notebook = nbf.v4.new_notebook()
notebook["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}

notebook["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Isfahan intersection traffic: graph mining and time-series analysis\n\n"
        "## tl;dr\n\n"
        "- The raw archive contains **{records:,} intersection records** across **{intersections} intersections** from October 2018 to January 2021.\n"
        "- Among the same {decline_n} trusted intersections in February and March 2020, median volume fell **{decline_pct:.1%}**; among {increase_n} paired intersections, April to May rebounded **{increase_pct:.1%}**. These are descriptive associations, not causal estimates.\n"
        "- At intersection **1081**, north and south approaches carry **{corridor_share:.1%}** of measured entries. It ranks **{volume_rank}/{ranked}** by volume but **{graph_rank}/{ranked}** by functional-similarity centrality.\n"
        "- The weekly-profile graph has **{communities} communities** and remains connected; DTW favors **{clusters} daily-shape clusters** (silhouette {silhouette:.2f}).\n"
        "- November 2020 is absent and December 2020 is partial, so neither is used as a complete month in headline comparisons."
        .format(
            records=summary["data_window"]["raw_intersection_records"],
            intersections=summary["data_window"]["intersections_with_raw_data"],
            decline_n=decline["paired_intersections"],
            decline_pct=decline["median_change"],
            increase_n=increase["paired_intersections"],
            increase_pct=increase["median_change"],
            corridor_share=site["north_south_corridor_share"],
            volume_rank=site["volume_rank"],
            graph_rank=site["functional_similarity_centrality_rank"],
            ranked=site["ranked_intersections"],
            communities=graph["communities"],
            clusters=clustering["selected_clusters"],
            silhouette=clustering["silhouette"],
        )
    ),
    nbf.v4.new_markdown_cell(
        "## Context & Methods\n\n"
        "The numbered SCATS boxes are lane-detector channels on approaches to the stop line. The analysis therefore calls them **approach entries** rather than exits. `0` remains a valid count; `2046`, `2047`, and `NA` are unavailable readings. Coverage normalization is applied only when at least half of the operational detector set is present.\n\n"
        "### Key Assumptions\n\n"
        "- Timestamps are interpreted in Isfahan local civil time.\n"
        "- Traffic volume is the sum across operational channels; normalization scales only for unavailable channels, not for missing time intervals.\n"
        "- Citywide level charts use trusted intersection-months; month-to-month changes use paired intersections.\n"
        "- The functional graph encodes similarity of 168-bin weekly profiles. Its edges do not imply a road connection or vehicle transfer.\n"
        "- 2019 is the only complete pre-pandemic year and supplies the common-panel month-of-year comparison."
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display\n\n"
        "PROJECT_ROOT = Path.cwd()\n"
        "TABLES = PROJECT_ROOT / 'outputs' / 'tables'\n"
        "FIGURES = PROJECT_ROOT / 'outputs' / 'figures'\n"
        "summary = json.loads((TABLES / 'analysis_summary.json').read_text(encoding='utf-8'))\n"
        "summary['data_window']"
    ),
    nbf.v4.new_markdown_cell("## Data\n\nThe compact source-month audit makes missing and partial coverage explicit."),
    nbf.v4.new_code_cell(
        "source_quality = pd.read_csv(TABLES / 'source_month_quality.csv')\n"
        "source_quality[['source_month', 'source_temporal_coverage', 'intersection_records', 'records_with_normalized_volume']].tail(8)"
    ),
    nbf.v4.new_markdown_cell(
        "## Results\n\n### The strongest network-wide discontinuity is March 2020\n\n"
        "The level plot uses the median intersection and an interquartile band. The paired-intersection result is the safer percentage comparison because detector availability changes over time."
    ),
    nbf.v4.new_code_cell("display(Image(filename=str(FIGURES / 'city_monthly_volume.png'), width=980))"),
    nbf.v4.new_code_cell(
        "paired = pd.read_csv(TABLES / 'paired_month_changes.csv')\n"
        "paired.sort_values('median_change').head(6)"
    ),
    nbf.v4.new_markdown_cell(
        "### Seasonal variation is modest in the 2019 common panel\n\n"
        "The peak-to-trough spread is much smaller than the March 2020 discontinuity, indicating that ordinary Gregorian month seasonality is a secondary driver in this archive."
    ),
    nbf.v4.new_code_cell("display(Image(filename=str(FIGURES / 'month_of_year_pattern.png'), width=920))"),
    nbf.v4.new_markdown_cell(
        "### 1081 is a north–south-oriented site with a pronounced Friday pattern\n\n"
        "The approach split is stable outside the early-2020 discontinuity. The weekday profile shows Friday materially below most other days during daytime hours."
    ),
    nbf.v4.new_code_cell(
        "display(Image(filename=str(FIGURES / 'intersection_1081_approaches.png'), width=980))\n"
        "display(Image(filename=str(FIGURES / 'intersection_1081_hourly_weekday.png'), width=920))"
    ),
    nbf.v4.new_markdown_cell(
        "### Functional communities reveal recurring operating shapes\n\n"
        "Nodes with many strong similarity edges are representative of common network behavior, not necessarily the busiest or geographically central intersections."
    ),
    nbf.v4.new_code_cell("display(Image(filename=str(FIGURES / 'functional_similarity_graph.png'), width=900))"),
    nbf.v4.new_code_cell(
        "nodes = pd.read_csv(TABLES / 'functional_graph_nodes.csv', encoding='utf-8-sig')\n"
        "nodes.sort_values('weighted_degree', ascending=False)[['intersection_id', 'community', 'dtw_cluster', 'weighted_degree', 'mean_volume']].head(12)"
    ),
    nbf.v4.new_markdown_cell(
        "## Limitations, uncertainty, and robustness checks\n\n"
        "- The archive contains a high unavailable-reading rate, and the trusted panel shrinks in late 2019. Headline changes therefore use within-intersection pairs.\n"
        "- November 2020 is missing; December 2020 covers only about one fifth of the month.\n"
        "- Geocoded coordinates are review candidates from OpenStreetMap/Nominatim, not municipal GIS truth.\n"
        "- The two-cluster DTW solution isolates a small outlier group; community results are more granular and should be used for operational segmentation.\n"
        "- Entry/exit conservation cannot be tested without a verified downstream intersection/link mapping."
    ),
    nbf.v4.new_markdown_cell(
        "## Takeaways\n\n"
        "1. Use the paired March decline and May rebound as the primary time-series event, with no causal label unless external policy/incident data are added.\n"
        "2. Treat 1081 as a useful representative monitoring site: it is low-volume but highly connected in behavior space, with a stable north–south orientation.\n"
        "3. Operationalize the four functional communities for peer baselines and anomaly detection; do not compare every intersection against one citywide average.\n"
        "4. Verify coordinates and road adjacency in QGIS/municipal GIS before building a physical directed road graph.\n"
        "5. Add holiday, Nowruz, Ramadan, weather, and traffic-control-event data before interpreting seasonal or causal mechanisms."
    ),
]

notebook_path = OUTPUT_DIR / "isfahan_intersection_analysis.ipynb"
nbf.write(notebook, notebook_path)
client = NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}})
client.execute()
nbf.write(notebook, notebook_path)
print("[notebook] executed and wrote {0}".format(notebook_path))
