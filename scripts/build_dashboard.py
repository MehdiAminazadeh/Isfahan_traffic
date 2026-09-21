from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isfahan_traffic.dashboard import build_dashboard_artifact
from isfahan_traffic.report_ui import build_analysis_pages
from isfahan_traffic.web_navigation import add_map_navigation


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Isfahan traffic explorer")
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help="Refresh the analysis tables without rebuilding HTML",
    )
    parser.add_argument(
        "--html-only", action="store_true", help="Build HTML from the saved analysis tables"
    )
    arguments = parser.parse_args()
    if arguments.artifact_only and arguments.html_only:
        parser.error("choose either artifact-only or html-only")
    #build the tables first unless only the interface needs to change
    if not arguments.html_only:
        build_dashboard_artifact()
    if arguments.artifact_only:
        return
    for output in build_analysis_pages():
        print("Saved " + str(output))
    add_map_navigation(PROJECT_ROOT / "outputs/isfahan_intersections_map.html")


if __name__ == "__main__":
    main()
