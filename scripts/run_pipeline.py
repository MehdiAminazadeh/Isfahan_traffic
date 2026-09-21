from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isfahan_traffic.analysis import analyze_all
from isfahan_traffic.context import build_external_context
from isfahan_traffic.geocode import geocode_intersections
from isfahan_traffic.ingest import ingest_all
from isfahan_traffic.metadata import build_metadata
from isfahan_traffic.visuals import build_visuals


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Isfahan SCATS analysis pipeline")
    parser.add_argument(
        "stage",
        choices=["metadata", "geocode", "ingest", "context", "analyze", "visualize", "all"],
        help="Pipeline stage to execute",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild cached geocodes or processed partitions")
    arguments = parser.parse_args()

    #separate checks let all stages run in dependency order
    if arguments.stage in ("metadata", "all"):
        build_metadata()
    if arguments.stage in ("geocode", "all"):
        geocode_intersections(force=arguments.force)
    if arguments.stage in ("ingest", "all"):
        ingest_all(force=arguments.force)
    if arguments.stage in ("context", "all"):
        build_external_context(force=arguments.force)
    if arguments.stage in ("analyze", "all"):
        analyze_all()
    if arguments.stage in ("visualize", "all"):
        build_visuals()


if __name__ == "__main__":
    main()
