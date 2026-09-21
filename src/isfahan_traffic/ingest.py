from __future__ import annotations

import calendar
import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from .parser import discover_raw_month_files, iter_scats_records, source_month
from .paths import PROJECT_ROOT, ensure_project_directories, load_settings


APPROACHES = ("south", "west", "north", "east")


def load_channel_map() -> Dict[int, List[Tuple[int, str]]]:
    path = PROJECT_ROOT / "data" / "metadata" / "detector_channels.csv"
    if not path.exists():
        raise FileNotFoundError("Run the metadata stage before ingest: {0}".format(path))
    detectors = pd.read_csv(path)
    detectors = detectors[detectors["operational"].astype(str).str.lower().isin(["true", "1"])]
    mapping: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for row in detectors.itertuples(index=False):
        mapping[int(row.intersection_id)].append((int(row.channel_id), str(row.approach)))
    return {key: sorted(value) for key, value in mapping.items()}


def _approach_metrics(
    channels: Mapping[int, object], selected: Sequence[Tuple[int, str]], minimum_coverage: float
) -> Dict[str, object]:
    output: Dict[str, object] = {}
    grouped: Dict[str, List[int]] = defaultdict(list)
    for channel_id, approach in selected:
        if approach in APPROACHES:
            grouped[approach].append(channel_id)
    for approach in APPROACHES:
        configured = grouped.get(approach, [])
        values = [channels.get(channel_id) for channel_id in configured]
        valid_values = [int(value) for value in values if value is not None]
        coverage = len(valid_values) / len(configured) if configured else None
        observed = sum(valid_values) if valid_values else None
        normalized = (
            observed / coverage
            if observed is not None and coverage is not None and coverage >= minimum_coverage
            else None
        )
        output[approach + "_volume"] = normalized
        output[approach + "_coverage"] = coverage
    return output


def ingest_file(
    path: Path,
    output_path: Path,
    channel_map: Mapping[int, Sequence[Tuple[int, str]]],
    minimum_coverage: float,
) -> Dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    fields = [
        "timestamp",
        "intersection_id",
        "volume_observed",
        "volume_normalized",
        "detectors_expected",
        "detectors_valid",
        "detector_coverage",
        "south_volume",
        "south_coverage",
        "west_volume",
        "west_coverage",
        "north_volume",
        "north_coverage",
        "east_volume",
        "east_coverage",
    ]
    record_count = 0
    normalized_count = 0
    intersection_ids = set()
    timestamps = set()
    first_seen = None
    last_seen = None

    with gzip.open(temporary_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in iter_scats_records(path):
            record_count += 1
            intersection_ids.add(record.intersection_id)
            timestamps.add(record.timestamp)
            first_seen = record.timestamp if first_seen is None else min(first_seen, record.timestamp)
            last_seen = record.timestamp if last_seen is None else max(last_seen, record.timestamp)

            selected = list(channel_map.get(record.intersection_id, []))
            if not selected:
                selected = [(channel_id, "unknown") for channel_id in sorted(record.channels)]
            values = [record.channels.get(channel_id) for channel_id, _ in selected]
            valid_values = [int(value) for value in values if value is not None]
            expected = len(selected)
            valid = len(valid_values)
            coverage = valid / expected if expected else 0.0
            observed = sum(valid_values) if valid_values else None
            #this adjustment covers missing detectors within a time interval
            #missing intervals need a separate coverage measure
            normalized = (
                observed / coverage
                if observed is not None and coverage >= minimum_coverage
                else None
            )
            if normalized is not None:
                normalized_count += 1

            row = {
                "timestamp": record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "intersection_id": record.intersection_id,
                "volume_observed": observed,
                "volume_normalized": round(normalized, 6) if normalized is not None else None,
                "detectors_expected": expected,
                "detectors_valid": valid,
                "detector_coverage": round(coverage, 6),
            }
            row.update(_approach_metrics(record.channels, selected, minimum_coverage))
            writer.writerow(row)

    temporary_path.replace(output_path)
    if first_seen is None:
        expected_timestamps = 0
    else:
        #a full day has 96 quarter hour intervals
        #these exports do not need a daylight saving adjustment
        expected_timestamps = calendar.monthrange(first_seen.year, first_seen.month)[1] * 96
    return {
        "source_file": path.name,
        "source_month": source_month(path),
        "first_timestamp": first_seen.isoformat() if first_seen else "",
        "last_timestamp": last_seen.isoformat() if last_seen else "",
        "timestamps_observed": len(timestamps),
        "timestamps_expected_full_month": expected_timestamps,
        "source_temporal_coverage": len(timestamps) / expected_timestamps if expected_timestamps else None,
        "intersection_records": record_count,
        "records_with_normalized_volume": normalized_count,
        "intersections": len(intersection_ids),
        "output_file": output_path.name,
    }


def ingest_all(force: bool = False) -> None:
    ensure_project_directories()
    settings = load_settings()
    raw_files = discover_raw_month_files(settings["source_root"])
    channel_map = load_channel_map()
    output_dir = PROJECT_ROOT / "data" / "processed" / "15min"
    summaries = []
    for index, path in enumerate(raw_files, start=1):
        month = source_month(path)
        output_path = output_dir / (month + ".csv.gz")
        print("[ingest] {0}/{1}: {2}".format(index, len(raw_files), path.name), flush=True)
        if output_path.exists() and not force:
            print("[ingest] keeping existing partition {0}".format(output_path.name), flush=True)
            continue
        summaries.append(
            ingest_file(
                path,
                output_path,
                channel_map,
                float(settings["minimum_detector_coverage"]),
            )
        )
    summary_path = PROJECT_ROOT / "outputs" / "tables" / "source_month_quality.csv"
    if summaries:
        pd.DataFrame(summaries).sort_values("source_month").to_csv(summary_path, index=False)
    print("[ingest] partitions available in {0}".format(output_dir), flush=True)
