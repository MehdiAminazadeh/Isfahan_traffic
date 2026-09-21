from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


TIMESTAMP_RE = re.compile(
    r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+"
    r"(\d{2})\s+([A-Za-z]+)\s+(\d{4})\s+(\d{2}:\d{2})$"
)
INTERSECTION_RE = re.compile(r"^Int\s+(\d+)\b")
CHANNEL_RE = re.compile(r"(\d+)\s*=\s*(NA|\d+)", re.IGNORECASE)
#these values mark missing detector data in SCATS
INVALID_SENTINELS = {2046, 2047}



@dataclass(frozen=True)
class ScatsRecord:
    timestamp: datetime
    intersection_id: int
    channels: Dict[int, Optional[int]]


def parse_timestamp(line: str) -> Optional[datetime]:
    if not TIMESTAMP_RE.match(line):
        return None
    return datetime.strptime(line, "%A %d %B %Y %H:%M")


def parse_intersection_block(
    timestamp: Optional[datetime], block: Optional[str]
) -> Optional[ScatsRecord]:
    if timestamp is None or not block:
        return None
    match = INTERSECTION_RE.match(block)
    if not match:
        return None
    channels: Dict[int, Optional[int]] = {}
    for channel_text, value_text in CHANNEL_RE.findall(block):
        channel_id = int(channel_text)
        if value_text.upper() == "NA":
            channels[channel_id] = None
        else:
            value = int(value_text)
            channels[channel_id] = None if value in INVALID_SENTINELS else value
    if not channels:
        return None
    return ScatsRecord(timestamp, int(match.group(1)), channels)


def iter_scats_records(path: Path) -> Iterator[ScatsRecord]:

    current_timestamp: Optional[datetime] = None
    current_block: Optional[str] = None

    #a record can span several lines
    #keep collecting it until the next timestamp or intersection header
    def flush() -> Optional[ScatsRecord]:
        return parse_intersection_block(current_timestamp, current_block)

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            parsed_time = parse_timestamp(line)
            if parsed_time is not None:
                record = flush()
                if record is not None:
                    yield record
                current_block = None
                current_timestamp = parsed_time
                continue
            
            if INTERSECTION_RE.match(line):
                record = flush()
                if record is not None:
                    yield record
                current_block = line
                continue

            if not line or line.lower().startswith("end of file"):
                continue
            if current_block is not None:
                current_block += " " + line

    record = flush()
    if record is not None:
        yield record


def first_timestamp(path: Path) -> datetime:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        
        
        for raw_line in handle:
            timestamp = parse_timestamp(raw_line.strip())
            if timestamp is not None:
                return timestamp
            
    raise ValueError("No SCATS timestamp found in {0}".format(path))


def discover_raw_month_files(source_root: Path) -> List[Path]:
    raw_root = source_root / "scats-97-99"
    candidates = [
        path
        for path in raw_root.rglob("*.txt")
        if path.name.lower() != "output.txt"
        and re.search(r"20(?:18|19|20|21)", path.name)
    ]
    #sort by the first timestamp because filenames use different date formats
    dated: List[Tuple[datetime, Path]] = []
    for path in candidates:
        dated.append((first_timestamp(path), path))
    return [path for _, path in sorted(dated, key=lambda item: item[0])]


def source_month(path: Path) -> str:
    return first_timestamp(path).strftime("%Y-%m")


