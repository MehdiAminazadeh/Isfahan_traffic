from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .paths import PROJECT_ROOT, ensure_project_directories, load_settings


USER_AGENT = "IsfahanTrafficResearch/0.1 (local academic traffic analysis)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PREFIXES = (
    "چهارراه",
    "چهار راه",
    "سه راه",
    "ميدان",
    "میدان",
    "شمال پل",
    "جنوب پل",
    "شمال",
    "جنوب",
)


def normalize_persian(text: str) -> str:
    return (
        str(text)
        .replace("ي", "ی")
        .replace("ك", "ک")
        .replace("ۀ", "ه")
        .replace("ة", "ه")
        .strip()
    )


def split_street_tokens(name: str) -> List[str]:
    normalized = normalize_persian(name)
    pieces = [piece.strip() for piece in re.split(r"\s*[-–—/()]\s*", normalized) if piece.strip()]
    cleaned = []
    for piece in pieces:
        for prefix in PREFIXES:
            if piece.startswith(prefix):
                piece = piece[len(prefix) :].strip()
                break
        if len(piece) >= 3:
            cleaned.append(piece)
    return cleaned


def _inside_bbox(latitude: float, longitude: float, bbox: Sequence[float]) -> bool:
    south, west, north, east = bbox
    return south <= latitude <= north and west <= longitude <= east




def _haversine(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    radius = 6371000.0
    lat1, lon1 = map(math.radians, left)
    lat2, lon2 = map(math.radians, right)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


class CachedNominatim:
    def __init__(self, cache_path: Path, delay_seconds: float, bbox: Sequence[float]):
        self.cache_path = cache_path
        self.delay_seconds = delay_seconds
        self.bbox = bbox
        self.cache: Dict[str, object] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.last_request = 0.0

    def _save(self) -> None:
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )


    def search(self, query: str, limit: int = 10) -> List[Dict[str, object]]:
        key = query + "|" + str(limit)
        if key in self.cache:
            return list(self.cache[key])
        elapsed = time.monotonic() - self.last_request
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        parameters = {
            "q": query,
            "format": "jsonv2",
            "limit": limit,
            "countrycodes": "ir",
            "addressdetails": 1,
            "accept-language": "fa,en",
        }
        request = Request(
            NOMINATIM_URL + "?" + urlencode(parameters),
            headers={"User-Agent": USER_AGENT},
        )
        with urlopen(request, timeout=45) as response:
            results = json.load(response)
        self.last_request = time.monotonic()
        bounded = []
        for result in results:
            latitude = float(result["lat"])
            longitude = float(result["lon"])
            if _inside_bbox(latitude, longitude, self.bbox):
                bounded.append(result)
        self.cache[key] = bounded
        self._save()
        return bounded


def _direct_match(
    geocoder: CachedNominatim, intersection_id: int, name: str, city: str, country: str
) -> Optional[Dict[str, object]]:
    query = "{0}, {1}, {2}".format(name, city, country)
    results = geocoder.search(query, limit=5)
    if not results:
        return None
    result = results[0]
    return {
        "intersection_id": intersection_id,
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
        "geocode_method": "direct_name",
        "geocode_confidence": "medium",
        "geocode_query": query,
        "match_distance_m": 0.0,
        "osm_type": result.get("osm_type", ""),
        "osm_id": result.get("osm_id", ""),
        "display_name": result.get("display_name", ""),
    }


def _street_pair_match(
    geocoder: CachedNominatim,
    intersection_id: int,
    tokens: Sequence[str],
    city: str,
    country: str,
) -> Optional[Dict[str, object]]:
    if len(tokens) < 2:
        return None
    #when the crossing is missing from search results
    #use the closest pair of road features to estimate its location
    best = None
    for left_index in range(min(len(tokens), 3)):
        for right_index in range(left_index + 1, min(len(tokens), 3)):
            left_query = "{0}, {1}, {2}".format(tokens[left_index], city, country)
            right_query = "{0}, {1}, {2}".format(tokens[right_index], city, country)
            left_results = geocoder.search(left_query, limit=15)
            right_results = geocoder.search(right_query, limit=15)
            for left in left_results:
                left_point = (float(left["lat"]), float(left["lon"]))
                for right in right_results:
                    right_point = (float(right["lat"]), float(right["lon"]))
                    distance = _haversine(left_point, right_point)
                    if best is None or distance < best[0]:
                        best = (distance, left_query, right_query, left, right)
    if best is None or best[0] > 1500:
        return None
    
    
    distance, left_query, right_query, left, right = best
    #the midpoint is approximate so keep the separation as a quality measure
    latitude = (float(left["lat"]) + float(right["lat"])) / 2
    longitude = (float(left["lon"]) + float(right["lon"])) / 2
    confidence = "high" if distance <= 150 else "medium" if distance <= 500 else "low"
    return {
        "intersection_id": intersection_id,
        "latitude": latitude,
        "longitude": longitude,
        "geocode_method": "street_pair",
        "geocode_confidence": confidence,
        "geocode_query": left_query + " | " + right_query,
        "match_distance_m": distance,
        "osm_type": "pair",
        "osm_id": "{0}|{1}".format(left.get("osm_id", ""), right.get("osm_id", "")),
        "display_name": "{0} | {1}".format(left.get("display_name", ""), right.get("display_name", "")),
    }


def geocode_intersections(force: bool = False) -> None:
    ensure_project_directories()
    settings = load_settings()
    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    intersections_path = metadata_dir / "intersections.csv"
    output_path = metadata_dir / "intersections_geocoded.csv"
    cache_path = metadata_dir / "geocode_cache.json"
    intersections = pd.read_csv(intersections_path, encoding="utf-8-sig")
    existing = pd.read_csv(output_path, encoding="utf-8-sig") if output_path.exists() else pd.DataFrame()
    existing_by_id = (
        {int(row.intersection_id): row._asdict() for row in existing.itertuples(index=False)}
        if not existing.empty
        else {}
    )
    geocoder = CachedNominatim(
        cache_path,
        float(settings["geocode_delay_seconds"]),
        settings["geocode_bbox"],
    )
    city = "اصفهان"
    country = "ایران"
    rows = []
    named = intersections[intersections["name_fa"].fillna("").ne("")]
    for index, row in enumerate(named.itertuples(index=False), start=1):
        intersection_id = int(row.intersection_id)
        print("[geocode] {0}/{1}: {2}".format(index, len(named), intersection_id), flush=True)
        if not force and intersection_id in existing_by_id:
            rows.append(existing_by_id[intersection_id])
            continue
        name = normalize_persian(row.name_fa)
        match = _direct_match(geocoder, intersection_id, name, city, country)
        if match is None:
            match = _street_pair_match(
                geocoder, intersection_id, split_street_tokens(name), city, country
            )
        if match is None:
            match = {
                "intersection_id": intersection_id,
                "latitude": "",
                "longitude": "",
                "geocode_method": "unmatched",
                "geocode_confidence": "none",
                "geocode_query": name,
                "match_distance_m": "",
                "osm_type": "",
                "osm_id": "",
                "display_name": "",
            }
        rows.append(match)
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    geocoded = pd.DataFrame(rows)
    merged = intersections.merge(geocoded, how="left", on="intersection_id")
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    located = merged["latitude"].notna() & merged["latitude"].astype(str).ne("")
    print("[geocode] located {0}/{1} named intersections".format(int(located.sum()), len(named)), flush=True)
