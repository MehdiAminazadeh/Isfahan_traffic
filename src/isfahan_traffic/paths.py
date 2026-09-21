from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_settings() -> Dict[str, Any]:
    settings_path = PROJECT_ROOT / "config" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    source_root = Path(settings["source_root"])
    if not source_root.is_absolute():
        source_root = (PROJECT_ROOT / source_root).resolve()
    settings["source_root"] = source_root
    settings["project_root"] = PROJECT_ROOT
    return settings


def ensure_project_directories() -> None:
    for relative in (
        "data/metadata",
        "data/processed/15min",
        "data/external/raw",
        "data/external/processed",
        "data/external/manual",
        "outputs/tables",
        "outputs/figures",
        "outputs/notebook",
        "outputs/report",
        "outputs/dashboard",
    ):
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)
