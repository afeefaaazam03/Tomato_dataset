from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path.resolve())
    return config


def output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config.get("output_root", "outputs/tomato_benchmark"))
    dirs = {
        "root": root,
        "audit": root / "00_data_audit",
        "annotations": root / "01_annotations",
        "splits": root / "02_splits",
        "detection": root / "03_detection",
        "segmentation": root / "04_segmentation",
        "counting": root / "05_counting",
        "classification": root / "06_classification",
        "analysis": root / "07_cross_task_analysis",
        "results": root / "results",
        "tables": root / "results" / "tables",
        "figures": root / "results" / "figures",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def class_slug(name: str) -> str:
    """Filesystem/identifier-safe version of a display class name, e.g. 'Light Red' -> 'light_red'."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")
