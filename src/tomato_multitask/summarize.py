from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import output_dirs
from .utils import json_load


PRIMARY_METRICS = {
    "classification": "macro_f1",
    "counting": "mae",
    "segmentation": "miou_foreground",
    "detection": "map50",
}


LOWER_IS_BETTER = {"mae", "rmse", "mape", "loss"}


def _run_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    dirs = output_dirs(config)
    rows: list[dict[str, Any]] = []
    for task in ["classification", "segmentation", "counting", "detection"]:
        run_root = dirs[task] / "runs"
        if not run_root.exists():
            continue
        for metrics_path in sorted(run_root.glob("*/test_metrics.json")):
            run_dir = metrics_path.parent
            metadata_path = run_dir / "metadata.json"
            metrics = json_load(metrics_path)
            metadata = json_load(metadata_path) if metadata_path.exists() else {}
            rows.append({"task": task, "run_dir": str(run_dir.resolve()), **metadata, **metrics})
    return rows


def _markdown_table(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        path.write_text("No runs found.\n", encoding="utf-8")
    else:
        path.write_text(frame.to_markdown(index=False), encoding="utf-8")


def _save_bar(frame: pd.DataFrame, metric: str, path: Path, lower_is_better: bool) -> None:
    if frame.empty or metric not in frame.columns:
        return
    plot_df = frame.copy().sort_values(metric, ascending=not lower_is_better)
    labels = plot_df.get("display_name", plot_df["method"]).fillna(plot_df["method"]).astype(str)
    plt.figure(figsize=(10, max(5, len(plot_df) * 0.38)))
    plt.barh(labels, plot_df[metric], color="#4c78a8")
    plt.title(f"{metric} by method")
    plt.xlabel(metric)
    if lower_is_better:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def summarize(config: dict[str, Any]) -> pd.DataFrame:
    dirs = output_dirs(config)
    tables = dirs["tables"]
    figures = dirs["figures"]
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    rows = _run_rows(config)
    summary = pd.DataFrame(rows)
    summary.to_csv(tables / "all_task_runs.csv", index=False)
    _markdown_table(summary, tables / "all_task_runs.md")

    for task, metric in PRIMARY_METRICS.items():
        frame = summary[summary["task"] == task].copy() if not summary.empty else pd.DataFrame()
        if frame.empty:
            continue
        if metric in frame.columns:
            frame = frame.sort_values(metric, ascending=metric in LOWER_IS_BETTER)
        frame.to_csv(tables / f"{task}_summary.csv", index=False)
        _markdown_table(frame, tables / f"{task}_summary.md")
        _save_bar(frame, metric, figures / f"{task}_{metric}_bar.png", lower_is_better=metric in LOWER_IS_BETTER)

    return summary
