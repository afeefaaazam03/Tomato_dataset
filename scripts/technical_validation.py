from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from tomato_multitask.config import load_config, output_dirs
from tomato_multitask.validation import run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Scientific-Data-style technical validation stats.")
    parser.add_argument("--config", default="configs/tomato_benchmark.yaml")
    parser.add_argument("--out", default="outputs/technical_validation.json")
    args = parser.parse_args()

    config = load_config(args.config)
    dirs = output_dirs(config)
    image_root = Path(config["data_root"])
    image_manifest = pd.read_csv(dirs["annotations"] / "image_manifest.csv")
    classes = list(config["classes"])

    result = run_validation(image_root, image_manifest, classes)
    quality_df = result.pop("quality_df")
    quality_df.to_csv(Path(args.out).with_name("image_quality_metrics.csv"), index=False)

    # trim near_duplicate_pairs for readability if huge
    result["near_duplicate_pair_count"] = len(result["near_duplicate_pairs"])
    result["near_duplicate_pairs_sample"] = result["near_duplicate_pairs"][:20]
    del result["near_duplicate_pairs"]
    result["exact_duplicate_group_count"] = len(result["exact_duplicate_groups"])
    result["exact_duplicate_groups_sample"] = dict(list(result["exact_duplicate_groups"].items())[:10])
    del result["exact_duplicate_groups"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_name('image_quality_metrics.csv')}")


if __name__ == "__main__":
    main()
