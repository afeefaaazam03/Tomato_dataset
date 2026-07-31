from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tomato_multitask.config import load_config
from tomato_multitask.summarize import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tomato ripeness benchmark runs.")
    parser.add_argument("--config", default="configs/tomato_benchmark.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = summarize(config)
    print(f"runs={len(summary)}")


if __name__ == "__main__":
    main()
