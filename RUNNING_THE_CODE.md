# Running The Code

These instructions assume you are working from the root of this repository on Windows
(PowerShell); Linux/macOS commands are noted where they differ.

## 1. Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build that matches your GPU/CPU environment if `requirements.txt`'s default
doesn't pick the right CUDA version for you (see https://pytorch.org/get-started/locally/).

## 2. Lay Out The Raw Data

The pipeline expects everything merged under a single folder:

```text
data/raw/Set1_535/
  20260416_123109_000009.jpg   <- RGB photos, directly in this folder (matches Json "file_name")
  20260416_123109_000019.jpg
  ...
  Json_Output/
    Set1_535.json               <- COCO-style: categories, images, per-instance polygon annotations
  Mask/
    Green/                      <- binary silhouette masks, one per class
    Breakers/
    Turning/
    Pink/
    Light Red/
    Red/
    Overall/                    <- combined "any tomato" mask, used for audit + component counts
```

This is exactly what you get by extracting the two source archives (mask/JSON export +
RGB image export) into the same destination folder -- they share the same `Set1_535/` root and
merge automatically. If your raw data lives somewhere else, either symlink/copy it to
`data/raw/Set1_535/`, or edit `data_root` (and `paths.image_dir` / `paths.mask_dir` /
`paths.json_dir` if the internal layout differs) in `configs/tomato_benchmark.yaml`.

## 3. Prepare Annotations (crops, splits, semantic masks, audit report)

```bash
python scripts/prepare_annotations.py --config configs/tomato_benchmark.yaml
```

This is idempotent -- it skips work if `outputs/.../01_annotations/*.csv` already exist. Pass
`--rebuild` to force a full re-run (e.g. after changing `crop_padding`, `mask_threshold`, or the
class list). Outputs land under `outputs/tomato_benchmark_535/`:

```text
00_data_audit/           dataset_audit.md, overlay contact sheets, audit figures
01_annotations/          image_manifest.csv, instances.csv, classification_crops.csv, semantic_masks/
02_splits/                per-split CSVs (images_train.csv, crops_val.csv, ...)
06_classification/crops/  one folder per class, one JPEG per tomato instance
03_detection/{coco,yolo}/ per-split COCO JSON + YOLO-format image/label export
```

Read `00_data_audit/dataset_audit.md` first -- it cross-checks the COCO annotation counts against
the mask connected-component counts per image, which catches mask/image misalignment or
orientation problems before you spend time training.

## 4. Train The Classifier

```bash
python scripts/run_task.py --config configs/tomato_benchmark.yaml \
  --task classification --method convnextv2_tiny_balanced
```

Useful flags: `--epochs`, `--batch-size`, `--device cuda|cpu`, `--seed`, and `--limit N` (caps
each split to N examples -- use this first to smoke-test the whole pipeline in under a minute
before committing to a full run). Available `--method` values are the keys under
`tasks.classification` in `configs/tomato_benchmark.yaml` (e.g. `resnet50`, `efficientnetv2_s`,
`swin_tiny`, `dinov2_vitb14_linear`, ...).

Each run writes to `outputs/tomato_benchmark_535/06_classification/runs/<timestamp>_<method>_seed<seed>/`:
`best.pt` (checkpoint), `history.csv` + `training_curves.png`, `test_metrics.json`,
`classification_report_test.csv`, `confusion_matrix_test.{csv,png}`, `sample_predictions.png`.

## 5. Train The Other Tasks (optional)

```bash
# Per-pixel ripeness segmentation
python scripts/run_task.py --config configs/tomato_benchmark.yaml \
  --task segmentation --method fpn_convnextv2_tiny

# Tomato-count regression per image
python scripts/run_task.py --config configs/tomato_benchmark.yaml \
  --task counting --method count_convnextv2_tiny

# Per-instance ripeness detection (bounding boxes)
python scripts/run_task.py --config configs/tomato_benchmark.yaml \
  --task detection --method fasterrcnn_resnet50_fpn_v2
```

## 6. Summarize Results

```bash
python scripts/summarize.py --config configs/tomato_benchmark.yaml
```

Writes leaderboard tables/figures per task to `outputs/tomato_benchmark_535/results/`.

## Smoke-Testing Without Waiting For A Full Run

```bash
python scripts/run_task.py --config configs/tomato_benchmark.yaml \
  --task classification --method resnet50 --epochs 1 --limit 12
```
