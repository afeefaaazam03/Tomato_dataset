# Tomato Ripeness Benchmark

A multi-task benchmark (classification, segmentation, counting, detection) for grading tomato
ripeness, structured after the [Blueberry-Ripeness-Dataset](https://github.com/iyyakuttiiyappan/Blueberry-Ripeness-Dataset)
project. **Classification is the primary, fully-configured task** -- it grades a cropped tomato
into one of six ripeness stages.

## Ripeness Classes

Per the Silal "Quality Standard Guide For Tomato":

| # | Class       | Definition                |
|---|-------------|----------------------------|
| 1 | Green       | 100% green                 |
| 2 | Breakers    | less than 10% colour       |
| 3 | Turning     | 10-40% colour               |
| 4 | Pink        | 40-80% colour                |
| 5 | Light Red   | 80-90% red                  |
| 6 | Red         | more than 90% red           |

These names match the `categories` in `Json_Output/*.json` and the `Mask/<name>/` folders in the
raw data release exactly -- do not rename them without updating `configs/tomato_benchmark.yaml`.

## Dataset

- **535 RGB images**, 7680x4320, greenhouse/field tomato clusters, annotated with per-instance
  COCO-style polygons (`Json_Output/Set1_535.json`) and per-class binary silhouette masks
  (`Mask/Green/`, `Mask/Breakers/`, `Mask/Turning/`, `Mask/Pink/`, `Mask/Light Red/`, `Mask/Red/`,
  `Mask/Overall/`).
- Raw layout expected at `data/raw/Set1_535/` (see [RUNNING_THE_CODE.md](RUNNING_THE_CODE.md)).
- The data-prep pipeline extracts one crop per tomato instance (mask connected component, padded
  bounding box) and labels it with its ripeness class -- this crop-level dataset is what the
  classifier trains on, mirroring the blueberry benchmark's `CropClassificationDataset`.

## Repository Layout

```text
configs/              Benchmark configuration (classes, split, hyperparameters, model zoo)
docs/                 Preprocessing pipeline and data-access notes
metadata_examples/    Small example CSV/JSON schema samples
scripts/              CLI entry points (prepare data, run a task, summarize results)
src/tomato_multitask/ Python package: annotations, datasets, models, engine, metrics, plots
data/raw/              Where you place the raw RGB images + Mask/ + Json_Output/ (not committed)
outputs/               Generated crops, splits, checkpoints, metrics, plots (not committed)
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows; use `source .venv/bin/activate` on Linux/macOS
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# 1. Build crops, splits, semantic masks, and the dataset audit report
python scripts/prepare_annotations.py --config configs/tomato_benchmark.yaml

# 2. Train the recommended classifier (ConvNeXtV2-Tiny, class-balanced)
python scripts/run_task.py --config configs/tomato_benchmark.yaml \
  --task classification --method convnextv2_tiny_balanced

# 3. Aggregate metrics/plots across every run you've done
python scripts/summarize.py --config configs/tomato_benchmark.yaml
```

See [RUNNING_THE_CODE.md](RUNNING_THE_CODE.md) for full instructions, including how to run the
segmentation/counting/detection tasks and how to interpret the outputs.

## Why classification is crop-based, not whole-image

A single greenhouse photo usually contains several tomatoes at different ripeness stages at once,
so "one label per photo" doesn't make sense. Instead (matching the blueberry benchmark's design),
the pipeline crops out each individual tomato instance -- using its mask connected component, with
padding -- and classifies that crop. `outputs/.../06_classification/crops/<class>/*.jpg` holds the
extracted crops after step 1 above.

## Model & Training

- Backbone: any `timm` model swappable via config; default/recommended is **ConvNeXtV2-Tiny**
  (`convnextv2_tiny.fcmae_ft_in22k_in1k`), a modern ImageNet-22k-pretrained CNN. ResNet-50,
  EfficientNetV2/B4, MobileNetV3, Swin-T, DeiT3, MaxViT, EVA02, and a frozen DINOv2 linear probe
  are also configured -- see `configs/tomato_benchmark.yaml`.
- AdamW + cosine LR schedule, mixed precision, early stopping on validation macro-F1, best
  checkpoint saved to `best.pt`.
- Class imbalance: `WeightedRandomSampler` for training batches + inverse-frequency class weights
  in the loss (the `_balanced` method variants), matching the blueberry repo's
  `resnet50_balanced` approach.
- Augmentation: horizontal flip, small random rotation, brightness/contrast/saturation jitter,
  and a deliberately small hue jitter -- see "Tomato-Specific Preprocessing Notes" below.

## Evaluation Outputs (per run, under `outputs/.../06_classification/runs/<run>/`)

- `test_metrics.json` -- accuracy, macro-F1, weighted-F1, balanced accuracy
- `classification_report_test.csv` -- per-class precision/recall/F1
- `confusion_matrix_test.{csv,png}`
- `sample_predictions.png` -- grid of correct + misclassified test crops with true/predicted labels
- `training_curves.png`, `history.csv` -- loss/metric per epoch

## Tomato-Specific Preprocessing Notes

- **Color space**: tomato ripening is a hue rotation across most of the visible spectrum
  (green -> red), unlike blueberry where the signal sits mostly in lightness/saturation on a
  narrower blue-purple range. Hue jitter is therefore applied much more cautiously than
  brightness/contrast/saturation jitter (`hue_jitter: 0.02` vs `jitter_strength: 0.15` in
  `configs/tomato_benchmark.yaml`) so augmentation doesn't accidentally mimic a different class.
- **Stem/calyx**: the green calyx at the top of a tomato is visually similar to the `Green` class
  and can bias color-based cues. `annotation.stem_trim_top_ratio` (default `0.0`, disabled) can
  crop away a top band of each instance box; only enable it after visually checking real crops,
  since it can cut into small or rotated fruit. See `docs/Preprocessing_Pipeline.md`.
- **Shape/orientation**: unlike blueberries (small, roughly uniform spheres), tomato instances
  vary a lot in apparent size and orientation depending on camera distance and cluster pose, so
  crops use padded, per-instance bounding boxes rather than a fixed aspect ratio, and rotation
  augmentation is used to build orientation invariance.

## Requirements

See `requirements.txt`. PyTorch + a CUDA build matching your GPU is recommended for real training
runs; CPU works for smoke-testing with `--limit`.
