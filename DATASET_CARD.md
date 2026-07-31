# Dataset Card

## Dataset Name

Set1_535: Tomato Ripeness Dataset (working name)

## Summary

535 RGB images (7680x4320) of tomato clusters, annotated with per-instance polygon segmentation
(COCO-style JSON) and per-class binary silhouette masks, covering six ripeness stages defined by
the Silal "Quality Standard Guide For Tomato".

## Classes

| # | Class     | Definition          |
|---|-----------|----------------------|
| 1 | Green     | 100% green            |
| 2 | Breakers  | less than 10% colour  |
| 3 | Turning   | 10-40% colour          |
| 4 | Pink      | 40-80% colour          |
| 5 | Light Red | 80-90% red             |
| 6 | Red       | more than 90% red      |

## Primary Data Products (as delivered)

- RGB JPEG images (7680x4320).
- COCO-style instance annotations (`Json_Output/*.json`): per-tomato polygon segmentation +
  ripeness category.
- Per-class binary PNG/JPEG silhouette masks (`Mask/<class>/`) plus a combined `Mask/Overall/`.

## Derived Data Products (built by `scripts/prepare_annotations.py`)

- Per-instance crops labeled by ripeness class (`06_classification/crops/<class>/*.jpg`) --
  used to train the classifier.
- Semantic segmentation masks (label IDs 1-6, 0=background) built from the per-class binary masks.
- Per-image instance counts (from the COCO annotations, cross-checked against mask connected
  components).
- A deterministic 70/15/15 train/val/test split (stratified by dominant class and count).
- Per-split COCO and YOLO-format detection exports.

## Intended Uses

- Ripeness-stage classification of individual tomatoes (primary use case for this repo).
- Ripeness-stage semantic segmentation.
- Tomato counting per image.
- Ripeness-stage detection/localization.

## Limitations

- Image count (535) is modest for training large transformer backbones from scratch; the
  configured methods all use ImageNet-pretrained backbones for this reason.
- The RGB source images are high resolution (7680x4320); the classification pipeline works on
  small per-instance crops, but segmentation/detection at full resolution is memory-intensive --
  see `task_defaults.segmentation.image_size` / `task_defaults.detection.image_size` in
  `configs/tomato_benchmark.yaml` for the working resolution actually used.
- Capture conditions (greenhouse/field lighting, camera angle) are not documented per-image; if
  you have that metadata, consider stratifying splits or reporting metrics by capture batch.
