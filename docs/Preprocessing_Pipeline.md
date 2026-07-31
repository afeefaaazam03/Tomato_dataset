# Preprocessing Pipeline

Implemented in `src/tomato_multitask/annotations.py::prepare_annotations`.

1. Raw RGB images (`data_root/*.jpg`) and the COCO-style annotation JSON
   (`data_root/Json_Output/*.json`) are scanned.
2. RGB images are EXIF-transposed to align with the (already upright) mask images.
3. Per-class binary silhouette masks (`Mask/<class>/`) are thresholded at 127 to boolean masks and
   combined into a semantic mask (label IDs 1-6, 0=background); pixels claimed by more than one
   class mask are counted as "conflict pixels" and reported in the audit (the last class in
   `classes` order wins for those pixels). These per-pixel masks are what the segmentation task
   trains on.
4. **Instances (and therefore classification/detection crops) are built directly from the COCO
   annotation's per-instance `bbox`/`segmentation`/`category_id` fields -- not from connected
   components of the binary masks.** Each instance's bbox is padded by `annotation.crop_padding`
   (expanded to `annotation.crop_min_size` if still too small), cropped from the RGB image, and
   saved to `06_classification/crops/<class>/<instance_id>.jpg`.

   *Why not mask connected components, like the blueberry benchmark does:* this dataset's mask
   images are lossy JPEGs, and JPEG compression ringing near hard silhouette edges frequently
   splits one tomato's mask into two or more separate blobs. On this release that produced a
   ~2x systematic overcount (roughly 19,000 mask-derived "instances" vs. 9,260 actual COCO
   annotations) -- i.e. a large fraction of mask-component crops would have been a single tomato
   cut in half. The COCO polygons are the authoritative, curated instance boundaries, so they
   decide crop count and boxes instead. The mask connected-component count is still computed and
   reported in `00_data_audit/dataset_audit.md` (`class_component_total` / per-class `*_components`
   columns) purely as a mask-quality QA signal, not as ground truth.
5. If `annotation.stem_trim_top_ratio > 0`, a thin band is cropped from the top of each padded box
   before saving, to reduce the influence of the green calyx/stem scar (see "Stem considerations"
   below). **Disabled by default** -- validate on real crops before enabling.
6. The `Mask/Overall/` binary mask is connected-component-counted as an independent
   "any tomato present" instance count, for the same QA purpose as (4).
7. Per-image tomato counts (used by the counting task) are read from the COCO JSON (grouping
   annotations by image + category).
8. A deterministic train/val/test split is generated (stratified by dominant ripeness class and a
   count quartile, `sklearn.model_selection.train_test_split`, seed from `split.seed`), unless
   `paths.fixed_split_manifest` points at a pre-existing split CSV.
9. Audit figures, an overlay contact sheet, and per-split COCO/YOLO detection exports are written.

## Color Space Considerations

Ripeness in tomato is fundamentally a **hue rotation**: green (~120 deg hue) through yellow/orange
(Breakers/Turning) to red (~0/360 deg hue). This is a much larger swing across the visible spectrum
than blueberry ripeness, which mostly varies in lightness/saturation within a narrower blue-purple
hue range. Two consequences:

- Training augmentation applies brightness/contrast/saturation jitter fairly aggressively
  (`jitter_strength`) but hue jitter very conservatively (`hue_jitter`, default `0.02` on
  torchvision's `[-0.5, 0.5]` hue scale) -- see `datasets.py::_color_jitter`. A larger hue jitter
  risks turning a `Turning` crop into something that looks like `Pink`, corrupting the label.
- If you later add classical (non-CNN) color-threshold heuristics, do them in HSV space, not RGB --
  hue is where the ripeness signal concentrates and it's far more lighting-invariant there.

## Stem/Calyx Considerations

The calyx (green stem cap) attaches at the top of a tomato in these top-down/angled captures and
is visually similar in hue to the `Green` class. Two mitigations are available, both conservative
by default:

- `annotation.stem_trim_top_ratio` (default `0.0`): crops away a fixed top fraction of each padded
  instance box before saving. This is a blunt heuristic -- it doesn't detect the calyx, it just
  assumes it's near the top -- so an aggressive setting can cut into small or rotated fruit. Tune
  it only after visually reviewing `06_classification/crops/` for your actual images.
- Padding itself (`annotation.crop_padding`, default `0.18`) is kept modest for the same reason:
  a smaller pad means less leaf/stem material enters the crop in the first place.

## Shape Considerations

Tomato instances vary far more in apparent size/aspect ratio than blueberries (camera distance,
cluster pose, partial occlusion), so:

- Crops use a padded axis-aligned bounding box per instance rather than a fixed-size circular
  crop, and are resized (not center-cropped) to the model's input size.
- Classification augmentation includes random rotation (`augmentation.classification.rotation_degrees`)
  to build orientation invariance, since a single fruit's ripeness doesn't depend on camera roll.

Primary script: `scripts/prepare_annotations.py`.
