from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import class_slug, output_dirs


PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (64, 160, 43),    # Green
    2: (176, 196, 60),   # Breakers
    3: (230, 175, 60),   # Turning
    4: (233, 122, 150),  # Pink
    5: (219, 90, 70),    # Light Red
    6: (185, 30, 30),    # Red
}
MASK_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


@dataclass(frozen=True)
class PreparedPaths:
    image_manifest: Path
    instances: Path
    crops: Path
    audit_report: Path


def _normalise_exts(values: list[str]) -> set[str]:
    return {value.lower() if value.startswith(".") else f".{value.lower()}" for value in values}


def _find_mask(mask_dir: Path, image_name: str) -> Path:
    image_path = Path(image_name)
    candidates = [mask_dir / image_path.name]
    candidates.extend(mask_dir / f"{image_path.stem}{extension}" for extension in MASK_EXTENSIONS)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing mask for {image_name} in {mask_dir}")


def _binary_mask_dir(mask_root: Path, aliases: list[str]) -> Path:
    existing = {path.name.lower(): path for path in mask_root.iterdir() if path.is_dir()}
    for alias in aliases:
        found = existing.get(alias.lower())
        if found is not None:
            return found
    raise FileNotFoundError(f"Could not find an overall/binary mask folder from aliases: {aliases} in {mask_root}")


def _find_json(json_dir: Path) -> Path:
    candidates = sorted(json_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No COCO-style annotation JSON found in {json_dir}")
    return candidates[0]


def _polygon_bbox(segmentation: list[list[float]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for part in segmentation:
        xs.extend(part[0::2])
        ys.extend(part[1::2])
    return min(xs), min(ys), max(xs), max(ys)


def _load_coco(
    json_path: Path,
    classes: list[str],
    image_stems: set[str],
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Read a COCO-style instance annotation file: per-image class counts (for audit, playing the
    same role the Excel "Image Wise Classname Count" workbook plays in the blueberry benchmark)
    AND the actual per-instance polygon/bbox records used to build classification crops.

    Instances are taken directly from these COCO annotations rather than from connected components
    of the per-class binary masks. On this dataset the mask JPEGs (a lossy format for what should
    be a binary silhouette) introduce compression ringing near hard edges that occasionally splits
    one tomato's mask into two or more separate blobs -- the dataset audit cross-checks this
    directly (COCO annotation count vs. mask connected-component count) and it is a large,
    systematic overcount on this release. The COCO polygons are the authoritative, human/model
    -produced instance boundaries, so they -- not the derived mask components -- decide how many
    crops come out of an image and where their boxes are.
    """
    with json_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    category_name_by_id = {int(cat["id"]): str(cat["name"]) for cat in coco.get("categories", [])}
    class_by_norm = {name.strip().lower(): name for name in classes}
    category_to_class: dict[int, str] = {}
    unmatched_categories: list[str] = []
    for category_id, name in category_name_by_id.items():
        matched = class_by_norm.get(name.strip().lower())
        if matched is None:
            unmatched_categories.append(name)
        else:
            category_to_class[category_id] = matched
    if unmatched_categories:
        raise ValueError(
            f"{json_path} defines categories that are not in config.classes: {unmatched_categories}. "
            f"Configured classes: {classes}"
        )

    image_stem_by_id = {int(img["id"]): Path(str(img["file_name"])).stem for img in coco.get("images", [])}

    counter: Counter[tuple[str, str]] = Counter()
    instances_by_stem: dict[str, list[dict[str, Any]]] = {}
    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        stem = image_stem_by_id.get(image_id)
        if stem is None or stem not in image_stems:
            continue
        class_name = category_to_class.get(int(annotation["category_id"]))
        if class_name is None:
            continue
        counter[(stem, class_name)] += 1

        segmentation = annotation.get("segmentation") or []
        if "bbox" in annotation:
            bx, by, bw, bh = annotation["bbox"]
            x1, y1, x2, y2 = float(bx), float(by), float(bx) + float(bw), float(by) + float(bh)
        else:
            x1, y1, x2, y2 = _polygon_bbox(segmentation)
        area = float(annotation.get("area", (x2 - x1) * (y2 - y1)))
        instances_by_stem.setdefault(stem, []).append(
            {
                "annotation_id": int(annotation["id"]),
                "class_name": class_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area": area,
                "segmentation": segmentation,
            }
        )

    rows = []
    for stem in sorted(image_stems):
        row: dict[str, Any] = {"stem": stem}
        for name in classes:
            row[name] = int(counter.get((stem, name), 0))
        row["Total"] = int(sum(row[name] for name in classes))
        rows.append(row)
    df = pd.DataFrame(rows)
    audit = {
        "json_path": str(json_path),
        "coco_images": len(image_stem_by_id),
        "coco_annotations": int(sum(counter.values())),
        "matched_image_stems": int((df["Total"] >= 0).sum()),
    }
    return df, instances_by_stem, audit


def _fixed_split_from_manifest(image_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame | None:
    split_manifest = config.get("paths", {}).get("fixed_split_manifest")
    if not split_manifest:
        return None
    manifest_path = Path(split_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Configured fixed split manifest does not exist: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    if "split" not in manifest.columns:
        raise ValueError(f"{manifest_path} must contain a 'split' column.")
    if "stem" not in manifest.columns:
        if "image_id" in manifest.columns:
            manifest["stem"] = manifest["image_id"].astype(str)
        elif "filename" in manifest.columns:
            manifest["stem"] = manifest["filename"].astype(str).map(lambda value: Path(value).stem)
        else:
            raise ValueError(f"{manifest_path} must contain one of: stem, image_id, filename.")

    allowed = {"train", "val", "test"}
    manifest = manifest[["stem", "split"]].copy()
    manifest["stem"] = manifest["stem"].astype(str)
    manifest["split"] = manifest["split"].astype(str).str.lower()
    invalid = sorted(set(manifest["split"]) - allowed)
    if invalid:
        raise ValueError(f"{manifest_path} contains unsupported split labels: {invalid}")
    if manifest["stem"].duplicated().any():
        duplicated = manifest.loc[manifest["stem"].duplicated(), "stem"].head(5).tolist()
        raise ValueError(f"{manifest_path} contains duplicate split rows, for example: {duplicated}")

    split_by_stem = manifest.set_index("stem")["split"].to_dict()
    df = image_df.copy()
    df["split"] = df["stem"].map(split_by_stem)
    missing = df.loc[df["split"].isna(), "stem"].head(10).tolist()
    if missing:
        raise ValueError(f"{manifest_path} is missing split assignments for stems: {missing}")
    return df.sort_values("stem").reset_index(drop=True)


_SESSION_FRAME_PATTERN = re.compile(r"^(\d{8}_\d{6})_(\d+)$")


def _block_id(stem: str, block_size: int) -> str:
    """Group images into leakage-safe blocks of temporally-adjacent frames.

    This dataset's images are extracted video frames (see docs/Preprocessing_Pipeline.md /
    DataDescriptor_Validation_Findings.md): filenames cluster into a handful of acquisition
    sessions with frame numbers ~10 apart, so images a few frames apart within a session are very
    likely near-duplicate views of the same fruit and must never be split across train/val/test.
    There are only 3 raw sessions, too few groups on their own for a 70/15/15 stratified split, so
    each session is further divided into fixed-width frame-number blocks; whole blocks (not whole
    sessions, and never individual images) are assigned to a single split.
    """
    match = _SESSION_FRAME_PATTERN.match(stem)
    if not match:
        return stem
    session, frame = match.groups()
    return f"{session}_{int(frame) // block_size:04d}"


def _safe_split(image_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    fixed_df = _fixed_split_from_manifest(image_df, config)
    if fixed_df is not None:
        return fixed_df

    split_cfg = config["split"]
    seed = int(split_cfg.get("seed", 42))
    train_ratio = float(split_cfg["train"])
    val_ratio = float(split_cfg["val"])
    test_ratio = float(split_cfg["test"])
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0.")

    df = image_df.copy()
    class_cols = config["classes"]
    block_size = int(split_cfg.get("block_size", 100))
    df["block_id"] = df["stem"].map(lambda stem: _block_id(stem, block_size))

    blocks = df.groupby("block_id")[class_cols + ["Total"]].sum()
    blocks["dominant_class"] = blocks[class_cols].idxmax(axis=1)
    try:
        bins = pd.qcut(blocks["Total"], q=min(4, max(2, blocks["Total"].nunique())), duplicates="drop")
        strata = blocks["dominant_class"].astype(str) + "_" + bins.astype(str)
        if strata.value_counts().min() < 2:
            strata = blocks["dominant_class"].astype(str)
        if strata.value_counts().min() < 2:
            strata = None
    except Exception:
        strata = None

    train_blocks, holdout_blocks = train_test_split(
        blocks.index, train_size=train_ratio, random_state=seed, shuffle=True, stratify=strata
    )
    holdout_strata = None
    if strata is not None:
        holdout_strata = strata.loc[holdout_blocks]
        if holdout_strata.value_counts().min() < 2:
            holdout_strata = None
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_blocks, test_blocks = train_test_split(
        holdout_blocks, train_size=relative_val, random_state=seed + 1, shuffle=True, stratify=holdout_strata
    )
    split_by_block = {
        **{block: "train" for block in train_blocks},
        **{block: "val" for block in val_blocks},
        **{block: "test" for block in test_blocks},
    }
    df["split"] = df["block_id"].map(split_by_block)
    return df.drop(columns=["block_id"]).sort_values("stem").reset_index(drop=True)


def _load_mask(path: Path, threshold: int) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > threshold


def _expand_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int, padding: float, min_size: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(max(bw * padding, 1)))
    pad_y = int(round(max(bh * padding, 1)))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    if x2 - x1 < min_size:
        extra = min_size - (x2 - x1)
        x1 = max(0, x1 - extra // 2)
        x2 = min(width, x2 + extra - extra // 2)
    if y2 - y1 < min_size:
        extra = min_size - (y2 - y1)
        y1 = max(0, y1 - extra // 2)
        y2 = min(height, y2 + extra - extra // 2)
    return x1, y1, x2, y2


def _trim_stem_region(crop: Image.Image, stem_trim_top_ratio: float) -> Image.Image:
    """Optionally crop away a thin band at the top of the padded box.

    The calyx/stem scar sits at the top of a tomato in these top-down captures and is a similar
    green hue to the 'Green' ripeness class, which can bias color-based ripeness cues. Disabled
    by default (ratio 0.0) -- enable and tune only after visually checking real crops, since an
    overly aggressive trim can cut into the fruit itself on small/rotated instances.
    """
    if stem_trim_top_ratio <= 0:
        return crop
    width, height = crop.size
    trim_px = int(round(height * stem_trim_top_ratio))
    if trim_px <= 0 or trim_px >= height:
        return crop
    return crop.crop((0, trim_px, width, height))


def _semantic_from_class_masks(class_masks: dict[str, np.ndarray], classes: list[str]) -> tuple[np.ndarray, int]:
    shape = next(iter(class_masks.values())).shape
    semantic = np.zeros(shape, dtype=np.uint8)
    stacked = np.stack([class_masks[name] for name in classes], axis=0)
    conflict_pixels = int((stacked.sum(axis=0) > 1).sum())
    for class_id, name in enumerate(classes, start=1):
        semantic[class_masks[name]] = class_id
    return semantic, conflict_pixels


def _save_overlay(image: Image.Image, semantic: np.ndarray, path: Path) -> None:
    rgb = np.asarray(image.convert("RGB")).copy()
    overlay = rgb.copy()
    for label, color in PALETTE.items():
        if label == 0:
            continue
        mask = semantic == label
        overlay[mask] = np.array(color, dtype=np.uint8)
    blended = (0.65 * rgb + 0.35 * overlay).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended).save(path, quality=92)


def _save_yolo_image(image: Image.Image, target_path: Path, image_size: int) -> tuple[int, int]:
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(target_path, quality=92)
    return image_size, image_size


def _write_yolo_labels(rows: pd.DataFrame, label_path: Path, width: int, height: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows.itertuples(index=False):
        x1, y1, x2, y2 = float(row.x1), float(row.y1), float(row.x2), float(row.y2)
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{int(row.class_index)} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _coco_for_split(image_df: pd.DataFrame, instances_df: pd.DataFrame, classes: list[str]) -> dict[str, Any]:
    image_id_by_stem = {stem: idx + 1 for idx, stem in enumerate(image_df["stem"].tolist())}
    images = [
        {
            "id": image_id_by_stem[row.stem],
            "file_name": row.filename,
            "path": row.image_path,
            "width": int(row.aligned_width),
            "height": int(row.aligned_height),
        }
        for row in image_df.itertuples(index=False)
    ]
    annotations = []
    for ann_id, row in enumerate(instances_df.itertuples(index=False), start=1):
        width = int(row.x2 - row.x1)
        height = int(row.y2 - row.y1)
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id_by_stem[row.stem],
                "category_id": int(row.class_index) + 1,
                "bbox": [float(row.x1), float(row.y1), float(width), float(height)],
                "area": float(row.area),
                "iscrowd": 0,
            }
        )
    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(classes)]
    return {"images": images, "annotations": annotations, "categories": categories}


def _save_audit_plots(image_df: pd.DataFrame, instances_df: pd.DataFrame, dirs: dict[str, Path], classes: list[str]) -> None:
    figures = dirs["audit"] / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    image_df["Total"].hist(bins=24)
    plt.title("Tomato Instance Count per Image")
    plt.xlabel("Tomatoes")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(figures / "count_histogram.png", dpi=180)
    plt.close()

    count_by_class = image_df[classes].sum().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    count_by_class.plot(kind="bar", color="#4c78a8")
    plt.title("COCO-Annotation Tomato Counts by Ripeness Class")
    plt.ylabel("Instances")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "annotation_class_counts.png", dpi=180)
    plt.close()

    crop_counts = instances_df["class_name"].value_counts().reindex(classes, fill_value=0)
    plt.figure(figsize=(9, 5))
    crop_counts.plot(kind="bar", color="#59a14f")
    plt.title("Extracted Mask-Component Instance/Crop Counts by Class")
    plt.ylabel("Instances")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "extracted_instance_counts.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(image_df["Total"], image_df["class_component_total"], s=18, alpha=0.75)
    lim = max(float(image_df["Total"].max()), float(image_df["class_component_total"].max())) + 2
    plt.plot([0, lim], [0, lim], color="black", linewidth=1)
    plt.title("COCO Annotation Count vs Mask Connected Components")
    plt.xlabel("COCO annotation total")
    plt.ylabel("Mask component total")
    plt.tight_layout()
    plt.savefig(figures / "count_vs_components.png", dpi=180)
    plt.close()


def prepare_annotations(config: dict[str, Any], rebuild: bool = False) -> PreparedPaths:
    dirs = output_dirs(config)
    data_root = Path(config["data_root"])
    image_root = data_root / config["paths"].get("image_dir", ".")
    mask_root = data_root / config["paths"].get("mask_dir", "Mask")
    json_root = data_root / config["paths"].get("json_dir", "Json_Output")
    classes = list(config["classes"])
    ann_cfg = config.get("annotation", {})
    threshold = int(ann_cfg.get("mask_threshold", 127))
    min_area = int(ann_cfg.get("min_component_area", 64))
    crop_padding = float(ann_cfg.get("crop_padding", 0.18))
    crop_min_size = int(ann_cfg.get("crop_min_size", 20))
    save_overlay_count = int(ann_cfg.get("save_overlay_count", 12))
    stem_trim_top_ratio = float(ann_cfg.get("stem_trim_top_ratio", 0.0))

    image_manifest_path = dirs["annotations"] / "image_manifest.csv"
    instances_path = dirs["annotations"] / "instances.csv"
    crops_path = dirs["annotations"] / "classification_crops.csv"
    audit_report_path = dirs["audit"] / "dataset_audit.md"
    if not rebuild and image_manifest_path.exists() and instances_path.exists() and crops_path.exists():
        return PreparedPaths(image_manifest_path, instances_path, crops_path, audit_report_path)

    if not image_root.exists():
        raise FileNotFoundError(
            f"RGB image folder not found at {image_root}. The raw RGB photos are not bundled in the "
            "annotation/mask archive -- copy them here (matching the 'file_name' values in the COCO "
            "JSON) before running data preparation. See RUNNING_THE_CODE.md."
        )

    extensions = _normalise_exts(ann_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"]))
    image_paths = sorted(path for path in image_root.iterdir() if path.suffix.lower() in extensions)
    if not image_paths:
        raise FileNotFoundError(f"No RGB images found in {image_root}")
    image_stems = {path.stem for path in image_paths}

    json_path = _find_json(json_root)
    counts_df, coco_instances_by_stem, count_audit = _load_coco(json_path, classes, image_stems)
    counts_by_stem = counts_df.set_index("stem")

    overall_dir = _binary_mask_dir(mask_root, ann_cfg.get("overall_mask_aliases", ["Overall", "All", "ALL", "all"]))
    class_mask_dirs = {name: mask_root / name for name in classes}
    missing_class_dirs = [str(path) for name, path in class_mask_dirs.items() if not path.exists()]
    if missing_class_dirs:
        raise FileNotFoundError(f"Missing per-class mask folders: {missing_class_dirs}")

    semantic_dir = dirs["annotations"] / "semantic_masks"
    overlay_dir = dirs["audit"] / "overlays"
    crop_root = dirs["classification"] / "crops"
    yolo_root = dirs["detection"] / "yolo"
    coco_root = dirs["detection"] / "coco"
    for path in [semantic_dir, overlay_dir, crop_root, yolo_root, coco_root]:
        path.mkdir(parents=True, exist_ok=True)

    image_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    total_conflict_pixels = 0
    missing_masks: list[str] = []

    for image_idx, image_path in enumerate(tqdm(image_paths, desc="prepare tomato annotations")):
        if image_path.stem not in counts_by_stem.index:
            continue
        with Image.open(image_path) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
        aligned_width, aligned_height = image.size

        try:
            mask_paths = {name: _find_mask(class_mask_dirs[name], image_path.name) for name in classes}
            binary_mask_path = _find_mask(overall_dir, image_path.name)
        except FileNotFoundError as exc:
            missing_masks.append(str(exc))
            continue

        class_masks = {name: _load_mask(path, threshold=threshold) for name, path in mask_paths.items()}
        binary_mask = _load_mask(binary_mask_path, threshold=threshold)
        if binary_mask.shape != (aligned_height, aligned_width):
            raise ValueError(
                f"Mask/image shape mismatch for {image_path.name}: "
                f"mask={binary_mask.shape}, image={(aligned_height, aligned_width)}. "
                "Images and masks must share the same orientation (EXIF-transpose the RGB source "
                "if the masks were rendered from an already-rotated image)."
            )

        semantic, conflict_pixels = _semantic_from_class_masks(class_masks, classes)
        total_conflict_pixels += conflict_pixels
        semantic_path = semantic_dir / f"{image_path.stem}.png"
        Image.fromarray(semantic).save(semantic_path)
        if image_idx < save_overlay_count:
            _save_overlay(image, semantic, overlay_dir / f"{image_path.stem}_overlay.jpg")

        counts_row = counts_by_stem.loc[image_path.stem]

        # Mask-derived connected components are computed for the audit only (see _load_coco's
        # docstring) -- they are NOT used to decide crops, because the lossy JPEG mask encoding
        # systematically over-segments single tomatoes into multiple components on this release.
        class_component_counts: dict[str, int] = {}
        for class_name in classes:
            labeled, _ = ndimage.label(class_masks[class_name])
            kept_for_class = 0
            for component_slice in ndimage.find_objects(labeled):
                if component_slice is None:
                    continue
                if int((labeled[component_slice] > 0).sum()) >= min_area:
                    kept_for_class += 1
            class_component_counts[class_name] = kept_for_class

        class_index_by_name = {name: idx for idx, name in enumerate(classes)}
        per_class_seq: dict[str, int] = {}
        for instance in coco_instances_by_stem.get(image_path.stem, []):
            if instance["area"] < min_area:
                continue
            class_name = instance["class_name"]
            class_index = class_index_by_name[class_name]
            slug = class_slug(class_name)
            per_class_seq[class_name] = per_class_seq.get(class_name, 0) + 1
            instance_id = f"{image_path.stem}_{slug}_{per_class_seq[class_name]:04d}"
            x1, y1, x2, y2 = (int(round(v)) for v in (instance["x1"], instance["y1"], instance["x2"], instance["y2"]))
            crop_bbox = _expand_bbox((x1, y1, x2, y2), aligned_width, aligned_height, crop_padding, crop_min_size)
            crop = image.crop(crop_bbox)
            crop = _trim_stem_region(crop, stem_trim_top_ratio)
            crop_path = crop_root / slug / f"{instance_id}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path, quality=94)
            row = {
                "instance_id": instance_id,
                "annotation_id": instance["annotation_id"],
                "stem": image_path.stem,
                "filename": image_path.name,
                "image_path": str(image_path.resolve()),
                "semantic_mask_path": str(semantic_path.resolve()),
                "segmentation": json.dumps(instance["segmentation"]),
                "class_name": class_name,
                "class_slug": slug,
                "class_index": class_index,
                "det_label": class_index + 1,
                "area": instance["area"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_x1": crop_bbox[0],
                "crop_y1": crop_bbox[1],
                "crop_x2": crop_bbox[2],
                "crop_y2": crop_bbox[3],
                "crop_path": str(crop_path.resolve()),
            }
            instance_rows.append(row)
            crop_rows.append(
                {
                    "crop_path": str(crop_path.resolve()),
                    "image_path": str(image_path.resolve()),
                    "stem": image_path.stem,
                    "instance_id": instance_id,
                    "class_name": class_name,
                    "label": class_index,
                }
            )

        overall_labeled, _ = ndimage.label(binary_mask)
        overall_components_kept = 0
        for component_number, component_slice in enumerate(ndimage.find_objects(overall_labeled), start=1):
            if component_slice is None:
                continue
            component = overall_labeled[component_slice] == component_number
            if int(component.sum()) >= min_area:
                overall_components_kept += 1

        row = {
            "stem": image_path.stem,
            "filename": image_path.name,
            "image_path": str(image_path.resolve()),
            "binary_mask_path": str(binary_mask_path.resolve()),
            "semantic_mask_path": str(semantic_path.resolve()),
            "aligned_width": aligned_width,
            "aligned_height": aligned_height,
            "foreground_pixels": int(binary_mask.sum()),
            "class_pixels_total": int(sum(mask.sum() for mask in class_masks.values())),
            "mask_conflict_pixels": conflict_pixels,
            "overall_component_total": int(overall_components_kept),
            "class_component_total": int(sum(class_component_counts.values())),
        }
        for class_name in classes:
            row[class_name] = int(counts_row[class_name])
            row[f"{class_slug(class_name)}_components"] = int(class_component_counts[class_name])
            row[f"{class_slug(class_name)}_pixels"] = int(class_masks[class_name].sum())
        row["Total"] = int(counts_row["Total"])
        image_rows.append(row)

    if missing_masks:
        raise FileNotFoundError(f"Missing {len(missing_masks)} mask files. First missing: {missing_masks[:5]}")

    image_df = pd.DataFrame(image_rows)
    if image_df.empty:
        raise RuntimeError("No images survived annotation preparation.")
    image_df = _safe_split(image_df, config)
    split_by_stem = image_df.set_index("stem")["split"].to_dict()
    instances_df = pd.DataFrame(instance_rows)
    crops_df = pd.DataFrame(crop_rows)
    instances_df["split"] = instances_df["stem"].map(split_by_stem)
    crops_df["split"] = crops_df["stem"].map(split_by_stem)

    image_df.to_csv(image_manifest_path, index=False)
    instances_df.to_csv(instances_path, index=False)
    crops_df.to_csv(crops_path, index=False)
    for split_name in ["train", "val", "test"]:
        image_df[image_df["split"] == split_name].to_csv(dirs["splits"] / f"images_{split_name}.csv", index=False)
        instances_df[instances_df["split"] == split_name].to_csv(dirs["splits"] / f"instances_{split_name}.csv", index=False)
        crops_df[crops_df["split"] == split_name].to_csv(dirs["splits"] / f"crops_{split_name}.csv", index=False)

    _save_audit_plots(image_df, instances_df, dirs, classes)

    yolo_size = int(ann_cfg.get("yolo_image_size", 1024))
    yolo_yaml = yolo_root / "tomato.yaml"
    for split_name in ["train", "val", "test"]:
        split_images = image_df[image_df["split"] == split_name]
        split_instances = instances_df[instances_df["split"] == split_name]
        split_coco = _coco_for_split(split_images, split_instances, classes)
        with (coco_root / f"instances_{split_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(split_coco, handle, indent=2)
        for image_row in split_images.itertuples(index=False):
            source_image_path = Path(image_row.image_path)
            with Image.open(source_image_path) as raw_image:
                aligned = ImageOps.exif_transpose(raw_image).convert("RGB")
            yolo_image_path = yolo_root / "images" / split_name / source_image_path.name
            _save_yolo_image(aligned, yolo_image_path, yolo_size)
            label_rows = split_instances[split_instances["stem"] == image_row.stem]
            labels_scaled = label_rows.copy()
            labels_scaled["x1"] = labels_scaled["x1"] * yolo_size / int(image_row.aligned_width)
            labels_scaled["x2"] = labels_scaled["x2"] * yolo_size / int(image_row.aligned_width)
            labels_scaled["y1"] = labels_scaled["y1"] * yolo_size / int(image_row.aligned_height)
            labels_scaled["y2"] = labels_scaled["y2"] * yolo_size / int(image_row.aligned_height)
            _write_yolo_labels(labels_scaled, yolo_root / "labels" / split_name / f"{image_row.stem}.txt", yolo_size, yolo_size)

    yolo_yaml.write_text(
        "\n".join(
            [
                f"path: {yolo_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(classes)}",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(classes)],
                "",
            ]
        ),
        encoding="utf-8",
    )

    audit_lines = [
        "# Tomato Ripeness Benchmark Dataset Audit",
        "",
        f"- RGB images prepared: {len(image_df)}",
        f"- Crops extracted (one per COCO instance annotation): {len(instances_df)}",
        f"- Overall mask folder used: `{overall_dir}`",
        f"- COCO annotation file used: `{count_audit['json_path']}`",
        f"- COCO images referenced: {count_audit['coco_images']}",
        f"- COCO annotations matched to prepared images: {count_audit['coco_annotations']}",
        f"- Total class-mask overlap/conflict pixels (semantic mask, pixel-level, not instance count): {total_conflict_pixels}",
        "",
        "## Split Counts",
        "",
        image_df["split"].value_counts().rename_axis("split").reset_index(name="images").to_markdown(index=False),
        "",
        "## COCO Annotation Count Totals (source of truth for instances/crops)",
        "",
        image_df[classes + ["Total"]].sum().to_frame("count").to_markdown(),
        "",
        "## QA: COCO Instance Count vs. Mask-Derived Connected-Component Count",
        "",
        "**Instances/crops are built from the COCO polygon annotations, not from these mask "
        "components.** This section is a data-quality check only: the per-class mask images in "
        "this release are lossy JPEGs, and JPEG compression ringing near hard silhouette edges "
        "can split a single tomato's mask into multiple connected components. A large gap below "
        "means the mask images are visually noisier than the COCO annotations for that image, but "
        "it does **not** affect crop quality since crops don't derive from these components.",
        "",
        f"- Mean absolute image-level difference: "
        f"{(image_df['Total'] - image_df['class_component_total']).abs().mean():.3f}",
        f"- Max absolute image-level difference: "
        f"{int((image_df['Total'] - image_df['class_component_total']).abs().max())}",
        "",
        "Images with the largest disagreement (for spot-checking mask quality only):",
        "",
        image_df.assign(abs_diff=(image_df["Total"] - image_df["class_component_total"]).abs())[
            ["filename", "split", "Total", "class_component_total", "overall_component_total", "abs_diff"]
        ]
        .sort_values("abs_diff", ascending=False)
        .head(20)
        .to_markdown(index=False),
        "",
        "## Generated Artifacts",
        "",
        f"- Image manifest: `{image_manifest_path}`",
        f"- Instance boxes/crops: `{instances_path}`",
        f"- Crop classification manifest: `{crops_path}`",
        f"- COCO detection JSON: `{coco_root}`",
        f"- YOLO detection export: `{yolo_root}`",
    ]
    audit_report_path.write_text("\n".join(audit_lines), encoding="utf-8")

    return PreparedPaths(image_manifest_path, instances_path, crops_path, audit_report_path)


def copy_result_artifacts(config: dict[str, Any]) -> None:
    dirs = output_dirs(config)
    target = dirs["results"] / "dataset_artifacts"
    target.mkdir(parents=True, exist_ok=True)
    for source in [
        dirs["audit"] / "dataset_audit.md",
        dirs["annotations"] / "image_manifest.csv",
        dirs["annotations"] / "instances.csv",
        dirs["annotations"] / "classification_crops.csv",
    ]:
        if source.exists():
            shutil.copy2(source, target / source.name)
