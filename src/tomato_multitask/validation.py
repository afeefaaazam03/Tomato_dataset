from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage

FRAME_PATTERN = re.compile(r"^(\d{8})_(\d{6})_(\d+)\.jpg$", re.IGNORECASE)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_quality_metrics(image: Image.Image) -> dict[str, float]:
    """Brightness, saturation and a Laplacian-variance sharpness proxy on a downsampled copy."""
    small = image.convert("RGB").resize((512, int(512 * image.height / image.width)), Image.Resampling.BILINEAR)
    arr = np.asarray(small).astype(np.float32)
    gray = arr.mean(axis=2)
    hsv = np.asarray(small.convert("HSV")).astype(np.float32)
    laplacian = ndimage.laplace(gray)
    return {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "saturation_mean": float(hsv[..., 1].mean()),
        "laplacian_var": float(laplacian.var()),
    }


def average_hash(image: Image.Image, hash_size: int = 8) -> int:
    """Cheap perceptual hash (no extra dependency) for near-duplicate screening."""
    small = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float32)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def parse_frame_name(name: str) -> tuple[str, str, int] | None:
    match = FRAME_PATTERN.match(name)
    if not match:
        return None
    date, time, frame = match.groups()
    return date, time, int(frame)


def run_validation(image_root: Path, image_manifest: pd.DataFrame, classes: list[str]) -> dict[str, Any]:
    """Compute the technical-validation statistics a Scientific Data descriptor needs, straight
    from the real files (not the mask-connected-component route, which annotations.py's own audit
    already shows is noisy on this release -- see dataset_audit.md)."""
    image_paths = sorted(image_root.glob("*.jpg"))

    rows = []
    hashes: dict[str, list[str]] = {}
    ahash_by_name: dict[str, int] = {}
    orientation_counter: Counter[str] = Counter()
    session_frames: dict[tuple[str, str], list[int]] = {}

    for path in image_paths:
        with Image.open(path) as raw:
            exif = raw.getexif()
            orientation_tag = exif.get(274)  # 274 = Orientation
            width, height = raw.size
            oriented = ImageOps.exif_transpose(raw).convert("RGB")
            quality = image_quality_metrics(oriented)
            ahash_by_name[path.name] = average_hash(oriented)

        digest = sha256_of(path)
        hashes.setdefault(digest, []).append(path.name)
        orientation_counter[str(orientation_tag) if orientation_tag is not None else "none"] += 1

        parsed = parse_frame_name(path.name)
        if parsed:
            date, time, frame = parsed
            session_frames.setdefault((date, time), []).append(frame)

        rows.append(
            {
                "filename": path.name,
                "width": width,
                "height": height,
                "orientation": "portrait" if height > width else "landscape",
                "exif_orientation_tag": orientation_tag,
                "sha256": digest,
                **quality,
            }
        )

    quality_df = pd.DataFrame(rows)

    # Exact duplicates
    exact_duplicate_groups = {digest: names for digest, names in hashes.items() if len(names) > 1}

    # Near-duplicates (perceptual hash, Hamming distance <= 4 out of 64 bits)
    names = list(ahash_by_name.keys())
    near_duplicate_pairs = []
    threshold = 4
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist = hamming(ahash_by_name[names[i]], ahash_by_name[names[j]])
            if dist <= threshold:
                near_duplicate_pairs.append((names[i], names[j], dist))

    # Session / frame-interval analysis (filename-derived acquisition provenance)
    session_summary = []
    for (date, time), frames in sorted(session_frames.items()):
        frames_sorted = sorted(frames)
        diffs = [b - a for a, b in zip(frames_sorted, frames_sorted[1:])]
        diff_counts = Counter(diffs)
        modal_interval = diff_counts.most_common(1)[0][0] if diff_counts else None
        session_summary.append(
            {
                "date": date,
                "time": time,
                "n_frames": len(frames_sorted),
                "frame_min": frames_sorted[0],
                "frame_max": frames_sorted[-1],
                "modal_frame_interval": modal_interval,
                "distinct_intervals": dict(sorted(diff_counts.items())),
            }
        )

    # Mask overlap / union-vs-overall audit at per-image granularity
    overlap_images = 0
    union_mismatch_images = 0
    for row in image_manifest.itertuples(index=False):
        conflict = getattr(row, "mask_conflict_pixels", 0)
        if conflict and conflict > 0:
            overlap_images += 1
        class_pixels_total = getattr(row, "class_pixels_total", None)
        foreground_pixels = getattr(row, "foreground_pixels", None)
        if class_pixels_total is not None and foreground_pixels is not None:
            if abs(int(class_pixels_total) - int(foreground_pixels)) > 0:
                union_mismatch_images += 1

    return {
        "n_images": len(image_paths),
        "quality_df": quality_df,
        "orientation_distribution": dict(orientation_counter),
        "resolution_counts": quality_df.groupby(["width", "height"]).size().to_dict(),
        "portrait_landscape_counts": quality_df["orientation"].value_counts().to_dict(),
        "exact_duplicate_groups": exact_duplicate_groups,
        "unique_hashes": len(hashes),
        "near_duplicate_pairs": near_duplicate_pairs,
        "session_summary": session_summary,
        "images_with_class_overlap": overlap_images,
        "images_with_union_mismatch": union_mismatch_images,
        "brightness_mean": float(quality_df["brightness_mean"].mean()),
        "brightness_std": float(quality_df["brightness_mean"].std()),
        "sharpness_mean": float(quality_df["laplacian_var"].mean()),
        "saturation_mean": float(quality_df["saturation_mean"].mean()),
    }
