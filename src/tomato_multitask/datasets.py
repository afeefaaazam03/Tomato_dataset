from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _mask(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L")


def _color_jitter(image: Image.Image, strength: float, hue_strength: float) -> Image.Image:
    """Brightness/contrast/saturation jitter plus a small, separately-scaled hue jitter.

    Ripeness in tomato is largely a hue rotation across the visible spectrum (green -> red),
    unlike blueberry where the class signal sits mostly in lightness/saturation on a narrower
    blue-purple range. A large random hue shift can therefore accidentally turn one ripeness
    class into the visual signature of another, so `hue_strength` defaults much smaller than the
    brightness/contrast/saturation `strength` and should be tuned cautiously.
    """
    if strength > 0:
        brightness = 1.0 + np.random.uniform(-strength, strength)
        contrast = 1.0 + np.random.uniform(-strength, strength)
        saturation = 1.0 + np.random.uniform(-strength, strength)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
    if hue_strength > 0:
        image = F.adjust_hue(image, float(np.random.uniform(-hue_strength, hue_strength)))
    return image


def _rotate(image: Image.Image, max_degrees: float) -> Image.Image:
    if max_degrees <= 0:
        return image
    angle = float(np.random.uniform(-max_degrees, max_degrees))
    if angle == 0:
        return image
    fill = tuple(int(value) for value in image.resize((1, 1), Image.Resampling.BILINEAR).getpixel((0, 0)))
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=fill)


def _image_tensor(image: Image.Image, size: int, normalize: bool = True) -> torch.Tensor:
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    tensor = F.to_tensor(image)
    if normalize:
        tensor = F.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor


class CropClassificationDataset(Dataset):
    """One tomato instance crop -> one ripeness label.

    Augmentation follows the tomato-specific brief: horizontal flip, small random rotation
    (a single fruit is roughly rotation-invariant), brightness/contrast/saturation jitter, and a
    deliberately small hue jitter (see `_color_jitter`).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int,
        augment: bool,
        rotation_degrees: float = 15.0,
        jitter_strength: float = 0.15,
        hue_jitter: float = 0.02,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.rotation_degrees = float(rotation_degrees)
        self.jitter_strength = float(jitter_strength)
        self.hue_jitter = float(hue_jitter)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image = _rgb(row["crop_path"])
        if self.augment:
            if np.random.rand() < 0.5:
                image = ImageOps.mirror(image)
            image = _rotate(image, self.rotation_degrees)
            image = _color_jitter(image, self.jitter_strength, self.hue_jitter)
        tensor = _image_tensor(image, self.image_size, normalize=True)
        return tensor, torch.tensor(int(row["label"]), dtype=torch.long), str(row["crop_path"])


class CountingDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, target: str, augment: bool):
        self.frame = frame.reset_index(drop=True)
        self.image_size = int(image_size)
        self.target = target
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image = _rgb(row["image_path"])
        if self.augment:
            if np.random.rand() < 0.5:
                image = ImageOps.mirror(image)
            image = _color_jitter(image, strength=0.10, hue_strength=0.0)
        tensor = _image_tensor(image, self.image_size, normalize=True)
        if self.target != "total":
            raise NotImplementedError(f"Only target='total' is supported, got {self.target!r}")
        value = torch.tensor([float(row["Total"])], dtype=torch.float32)
        return tensor, value, str(row["image_path"])


class SegmentationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, augment: bool):
        self.frame = frame.reset_index(drop=True)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image = _rgb(row["image_path"])
        mask = _mask(row["semantic_mask_path"])
        if self.augment and np.random.rand() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if self.augment:
            image = _color_jitter(image, strength=0.10, hue_strength=0.0)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_tensor = F.normalize(F.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_tensor = torch.as_tensor(np.asarray(mask).copy(), dtype=torch.long)
        return image_tensor, mask_tensor, str(row["image_path"])


class DetectionDataset(Dataset):
    def __init__(
        self,
        image_frame: pd.DataFrame,
        instances_frame: pd.DataFrame,
        image_size: int,
        include_masks: bool = False,
        augment: bool = False,
    ):
        self.image_frame = image_frame.reset_index(drop=True)
        self.instances_by_stem = {
            stem: group.reset_index(drop=True) for stem, group in instances_frame.groupby("stem", sort=False)
        }
        self.image_size = int(image_size)
        self.include_masks = bool(include_masks)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.image_frame)

    def _instance_masks(self, rows: pd.DataFrame, original_size: tuple[int, int], flip: bool) -> torch.Tensor:
        """Rasterize each instance's stored COCO polygon into a full-image binary mask.

        Built straight from the polygon (not from the per-class mask JPEGs) so it isn't affected
        by the JPEG-compression instance-splitting artifacts documented in annotations.py.
        """
        width, height = original_size
        masks: list[torch.Tensor] = []
        for row in rows.itertuples(index=False):
            canvas = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(canvas)
            for part in json.loads(row.segmentation):
                if len(part) >= 6:
                    draw.polygon(list(zip(part[0::2], part[1::2])), fill=255)
            if flip:
                canvas = ImageOps.mirror(canvas)
            canvas = canvas.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            masks.append(torch.as_tensor((np.asarray(canvas).copy() > 0), dtype=torch.uint8))
        if not masks:
            return torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8)
        return torch.stack(masks, dim=0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.image_frame.iloc[index]
        image = _rgb(row["image_path"])
        original_width = int(row["aligned_width"])
        original_height = int(row["aligned_height"])
        rows = self.instances_by_stem.get(row["stem"], pd.DataFrame())
        flip = bool(self.augment and np.random.rand() < 0.5)
        if flip:
            image = ImageOps.mirror(image)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image_tensor = F.to_tensor(image)

        if rows.empty:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            masks = torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8)
        else:
            boxes_np = rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32).copy()
            if flip:
                x1 = boxes_np[:, 0].copy()
                x2 = boxes_np[:, 2].copy()
                boxes_np[:, 0] = original_width - x2
                boxes_np[:, 2] = original_width - x1
            boxes_np[:, [0, 2]] *= self.image_size / original_width
            boxes_np[:, [1, 3]] *= self.image_size / original_height
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels = torch.as_tensor(rows["det_label"].to_numpy(dtype=np.int64).copy(), dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            masks = (
                self._instance_masks(rows, original_size=(original_width, original_height), flip=flip)
                if self.include_masks
                else None
            )

        target: dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        if self.include_masks:
            target["masks"] = masks
        return image_tensor, target


def detection_collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[Any], list[Any]]:
    images, targets = zip(*batch)
    return list(images), list(targets)
