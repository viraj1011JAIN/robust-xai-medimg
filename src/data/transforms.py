from __future__ import annotations

from typing import Literal

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics (torchvision convention)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_transforms",
    "cxr_train",
    "cxr_val",
    "derm_train",
    "derm_val",
]


# -----------------------------
# Domain-specific pipelines
# -----------------------------


def cxr_train(img_size: int = 224) -> A.BasicTransform:
    """
    CXR training augments.
    Conservative geometry, no color jitter (CXRs are effectively grayscale).
    """
    return A.Compose(
        [
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def cxr_val(img_size: int = 224) -> A.BasicTransform:
    """
    Deterministic CXR validation pipeline.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=int(img_size * 1.15)),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size, border_mode=0, value=0
            ),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def derm_train(img_size: int = 224) -> A.BasicTransform:
    """
    Dermatoscopic image training augments.
    Slightly richer geometry & mild color jitter, still safe for medical images.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, p=0.5),
            A.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.2
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def derm_val(img_size: int = 224) -> A.BasicTransform:
    """
    Deterministic Derm validation pipeline.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=int(img_size * 1.15)),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size, border_mode=0, value=0
            ),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# -----------------------------
# Factory expected by tests
# -----------------------------


def build_transforms(
    *,
    domain: Literal["cxr", "derm"],
    split: Literal["train", "val", "test"],
    size: int = 224,
) -> A.BasicTransform:
    """
    Return an Albumentations Compose that:
      - accepts {"image": HxWxC uint8}
      - returns {"image": Tensor[C,H,W]} normalized by ImageNet stats
      - train pipelines include random ops; val/test are deterministic

    Notes:
      - Tests only require correct shape (3, size, size) and normalized ranges;
        ImageNet normalization ensures outputs lie well within [-5, 5].
    """
    d = domain.lower()
    s = split.lower()
    if d not in {"cxr", "derm"}:
        raise ValueError(f"Unknown domain: {domain!r}")
    if s not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split!r}")

    if d == "cxr":
        return cxr_train(size) if s == "train" else cxr_val(size)
    else:  # derm
        return derm_train(size) if s == "train" else derm_val(size)
