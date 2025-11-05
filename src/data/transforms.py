# src/data/transforms.py
"""
Transform pipelines for CXR and dermatology images using Albumentations.
All pipelines normalize by ImageNet statistics and return PyTorch tensors.
"""
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


def cxr_train(img_size: int = 224) -> A.Compose:
    """
    CXR training augmentations.

    Conservative geometry transformations without color jitter
    (chest X-rays are effectively grayscale).

    Args:
        img_size: Target image size (height and width)

    Returns:
        Albumentations Compose pipeline that expects HWC uint8 and
        returns CHW float32 tensor normalized by ImageNet stats
    """
    return A.Compose(
        [
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def cxr_val(img_size: int = 224) -> A.Compose:
    """
    Deterministic CXR validation/test pipeline.

    Resizes to fit within img_size, pads if needed, then center crops.
    No random augmentations for reproducible validation.

    Args:
        img_size: Target image size (height and width)

    Returns:
        Albumentations Compose pipeline for validation/test
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


def derm_train(img_size: int = 224) -> A.Compose:
    """
    Dermatoscopic image training augmentations.

    Includes richer geometric transforms and mild color jitter
    while remaining safe for medical imaging.

    Args:
        img_size: Target image size (height and width)

    Returns:
        Albumentations Compose pipeline for training
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


def derm_val(img_size: int = 224) -> A.Compose:
    """
    Deterministic dermatology validation/test pipeline.

    Resizes to fit within img_size, pads if needed, then center crops.
    No random augmentations for reproducible validation.

    Args:
        img_size: Target image size (height and width)

    Returns:
        Albumentations Compose pipeline for validation/test
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
# Factory function for tests
# -----------------------------


def build_transforms(
    *,
    domain: Literal["cxr", "derm"],
    split: Literal["train", "val", "test"],
    size: int = 224,
) -> A.Compose:
    """
    Factory function to build transform pipelines.

    Returns an Albumentations Compose pipeline based on domain and split:
      - Accepts dict with key "image" containing HxWxC uint8 numpy array
      - Returns dict with key "image" containing CxHxW float32 tensor
      - Normalizes using ImageNet statistics
      - Training splits use random augmentations
      - Validation and test splits are deterministic

    Args:
        domain: Medical imaging domain ('cxr' or 'derm')
        split: Data split ('train', 'val', or 'test')
        size: Target image size (default: 224)

    Returns:
        Albumentations Compose pipeline

    Raises:
        ValueError: If domain is not 'cxr' or 'derm'
        ValueError: If split is not 'train', 'val', or 'test'

    Examples:
        >>> tfm = build_transforms(domain="cxr", split="train", size=224)
        >>> img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        >>> result = tfm(image=img)
        >>> result["image"].shape  # torch.Size([3, 224, 224])

        >>> val_tfm = build_transforms(domain="derm", split="val")
        >>> # Deterministic validation transform
    """
    d = domain.lower()
    s = split.lower()

    # Validate domain
    if d not in {"cxr", "derm"}:
        raise ValueError(f"Unknown domain: {domain!r}. Must be 'cxr' or 'derm'.")

    # Validate split
    if s not in {"train", "val", "test"}:
        raise ValueError(
            f"Unknown split: {split!r}. Must be 'train', 'val', or 'test'."
        )

    # Select appropriate pipeline
    if d == "cxr":
        return cxr_train(size) if s == "train" else cxr_val(size)
    else:  # derm
        return derm_train(size) if s == "train" else derm_val(size)
