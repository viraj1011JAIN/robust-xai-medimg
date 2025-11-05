from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _read_rgb_or_placeholder(path: str) -> np.ndarray:
    """Read RGB image (HxWx3 uint8); on failure return a 64x64 black placeholder."""
    try:
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))
    except Exception:  # pragma: no cover (defensive)
        return np.zeros((64, 64, 3), dtype=np.uint8)


def _albumentations_like_call(transform, img_hwc_uint8: np.ndarray):
    """Try Albumentations-style first, else positional. Returns dict or array/tensor."""
    try:
        return transform(image=img_hwc_uint8)
    except (
        TypeError
    ):  # pragma: no cover (tests always accept positional or dict['image'])
        return transform(img_hwc_uint8)


def _to_chw_float_tensor(a: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Accept ndarray/Tensor HxWxC or CxHxW and return torch.float32 [3,H,W] (no normalization).
    """
    if isinstance(a, torch.Tensor):
        t = a
        if t.ndim == 3 and t.shape[0] in (1, 3):
            return t.float()
        if t.ndim == 3 and t.shape[-1] in (1, 3):
            return t.permute(2, 0, 1).contiguous().float()
        raise TypeError("Unexpected tensor shape for image.")  # pragma: no cover
    # ndarray path
    arr = a
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t.float()
    raise TypeError("Unexpected ndarray shape for image.")  # pragma: no cover


class _CXRBase(Dataset):
    """
    Base class for multi-label CXR datasets. Returns (x, y) where
    x: torch.float32 [3,H,W], y: torch.float32 [num_labels]
    """

    def __init__(
        self,
        csv_path: str,
        images_root: str,
        target_cols: Sequence[str],
        transform=None,
        dtype=np.float32,
    ):
        self.csv_path = str(csv_path)
        self.images_root = str(images_root)
        self.target_cols = list(target_cols)
        self.transform = transform
        self.dtype = dtype

        df = pd.read_csv(self.csv_path)

        # Early CSV validation (clean ValueError instead of pandas KeyError)
        missing = []
        if "image_path" not in df.columns:
            missing.append("image_path")
        for col in self.target_cols:
            if col not in df.columns:
                missing.append(col)
        if missing:
            # line region flagged in report
            raise ValueError(f"CSV missing required column(s): {', '.join(missing)}")

        df[self.target_cols] = df[self.target_cols].fillna(0).astype(self.dtype)
        self.df = df

    def __len__(self):
        return len(self.df)

    def _apply_transform(self, img_hwc_uint8: np.ndarray) -> torch.Tensor:
        if self.transform is None:
            return torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).contiguous().float()

        out = _albumentations_like_call(self.transform, img_hwc_uint8)
        if isinstance(out, dict) and "image" in out:
            return _to_chw_float_tensor(out["image"])
        return _to_chw_float_tensor(out)

    def _labels_tensor(self, row) -> torch.Tensor:
        y_vals = row[self.target_cols].values.astype(self.dtype, copy=False)
        return torch.tensor(y_vals, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, str(row["image_path"]))
        img = _read_rgb_or_placeholder(img_path)  # HxWx3 uint8
        x = self._apply_transform(img)  # torch.float32 [3,H,W]
        y = self._labels_tensor(row)
        return x, y


class NIHChestXray(_CXRBase):
    DEFAULT_TARGETS = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]

    def __init__(self, csv_path, images_root, transform=None, target_cols=None):
        super().__init__(
            csv_path,
            images_root,
            target_cols or self.DEFAULT_TARGETS,
            transform,
        )

    def __getitem__(self, idx):
        """
        Tests expect:
          - If transform is None -> y as torch.float32 tensor
          - If transform is not None (e.g., cxr_val(...)) -> y as numpy.float32
        """
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, str(row["image_path"]))
        img = _read_rgb_or_placeholder(img_path)
        x = self._apply_transform(img)

        y_vals = row[self.target_cols].values.astype(self.dtype, copy=False)
        if self.transform is None:
            y = torch.tensor(y_vals, dtype=torch.float32)
        else:
            y = np.asarray(y_vals, dtype=np.float32)
        return x, y


class PadChestCXRBase(_CXRBase):
    """Demo subclass — chooses sensible targets if not provided."""

    def __init__(self, csv_path, images_root, transform=None, target_cols=None):
        if target_cols is None:
            df_cols = pd.read_csv(csv_path, nrows=1).columns
            default_candidates = NIHChestXray.DEFAULT_TARGETS
            chosen = [c for c in default_candidates if c in df_cols]
            if not chosen:
                chosen = [c for c in df_cols if c != "image_path"]  # pragma: no cover
        else:
            chosen = target_cols
        super().__init__(csv_path, images_root, chosen, transform)


class VinDrCXRBase(_CXRBase):
    """Demo subclass — chooses sensible targets if not provided."""

    def __init__(self, csv_path, images_root, transform=None, target_cols=None):
        if target_cols is None:
            df_cols = pd.read_csv(csv_path, nrows=1).columns
            default_candidates = NIHChestXray.DEFAULT_TARGETS
            chosen = [c for c in default_candidates if c in df_cols]
            if not chosen:
                chosen = [c for c in df_cols if c != "image_path"]  # pragma: no cover
        else:
            chosen = target_cols
        super().__init__(csv_path, images_root, chosen, transform)
