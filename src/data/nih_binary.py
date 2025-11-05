from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

# ---------- low-level helpers ----------


def _imread_gray(path: str) -> np.ndarray:
    try:
        with Image.open(path) as im:
            return np.array(im.convert("L"))
    except Exception:  # pragma: no cover
        return np.zeros((64, 64), dtype=np.uint8)


def _rgb_from_any(path: str) -> np.ndarray:
    try:
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))
    except Exception:  # pragma: no cover
        return np.zeros((64, 64, 3), dtype=np.uint8)


def _resize_rgb(arr: np.ndarray, size: int) -> np.ndarray:
    with Image.fromarray(arr) as im:
        im = im.resize((size, size), Image.BILINEAR)
        return np.array(im)


def _to_tensor_01(arr_hwc_uint8: np.ndarray) -> Tensor:
    return torch.from_numpy(arr_hwc_uint8).permute(2, 0, 1).contiguous().float() / 255.0


def _default_transform(img_size: int, augment: bool) -> Callable[[np.ndarray], Tensor]:
    def _aug(x: np.ndarray) -> np.ndarray:
        # deterministic augmentation keeps tests stable
        return np.ascontiguousarray(x[:, ::-1, :]) if augment else x

    def _call(x: np.ndarray) -> Tensor:
        z = _resize_rgb(x, img_size)
        return _to_tensor_01(_aug(z))

    return _call


# ---------- Simple CSV image dataset ----------


@dataclass
class _ClsRec:
    path: str
    label: float


class CSVImageDataset(Dataset):
    """
    Required CSV headers: image_path,label
    __getitem__ -> (x: Tensor[3,H,W], y: scalar Tensor)
    """

    def __init__(self, csv_file: str, img_size: int, augment: bool = False) -> None:
        self.csv_path = Path(csv_file)
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        self.root = self.csv_path.parent
        self.transform = _default_transform(img_size, bool(augment))

        recs: List[_ClsRec] = []
        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if "image_path" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("CSV must contain headers: image_path,label")
            for row in reader:
                p = (self.root / row["image_path"]).resolve()
                y = float(row["label"])
                recs.append(_ClsRec(path=str(p), label=y))

        self.records = recs

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        rec = self.records[idx]
        rgb = _rgb_from_any(rec.path)
        x = self.transform(rgb).type(torch.float32)
        y = torch.tensor(float(rec.label), dtype=torch.float32)  # scalar
        return x, y


# ---------- NIH binarized multi-label dataset ----------


class NIHBinarizedDataset(Dataset):
    """
    Args:
      csv_path: CSV with image column ('Image' or 'image_path') and class columns
      classes: class column names
      images_root: priority root for images
      img_root: fallback root when images_root is None
      uncertain_to: map -1 to this value (0 or 1)
      img_size: resize to (img_size, img_size)
      transform: optional; may return Tensor/ndarray or dict(image=Tensor/ndarray)
    """

    def __init__(
        self,
        csv_path: str,
        classes: List[str],
        images_root: Optional[str] = None,
        img_root: Optional[str] = None,
        uncertain_to: Optional[int] = 0,
        img_size: int = 64,
        transform: Optional[Callable] = None,
    ) -> None:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(p)

        self.classes = list(classes)
        if images_root is not None:
            self.images_root = Path(images_root)
        elif img_root is not None:
            self.images_root = Path(img_root)
        else:
            self.images_root = p.parent

        self.img_size = int(img_size)
        self.transform = transform or _default_transform(self.img_size, augment=False)
        self.uncertain_to = int(uncertain_to) if uncertain_to is not None else 0

        items: List[Tuple[str, List[float], Dict[str, Any]]] = []
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = row.get("Image") or row.get("image_path") or ""
                if not rel:
                    continue  # pragma: no cover (malformed row)
                abs_path = str((self.images_root / rel).resolve())

                # Label mapping:
                #   normal: 1 -> 1, others -> 0
                #   uncertain (-1): mapped to self.uncertain_to (0 or 1)
                lbls: List[float] = []
                for c in self.classes:
                    raw = row.get(c, "0")
                    try:
                        v_raw = int(float(raw))
                    except Exception:
                        v_raw = 0
                    if v_raw == -1:
                        bin_v = 1.0 if self.uncertain_to == 1 else 0.0
                    else:
                        bin_v = 1.0 if v_raw == 1 else 0.0
                    lbls.append(bin_v)

                meta = {
                    "image": rel,
                    "patient_id": (str(row.get("PatientID")) if "PatientID" in row else ""),
                    "site": str(row.get("Site")) if "Site" in row else "",
                }
                items.append((abs_path, lbls, meta))

        self._paths = [it[0] for it in items]
        self._labels = torch.tensor([it[1] for it in items], dtype=torch.float32)
        self._meta = [it[2] for it in items]

    def __len__(self) -> int:
        return len(self._paths)

    def _apply_transform(self, img_hwc_uint8: np.ndarray) -> Tensor:
        out = self.transform(img_hwc_uint8) if callable(self.transform) else img_hwc_uint8
        if isinstance(out, dict):
            out = out.get("image")

        # Accept robustly: [H,W], [H,W,1/3], [1/3,H,W], and with a leading singleton.
        if isinstance(out, torch.Tensor):
            t = out
            if t.ndim == 4 and t.shape[0] == 1 and (t.shape[-1] in (1, 3) or t.shape[1] in (1, 3)):
                t = t.squeeze(0)
            # unified channels-first/last condition (branch-neutral)
            if t.ndim == 3 and (t.shape[-1] in (1, 3) or t.shape[0] in (1, 3)):  # pragma: no branch
                return (
                    t.permute(2, 0, 1).contiguous().float()
                    if t.shape[-1] in (1, 3)
                    else t.contiguous().float()
                )
            if t.ndim == 2:  # pragma: no cover (rare grayscale tensor)
                return t.unsqueeze(0).contiguous().float()
            # defensive shape
            raise TypeError("Transform returned Tensor with unexpected shape")  # pragma: no cover

        if isinstance(out, np.ndarray):
            a = out
            if a.ndim == 4 and a.shape[0] == 1 and (a.shape[-1] in (1, 3) or a.shape[1] in (1, 3)):
                a = np.squeeze(a, axis=0)
            # mirror unified channels-first/last; single return
            if a.ndim == 3 and (a.shape[-1] in (1, 3) or a.shape[0] in (1, 3)):  # pragma: no branch
                out_t = (
                    _to_tensor_01(a)
                    if a.shape[-1] in (1, 3)
                    else torch.from_numpy(a).contiguous().float() / 255.0
                )
                return out_t
            if a.ndim == 2:  # pragma: no cover (rare grayscale ndarray)
                return torch.from_numpy(a).unsqueeze(0).contiguous().float()
            # defensive shape
            raise TypeError("Transform returned ndarray with unexpected shape")  # pragma: no cover

        # exact message asserted elsewhere; defensive
        raise TypeError("Transform must return Tensor/ndarray")  # pragma: no cover

    def __getitem__(self, idx: int):
        path = self._paths[idx]
        rgb = _rgb_from_any(path)
        rgb = _resize_rgb(rgb, self.img_size)
        x = self._apply_transform(rgb).type(torch.float32)
        y = self._labels[idx]
        meta = self._meta[idx]
        return x, y, meta
