from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T  # needed for CSVImageDataset

# -----------------------------
# NIH binarized dataset (for tests)
# -----------------------------


@dataclass
class _NIHRow:
    image: str
    finding: str
    # per-class labels appear as columns named exactly like the classes in `classes`
    extras: Dict[str, str]  # for patient/site etc.


def _imread_gray(path: str) -> np.ndarray:
    """
    Read image as grayscale HxW uint8 array.
    Exists so tests can monkeypatch it.
    """
    img = Image.open(path).convert("L")
    return np.asarray(img)


class NIHBinarizedDataset(Dataset):
    """
    Minimal NIH CSV reader for tests.

    CSV columns expected:
      - Image (filename)
      - Finding (string; not used for label vector)
      - One column per class in `classes`
      - PatientID, Site (metadata)
    """

    def __init__(
        self,
        csv_path: str,
        classes: Sequence[str],
        uncertain_to: int = 0,
        transform=None,
        img_root: str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.uncertain_to = int(uncertain_to)
        self.transform = transform  # optional PIL/torch/Albumentations-like
        self.img_root = Path(img_root) if img_root else self.csv_path.parent

        rows: List[_NIHRow] = []
        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                extras = {
                    "PatientID": row.get("PatientID", ""),
                    "Site": row.get("Site", ""),
                }
                rows.append(
                    _NIHRow(
                        image=row["Image"],
                        finding=row.get("Finding", ""),
                        extras=extras,
                    )
                )
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def _labels_from_row(self, row_dict: Dict[str, str]) -> torch.Tensor:
        """
        Build a binary vector in class order.
        Values: 1 stays 1; 0 stays 0; -1 -> `uncertain_to` (0 or 1).
        """
        vec = np.zeros(len(self.classes), dtype=np.int64)
        for i, c in enumerate(self.classes):
            raw = row_dict.get(c, 0)
            try:
                v = int(raw)
            except Exception:
                v = 0
            if v == -1:
                v = self.uncertain_to
            vec[i] = 1 if v == 1 else 0
        return torch.tensor(vec, dtype=torch.int64)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # we need original row dict again to read class columns
        with self.csv_path.open("r", newline="") as f:
            reader = list(csv.DictReader(f))
            row = reader[idx]

        # image
        img_path = os.path.join(self.img_root, row["Image"])
        arr = _imread_gray(img_path)  # HxW uint8
        x = torch.from_numpy(arr).unsqueeze(0).float() / 255.0  # [1,H,W], 0..1

        # labels
        y = self._labels_from_row(row)  # torch int64 on CPU (has .numpy())

        # meta
        meta = {
            "patient_id": row.get("PatientID", ""),
            "site": row.get("Site", ""),
            "image": row.get("Image", ""),
        }

        # allow an optional transform to modify x (and possibly y)
        if self.transform is not None:
            out = self.transform(image=x) if callable(self.transform) else x
            # be tolerant: if transform returns a dict like Albumentations
            if isinstance(out, dict) and "image" in out:
                x = out["image"]
            elif isinstance(out, torch.Tensor):
                x = out

        return x, y, meta


# -----------------------------
# Generic CSVImageDataset (baseline import expects it here)
# -----------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class _Record:
    path: Path
    label: float


def _build_transform(img_size: int, augment: bool) -> T.Compose:
    if augment:
        return T.Compose(
            [
                T.Resize(int(img_size * 1.15)),
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(int(img_size * 1.15)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


class CSVImageDataset(Dataset):
    """
    Expects a CSV with headers:
        image_path,label
    where image_path is relative to the CSV directory.
    """

    def __init__(self, csv_file: str, img_size: int, augment: bool = False) -> None:
        self.csv_path = Path(csv_file)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.root = self.csv_path.parent
        self.transform = _build_transform(int(img_size), bool(augment))

        records: List[_Record] = []
        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if (
                "image_path" not in reader.fieldnames
                or "label" not in reader.fieldnames
            ):
                raise ValueError("CSV must contain headers: image_path,label")

            for row in reader:
                p = (self.root / row["image_path"]).resolve()
                y = float(row["label"])
                records.append(_Record(path=p, label=y))

        self._items = records
        # for class weighting / AUROC stats usage in training
        self.y = [r.label for r in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self._items[idx]
        img = Image.open(rec.path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(rec.label, dtype=torch.float)
        return x, y
