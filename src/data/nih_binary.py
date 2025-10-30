from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

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
            if "image_path" not in reader.fieldnames or "label" not in reader.fieldnames:
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
