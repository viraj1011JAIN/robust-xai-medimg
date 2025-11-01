from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class _CXRBase(Dataset):
    def __init__(
        self, csv_path, images_root, target_cols, transform=None, dtype=np.float32
    ):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.target_cols = list(target_cols)
        self.transform = transform
        self.dtype = dtype
        self.df[self.target_cols] = (
            self.df[self.target_cols].fillna(0).astype(self.dtype)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, row["image_path"])
        img = Image.open(img_path).convert("RGB")
        import numpy as _np

        img = _np.array(img)
        if self.transform:
            img = self.transform(image=img)["image"]
        y = self.df.loc[row.name, self.target_cols].values.astype(self.dtype)
        return img, y


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
            csv_path, images_root, target_cols or self.DEFAULT_TARGETS, transform
        )


class PadChest(_CXRBase):
    def __init__(self, csv_path, images_root, transform=None, target_cols=None):
        super().__init__(csv_path, images_root, target_cols, transform)


class VinDrCXR(_CXRBase):
    def __init__(self, csv_path, images_root, transform=None, target_cols=None):
        super().__init__(csv_path, images_root, target_cols, transform)
