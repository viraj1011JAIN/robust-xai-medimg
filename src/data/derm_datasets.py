from __future__ import annotations

import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(
        self, csv_path, images_root, label_col="label", meta_cols=("center", "age", "sex", "location"), transform=None
    ):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.label_col = label_col
        self.meta_cols = list(meta_cols)
        self.transform = transform

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
        y = int(row[self.label_col])
        meta = {k: row.get(k, None) for k in self.meta_cols}
        return img, y, meta
