import csv

import numpy as np
import torch
from PIL import Image

from src.data.nih_binary import NIHBinarizedDataset


def _chw_numpy_transform(img_hwc_uint8: np.ndarray):
    h, w, _ = img_hwc_uint8.shape
    return np.transpose(img_hwc_uint8, (2, 0, 1))  # CHW ndarray


def test_nih_apply_transform_ndarray_chw(tmp_path):
    # csv with one sample and two classes
    csv_path = tmp_path / "nih.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image", "A", "B"])
        writer.writeheader()
        writer.writerow({"Image": "x.png", "A": "1", "B": "-1"})

    # image
    img_path = tmp_path / "x.png"
    Image.new("RGB", (3, 3)).save(img_path)

    ds = NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["A", "B"],
        images_root=str(tmp_path),
        uncertain_to=0,
        img_size=8,
        transform=_chw_numpy_transform,
    )
    x, y, meta = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 3 and x.ndim == 3
