import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.nih_binary import NIHBinarizedDataset


def _csv(tmp_path):
    p = tmp_path / "nih.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Image", "A"])
        w.writeheader()
        w.writerow({"Image": "x.png", "A": "1"})
    return p


def _img(tmp_path):
    imgp = tmp_path / "x.png"
    Image.fromarray((np.random.rand(8, 8) * 255).astype("uint8")).save(imgp)
    return imgp


def test_transform_returns_2d_ndarray(tmp_path):
    csvp = _csv(tmp_path)
    _img(tmp_path)

    # Transform that returns an HxW uint8 ndarray
    def tf(_img_hwc_uint8):
        return (np.random.rand(8, 8) * 255).astype("uint8")

    ds = NIHBinarizedDataset(
        csv_path=str(csvp),
        classes=["A"],
        images_root=str(tmp_path),
        img_size=8,
        transform=tf,
    )

    x, y, meta = ds[0]
    # This asserts the 2D -> unsqueeze(0) branch ran
    assert x.shape == (1, 8, 8)
    assert y.shape == (1,)
    assert meta["image"] == "x.png"
