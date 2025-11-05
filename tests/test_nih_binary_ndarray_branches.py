import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.nih_binary import NIHBinarizedDataset, _to_tensor_01


def _make_min_csv(tmp_path, rows, fieldnames):
    p = tmp_path / "d.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _write_gray(img_path, size=(8, 8)):
    Image.fromarray((np.random.rand(*size) * 255).astype("uint8")).save(img_path)


def _dataset(tmp_path, transform=None, uncertain_to=0):
    csvp = _make_min_csv(
        tmp_path,
        [{"Image": "x.png", "A": "1"}],
        ["Image", "A"],
    )
    imgp = tmp_path / "x.png"
    _write_gray(imgp, (8, 8))
    return NIHBinarizedDataset(
        csv_path=str(csvp),
        classes=["A"],
        images_root=str(tmp_path),
        img_size=8,
        transform=transform,
        uncertain_to=uncertain_to,
    )


def test_apply_transform_numpy_2d(tmp_path):
    def tf(img):
        # return HxW ndarray
        return np.random.randint(0, 255, size=(8, 8), dtype=np.uint8)

    ds = _dataset(tmp_path, transform=tf)
    x, y, meta = ds[0]
    assert x.shape == (1, 8, 8)


def test_apply_transform_numpy_chw(tmp_path):
    def tf(img):
        # return CxHxW ndarray
        arr = np.random.randint(0, 255, size=(3, 8, 8), dtype=np.uint8)
        return arr

    ds = _dataset(tmp_path, transform=tf)
    x, _, _ = ds[0]
    assert x.shape == (3, 8, 8)


def test_apply_transform_numpy_nhwc_singleton(tmp_path):
    def tf(img):
        # return NxHxWxC with N=1 -> path squeezes leading singleton
        arr = np.random.randint(0, 255, size=(1, 8, 8, 3), dtype=np.uint8)
        return arr

    ds = _dataset(tmp_path, transform=tf)
    x, _, _ = ds[0]
    assert x.shape == (3, 8, 8)


def test_apply_transform_ndarray_invalid_raises(tmp_path):
    def tf(img):
        return np.zeros((8,), dtype=np.uint8)  # invalid shape triggers specific error

    ds = _dataset(tmp_path, transform=tf)
    with pytest.raises(TypeError, match="ndarray with unexpected shape"):
        _ = ds[0]


def test_uncertain_to_one_maps_to_one(tmp_path):
    # CSV row with -1 should map to 1 when uncertain_to=1
    csvp = _make_min_csv(
        tmp_path,
        [{"Image": "x.png", "A": "-1"}],
        ["Image", "A"],
    )
    imgp = tmp_path / "x.png"
    _write_gray(imgp, (8, 8))
    ds = NIHBinarizedDataset(
        csv_path=str(csvp),
        classes=["A"],
        images_root=str(tmp_path),
        img_size=8,
        uncertain_to=1,
    )
    _, y, _ = ds[0]
    assert float(y[0].item()) == 1.0
