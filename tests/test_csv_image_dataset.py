import csv
from pathlib import Path

import torch
from PIL import Image

from src.data.nih_binary import CSVImageDataset


def _write_img(p: Path, size=(16, 16), color=(128, 128, 128)):
    img = Image.new("RGB", size, color)
    img.save(p)


def test_csv_image_dataset_happy_path(tmp_path):
    root = Path(tmp_path)
    # images
    img1 = root / "a.jpg"
    img2 = root / "b.jpg"
    _write_img(img1)
    _write_img(img2)

    # csv (headers: image_path,label)
    csv_path = root / "data.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label"])
        w.writeheader()
        w.writerow({"image_path": "a.jpg", "label": 0})
        w.writerow({"image_path": "b.jpg", "label": 1})

    ds = CSVImageDataset(str(csv_path), img_size=32, augment=False)
    assert len(ds) == 2

    x0, y0 = ds[0]
    assert isinstance(x0, torch.Tensor)
    assert y0.dtype.is_floating_point
    assert x0.shape[-1] == 32 and x0.shape[-2] == 32  # H=W=32 after transform


def test_csv_image_dataset_missing_headers_raises(tmp_path):
    root = Path(tmp_path)
    bad_csv = root / "bad.csv"
    with bad_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["foo", "bar"])
        w.writeheader()
        w.writerow({"foo": "a.jpg", "bar": 0})
    try:
        _ = CSVImageDataset(str(bad_csv), img_size=32, augment=False)
        raised = False
    except ValueError as e:
        raised = "image_path,label" in str(e)
    assert raised
