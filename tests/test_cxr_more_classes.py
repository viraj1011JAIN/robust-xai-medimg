# tests/test_cxr_more_classes.py
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import src.data.cxr_datasets as C


def _mk_img(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(12, 12, 3) * 255).astype("uint8")).save(p)


def _albumentations_like_tfm(np_img):
    # Simulate Albumentations returning dict{"image": tensor[C,H,W]}
    t = torch.from_numpy(np_img.transpose(2, 0, 1)).float() / 255.0
    return {"image": t}


def test_padchest_and_vindr_ctor_and_transform(tmp_path):
    imgp = tmp_path / "imgs" / "x.png"
    _mk_img(imgp)
    csv = tmp_path / "cxr.csv"
    pd.DataFrame({"image_path": ["imgs/x.png"], "A": [1], "B": [0]}).to_csv(csv, index=False)

    for Cls in (getattr(C, "PadChest", None), getattr(C, "VinDrCXR", None)):
        if Cls is None:
            continue
        ds = Cls(
            str(csv),
            str(tmp_path),
            transform=_albumentations_like_tfm,
            target_cols=["A", "B"],
        )
        x, y = ds[0]
        assert isinstance(x, torch.Tensor) and x.shape[0] == 3
        assert isinstance(y, torch.Tensor) and y.dtype == torch.float32 and y.numel() == 2
