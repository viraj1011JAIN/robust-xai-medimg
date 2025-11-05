# tests/test_cxr_all_edges.py
import numpy as np
import pandas as pd
import torch
from PIL import Image

import src.data.cxr_datasets as C


def _mk_img(p, w=10, h=8):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(h, w, 3) * 255).astype("uint8")).save(p)


def _albumentations_like_tfm(image):
    # Emulate A.Compose(...)(image=HWC np.uint8) -> {"image": Tensor[C,H,W]}
    arr = image.astype("float32") / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return {"image": t}


def test_cxr_transform_none_and_targets(tmp_path):
    imgp = tmp_path / "imgs" / "a.png"
    _mk_img(imgp)
    csv = tmp_path / "cxr.csv"
    pd.DataFrame({"image_path": ["imgs/a.png"], "A": [1], "B": [0]}).to_csv(
        csv, index=False
    )

    ds = C.NIHChestXray(str(csv), str(tmp_path), transform=None, target_cols=["A", "B"])
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and x.shape[0] == 3
    # y should be torch.float32 vector
    assert isinstance(y, torch.Tensor) and y.dtype == torch.float32 and y.numel() == 2


def test_cxr_albumentations_like_and_placeholder(tmp_path):
    # CSV points to a missing file -> placeholder path + albumentations-like transform
    csv = tmp_path / "data.csv"
    pd.DataFrame({"image_path": ["missing.png"], "A": [1]}).to_csv(csv, index=False)

    Base = getattr(C, "NIHChestXRayCXRBase", None) or getattr(C, "_CXRBase")
    ds = Base(
        csv_path=str(csv),
        images_root=str(tmp_path),
        target_cols=["A"],
        transform=_albumentations_like_tfm,
    )
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and x.shape[0] == 3
    assert isinstance(y, torch.Tensor) and y.dtype == torch.float32 and y.numel() == 1
