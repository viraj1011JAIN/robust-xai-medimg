# tests/test_derm_edges.py
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.derm_datasets import \
    ISICDataset  # adjust if your class name differs


def test_isic_none_transform_and_tensor(tmp_path):
    img = tmp_path / "imgs" / "a.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(12, 12, 3) * 255).astype("uint8")).save(img)

    csv = tmp_path / "isic.csv"
    pd.DataFrame(
        [
            {
                "image_path": "imgs/a.png",
                "label": 1,
                "center": "c",
                "age": 30,
                "sex": "m",
                "location": "b",
            }
        ]
    ).to_csv(csv, index=False)

    ds = ISICDataset(csv_path=str(csv), images_root=str(tmp_path), transform=None)
    x, y, meta = ds[0]
    assert isinstance(x, torch.Tensor) and x.ndim == 3 and x.shape[0] == 3
    assert y in (0, 1)
    assert {"center", "age", "sex", "location"} <= set(meta.keys())


def test_isic_transform_dict_and_bad_row_fields(tmp_path):
    img = tmp_path / "imgs" / "b.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(10, 10, 3) * 255).astype("uint8")).save(img)

    # Make csv with extra/unexpected fields to hit guards
    csv = tmp_path / "isic2.csv"
    pd.DataFrame(
        [
            {
                "image_path": "imgs/b.png",
                "label": -1,  # if your impl remaps -1 -> uncertain_to, this covers it
                "center": "",
                "age": "",
                "sex": "",
                "location": "",
                "weird": 123,
            }
        ]
    ).to_csv(csv, index=False)

    def _tfm(img_np):
        # albumentations-like: HWC uint8 -> dict{"image": Tensor}
        arr = img_np.astype("float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return {"image": t}

    ds = ISICDataset(
        csv_path=str(csv), images_root=str(tmp_path), transform=_tfm, uncertain_to=0
    )
    x, y, meta = ds[0]
    assert isinstance(x, torch.Tensor) and x.shape[0] == 3
    assert int(y) in (0, 1)  # -1 remapped or coerced path exercised
    assert "image" in meta and isinstance(meta["image"], str)
