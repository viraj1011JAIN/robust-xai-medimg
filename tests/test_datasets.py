import numpy as np

from src.data.cxr_datasets import NIHChestXray
from src.data.derm_datasets import ISICDataset
from src.data.transforms import cxr_val, derm_val


def test_cxr_shape_and_dtype(tmp_path):
    import pandas as pd
    from PIL import Image

    root = tmp_path / "imgs"
    root.mkdir()
    img_path = root / "a.jpg"
    Image.new("RGB", (8, 8)).save(img_path)
    csv = tmp_path / "cxr.csv"
    pd.DataFrame({"image_path": ["a.jpg"], "Atelectasis": [0], "Cardiomegaly": [1]}).to_csv(csv, index=False)
    ds = NIHChestXray(str(csv), str(root), transform=cxr_val(64), target_cols=["Atelectasis", "Cardiomegaly"])
    x, y = ds[0]
    import torch

    assert x.dtype == torch.float32
    assert y.dtype == np.float32
    assert tuple(x.shape[-2:]) == (64, 64)


def test_isic_single_label(tmp_path):
    import pandas as pd
    from PIL import Image

    root = tmp_path / "imgs"
    root.mkdir()
    img = root / "b.jpg"
    Image.new("RGB", (10, 10)).save(img)
    csv = tmp_path / "isic.csv"
    pd.DataFrame(
        {"image_path": ["b.jpg"], "label": [1], "center": ["X"], "age": [60], "sex": ["M"], "location": ["back"]}
    ).to_csv(csv, index=False)
    ds = ISICDataset(str(csv), str(root), transform=derm_val(32))
    x, y, meta = ds[0]
    assert y in {0, 1}
    assert tuple(x.shape[-2:]) == (32, 32)
    assert set(meta.keys()) == {"center", "age", "sex", "location"}
