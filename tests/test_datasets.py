# tests/test_datasets.py
import numpy as np
import torch

from src.data.cxr_datasets import NIHChestXray
from src.data.derm_datasets import ISICDataset
from src.data.transforms import cxr_val, derm_val


def test_cxr_shape_and_dtype(tmp_path):
    """Test CXR dataset shape and dtype."""
    import pandas as pd
    from PIL import Image

    root = tmp_path / "imgs"
    root.mkdir()
    img_path = root / "a.jpg"
    Image.new("RGB", (8, 8)).save(img_path)
    csv = tmp_path / "cxr.csv"
    pd.DataFrame(
        {"image_path": ["a.jpg"], "Atelectasis": [0], "Cardiomegaly": [1]}
    ).to_csv(csv, index=False)
    ds = NIHChestXray(
        str(csv),
        str(root),
        transform=cxr_val(64),
        target_cols=["Atelectasis", "Cardiomegaly"],
    )
    x, y = ds[0]

    assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
    assert y.dtype == np.float32, f"Expected numpy float32, got {y.dtype}"
    assert tuple(x.shape[-2:]) == (64, 64), f"Expected (64, 64), got {x.shape[-2:]}"


def test_isic_single_label(tmp_path):
    """Test ISIC dataset with single label."""
    import pandas as pd
    from PIL import Image

    root = tmp_path / "imgs"
    root.mkdir()
    img = root / "b.jpg"
    Image.new("RGB", (10, 10)).save(img)
    csv = tmp_path / "isic.csv"
    pd.DataFrame(
        {
            "image_path": ["b.jpg"],
            "label": [1],
            "center": ["X"],
            "age": [60],
            "sex": ["M"],
            "location": ["back"],
        }
    ).to_csv(csv, index=False)
    ds = ISICDataset(str(csv), str(root), transform=derm_val(32))
    x, y, meta = ds[0]

    # FIX: Extract value from tensor before checking
    if isinstance(y, torch.Tensor):
        y_val = float(y.item())
    else:
        y_val = float(y)

    assert y_val in {0.0, 1.0}, f"Label should be 0 or 1, got {y_val}"
    assert tuple(x.shape[-2:]) == (32, 32), f"Expected (32, 32), got {x.shape[-2:]}"
    assert set(meta.keys()) == {
        "image",
        "center",
        "age",
        "sex",
        "location",
    }, f"Expected keys with 'image', got {set(meta.keys())}"
