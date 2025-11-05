import numpy as np
import pandas as pd
from PIL import Image

import src.data.cxr_datasets as C


def _mk_img(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(8, 8, 3) * 255).astype("uint8")).save(p)


def _num_elements(y):
    # Works for both numpy arrays and torch tensors
    if hasattr(y, "numel"):
        return int(y.numel())
    if hasattr(y, "size"):
        # numpy.ndarray.size
        return int(y.size)
    # fallback for sequences
    try:
        return len(y)
    except Exception:
        return 1


def test_cxr_missing_image_returns_placeholder(tmp_path):
    # Build CSV that points to a non-existent image to hit defensive branch
    csv = tmp_path / "data.csv"
    pd.DataFrame({"image_path": ["missing.png"], "A": [1]}).to_csv(csv, index=False)

    # Prefer the public class; fall back to the internal base if not found
    Base = getattr(C, "NIHChestXray", None) or getattr(C, "_CXRBase")

    ds = Base(
        csv_path=str(csv),
        images_root=str(tmp_path),
        target_cols=["A"],
        transform=None,
    )
    x, y = ds[0]
    assert x.shape[0] == 3
    assert _num_elements(y) == 1
