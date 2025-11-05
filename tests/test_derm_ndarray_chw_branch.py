import numpy as np
import pandas as pd
import torch

from src.data.derm_datasets import ISICDataset


def _chw_numpy_transform(img_hwc_uint8: np.ndarray):
    # Return numpy in CHW to hit the line that converts CHW ndarray -> Tensor
    h, w, _ = img_hwc_uint8.shape
    chw = np.transpose(img_hwc_uint8, (2, 0, 1))  # shape (3,H,W)
    return chw


def test_isic_apply_transform_ndarray_chw(tmp_path):
    # minimal CSV + dummy image
    root = tmp_path / "imgs"
    root.mkdir()
    (root / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # tiny placeholder
    # write a real tiny image (3x3) so PIL can read:
    import PIL.Image as Image

    Image.new("RGB", (3, 3)).save(root / "a.png")

    df = pd.DataFrame({"image_path": ["a.png"], "label": [1]})
    csv = tmp_path / "isic.csv"
    df.to_csv(csv, index=False)

    ds = ISICDataset(
        csv_path=str(csv),
        images_root=str(root),
        transform=_chw_numpy_transform,
        target_col="label",
        uncertain_to=0,
    )
    x, y, meta = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 3 and x.ndim == 3
