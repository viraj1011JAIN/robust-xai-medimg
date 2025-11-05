import numpy as np
import torch
from PIL import Image


def test_nih_transform_ndarray_channels_first(tmp_path):
    """
    Covers the ndarray path in NIHBinarizedDataset._apply_transform where
    the transform returns a channels-first ndarray shape [3, H, W].
    This hits the branch that was previously unexecuted in coverage.
    """
    from src.data.nih_binary import NIHBinarizedDataset

    # 1) Make a tiny RGB image
    img_h, img_w = 8, 8
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    img_path = tmp_path / "img.png"
    Image.fromarray(img).save(img_path)

    # 2) Write a minimal CSV with one row and one class column
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Image,Foo\nimg.png,1\n", encoding="utf-8")

    # 3) Build dataset with a transform that RETURNS an ndarray in [3, H, W]
    def ch_first_ndarray_transform(_):
        return np.zeros((3, img_h, img_w), dtype=np.uint8)

    ds = NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["Foo"],
        images_root=str(tmp_path),
        img_size=img_h,
        transform=ch_first_ndarray_transform,
    )

    # 4) Fetch one item; this triggers the ndarray path in _apply_transform
    x, y, meta = ds[0]

    # Assertions: tensor returned, channels-first, float dtype in [0,1]
    assert isinstance(x, torch.Tensor)
    assert x.ndim == 3 and x.shape[0] == 3 and x.shape[1] == img_h and x.shape[2] == img_w
    assert x.dtype.is_floating_point
    assert torch.all(x >= 0) and torch.all(x <= 1)
    assert y.shape[-1] == 1  # one class
    assert isinstance(meta, dict)
