import numpy as np
import torch
from PIL import Image


def test_nih_transform_dict_image_ndarray_channels_last(tmp_path):
    """
    Covers the dict -> ndarray path in NIHBinarizedDataset._apply_transform:
        if isinstance(out, dict): out = out.get("image")
    Then uses an ndarray [H, W, 3] so _to_tensor_01(a) branch runs.
    """
    from src.data.nih_binary import NIHBinarizedDataset

    H = W = 8
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img_path = tmp_path / "img.png"
    Image.fromarray(img).save(img_path)

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Image,Foo\nimg.png,1\n", encoding="utf-8")

    # Transform returns a dict with an ndarray under "image"
    def dict_transform(_):
        return {"image": np.zeros((H, W, 3), dtype=np.uint8)}

    ds = NIHBinarizedDataset(
        csv_path=str(csv_path),
        classes=["Foo"],
        images_root=str(tmp_path),
        img_size=H,
        transform=dict_transform,
    )

    x, y, meta = ds[0]

    # Assert tensor, channels-first, correct dtype/shape
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, H, W)
    assert x.dtype.is_floating_point
    assert torch.all(x >= 0) and torch.all(x <= 1)
    assert y.shape[-1] == 1
    assert isinstance(meta, dict)
