# tests/test_cxr_datasets_len_and_subclasses.py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import src.data.cxr_datasets as C


def _mk_img(p, size=(16, 16)):
    """Helper to create a test image."""
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(*size, 3) * 255).astype("uint8")).save(p)


def _pick_public_cxr_class():
    """Select an available public CXR dataset class."""
    for name in ("PadChestCXRBase", "VinDrCXRBase"):
        if hasattr(C, name):
            return getattr(C, name)
    return None


def test_cxrbase_len_and_getitem(tmp_path):
    """Test CXR dataset __len__ and __getitem__ methods."""
    cls = _pick_public_cxr_class()
    if cls is None:
        pytest.skip("No public CXR subclass exported (PadChest/VinDr).")

    img = tmp_path / "imgs" / "a.png"
    _mk_img(img)
    csv = tmp_path / "data.csv"
    pd.DataFrame({"image_path": ["imgs/a.png"], "A": [1], "B": [0]}).to_csv(csv, index=False)

    ds = cls(
        csv_path=str(csv),
        images_root=str(tmp_path),
        target_cols=["A", "B"],
        transform=None,
    )
    assert len(ds) == 1, "Dataset should have 1 sample"
    item = ds[0]

    # Handle both (x, y) and (x, y, meta) return formats
    if isinstance(item, tuple) and len(item) == 3:
        x, y, _ = item
    else:
        x, y = item

    # Verify image has 3 channels (RGB)
    try:
        import torch

        if isinstance(x, torch.Tensor):
            assert x.shape[0] == 3 or x.shape[-1] == 3, "Should have 3 channels"
        else:
            assert x.shape[-1] == 3 or x.shape[0] == 3, "Should have 3 channels"
    except ImportError:
        assert 3 in x.shape, "Should have 3 channels"

    assert y.shape[-1] == 2, "Should have 2 target labels"


def test_subclass_inits_are_executed_if_present(tmp_path):
    """Test that subclass __init__ methods execute without errors."""
    img = tmp_path / "imgs" / "a.png"
    _mk_img(img)
    csv = tmp_path / "data.csv"

    # Create all required columns for CXR datasets
    cols = [
        "image_path",
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Nodule",
        "Mass",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
    ]
    row = {c: 0 for c in cols}
    row["image_path"] = "imgs/a.png"
    pd.DataFrame([row]).to_csv(csv, index=False)

    # Test instantiation of available subclasses
    for name in ("PadChestCXRBase", "VinDrCXRBase"):
        if hasattr(C, name):
            cls = getattr(C, name)
            ds = cls(
                csv_path=str(csv),
                images_root=str(tmp_path),
                transform=None,
                target_cols=None,
            )
            assert ds is not None, f"{name} should instantiate successfully"
