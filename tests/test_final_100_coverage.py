# tests/test_final_100_coverage.py
"""
Final comprehensive tests to achieve 100% coverage on all 5 modules.
These tests target the specific missing lines identified in the coverage report.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

# --------------------------
# CXR datasets
# --------------------------


def test_cxr_csv_validation_missing_columns(tmp_path):
    """Test CXR dataset CSV validation for missing columns."""
    from src.data.cxr_datasets import NIHChestXray

    # CSV missing image_path column
    csv_path = tmp_path / "bad1.csv"
    pd.DataFrame({"wrong": [1], "Atelectasis": [0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="image_path"):
        NIHChestXray(str(csv_path), str(tmp_path), target_cols=["Atelectasis"])


def test_cxr_csv_validation_missing_target_cols(tmp_path):
    """Test CXR dataset CSV validation for missing target columns."""
    from src.data.cxr_datasets import NIHChestXray

    # CSV missing target column
    csv_path = tmp_path / "bad2.csv"
    pd.DataFrame({"image_path": ["a.png"], "wrong": [0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Atelectasis"):
        NIHChestXray(str(csv_path), str(tmp_path), target_cols=["Atelectasis"])


def test_cxr_getitem_with_transform(tmp_path):
    """Test NIHChestXray __getitem__ with transform (line 98)."""
    from src.data.cxr_datasets import NIHChestXray
    from src.data.transforms import cxr_val

    # Create test image
    img_path = tmp_path / "imgs" / "test.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(img_path)

    # Create CSV
    csv_path = tmp_path / "data.csv"
    pd.DataFrame([{"image_path": "imgs/test.png", "Atelectasis": 1}]).to_csv(csv_path, index=False)

    ds = NIHChestXray(
        str(csv_path),
        str(tmp_path),
        transform=cxr_val(32),
        target_cols=["Atelectasis"],
    )
    x, y = ds[0]

    # With transform, y should be numpy.float32
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32


# --------------------------
# Derm datasets
# --------------------------


def test_derm_metadata_missing_columns(tmp_path):
    """Test ISIC dataset with missing metadata columns."""
    from src.data.derm_datasets import ISICDataset

    img_path = tmp_path / "imgs" / "test.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(img_path)

    # CSV without optional metadata columns
    csv_path = tmp_path / "minimal.csv"
    pd.DataFrame([{"image_path": "imgs/test.png", "label": 0}]).to_csv(csv_path, index=False)

    ds = ISICDataset(csv_path=str(csv_path), images_root=str(tmp_path))
    x, y, meta = ds[0]

    # Should have empty strings for missing metadata
    assert meta["center"] == ""
    assert meta["age"] == ""
    assert meta["sex"] == ""
    assert meta["location"] == ""


# --------------------------
# NIH binary
# --------------------------


def test_nih_csv_not_found(tmp_path):
    """Test NIHBinarizedDataset with missing CSV file."""
    from src.data.nih_binary import NIHBinarizedDataset

    with pytest.raises(FileNotFoundError):
        NIHBinarizedDataset(csv_path=str(tmp_path / "nonexistent.csv"), classes=["A"])


def test_csv_image_dataset_missing_file(tmp_path):
    """Test CSVImageDataset with missing file."""
    from src.data.nih_binary import CSVImageDataset

    with pytest.raises(FileNotFoundError):
        CSVImageDataset(csv_file=str(tmp_path / "missing.csv"), img_size=32)


def test_csv_image_dataset_invalid_headers(tmp_path):
    """Test CSVImageDataset with invalid headers."""
    from src.data.nih_binary import CSVImageDataset

    # CSV with wrong headers
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame([{"wrong1": "x", "wrong2": "y"}]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="image_path,label"):
        CSVImageDataset(csv_file=str(csv_path), img_size=32)


# --------------------------
# Export helpers / Grad-CAM integration
# --------------------------


def test_export_gradcam_fallback_paths(tmp_path):
    """Test _make_gc and _run_generate fallback paths."""
    from src.xai import export as E

    model = E.build_model()
    gc = E._make_gc(model)

    x = torch.rand(1, 3, 224, 224)
    result = E._run_generate(gc, x)
    assert isinstance(result, torch.Tensor)


def test_export_save_gradcam_various_heat_shapes(tmp_path):
    """Test save_gradcam_png with various heatmap shapes."""
    from src.xai import export as E

    model = E.build_model()

    # Batch size > 1 to cover averaging path
    x = torch.rand(2, 3, 224, 224)
    out_path = tmp_path / "cam_batch.png"
    E.save_gradcam_png(model, x, out_path)
    assert out_path.exists() and out_path.stat().st_size > 0

    # Different size to cover interpolation path
    x = torch.rand(1, 3, 128, 128)
    out_path2 = tmp_path / "cam_128.png"
    E.save_gradcam_png(model, x, out_path2)
    assert out_path2.exists() and out_path2.stat().st_size > 0


# --------------------------
# GradCAM class / functional API
# --------------------------


def test_gradcam_invalid_layer_name():
    """Test GradCAM with invalid layer name."""
    from src.xai.gradcam import GradCAM

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3))

    with pytest.raises(ValueError, match="not found"):
        GradCAM(model, target_layer_name="nonexistent_layer")


def test_gradcam_generate_with_two_args():
    """Test GradCAM generate() with (model, x) signature."""
    from src.xai.gradcam import GradCAM

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 2)

        def forward(self, x):
            x = self.layer4(x)
            feat = x
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x, feat

    model = TestModel().eval()
    cam = GradCAM(model, target_layer_name="layer4")

    x = torch.rand(1, 3, 32, 32)
    result = cam.generate(model, x)  # two-arg path triggers squeeze
    assert result.ndim == 2  # [H, W]


def test_gradcam_with_target_class_kwarg():
    """Test GradCAM with target_class kwarg."""
    from src.xai.gradcam import GradCAM

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 2)

        def forward(self, x):
            x = self.layer4(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = TestModel().eval()
    cam = GradCAM(model, target_layer_name="layer4")

    x = torch.rand(1, 3, 32, 32)
    result = cam.generate(x, target_class=1)
    assert isinstance(result, torch.Tensor)
    cam.remove()


def test_gradcam_logits_special_cases():
    """Test GradCAM with different logits shapes."""
    from src.xai.gradcam import GradCAM

    # Model that returns scalar logits
    class ScalarModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            x = self.layer4(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            out = self.fc(x).squeeze()  # scalar or 1D
            return out

    model = ScalarModel().eval()
    cam = GradCAM(model, target_layer_name="layer4")

    x = torch.rand(1, 3, 32, 32)
    result = cam.generate(x)
    assert isinstance(result, torch.Tensor)
    cam.remove()


def test_gradcam_functional_api():
    """Test gradcam() functional API."""
    from src.xai.gradcam import gradcam

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 2)

        def forward(self, x):
            x = self.layer4(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = TestModel().eval()
    x = torch.rand(2, 3, 32, 32)

    result = gradcam(model, x, class_idx=0, layer="layer4")
    assert result.shape == (2, 32, 32)


def test_gradcam_remove_hooks():
    """Test both remove() and remove_hooks() methods without invalid kwargs on Conv2d."""
    from src.xai.gradcam import GradCAM

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            # Give the conv layer the actual attribute name "layer4"
            self.layer4 = nn.Conv2d(3, 64, 3, padding=1)

        def forward(self, x):
            return self.layer4(x)

    model = Tiny().eval()

    # Initialize with the layer name as exposed via named_modules()
    cam = GradCAM(model, target_layer_name="layer4")

    # Exercise both removal APIs
    cam.remove()
    cam.remove_hooks()
