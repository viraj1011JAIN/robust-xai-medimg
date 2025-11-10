# tests/test_transforms_coverage.py
"""
Additional tests to achieve 100% coverage for transforms.py
"""
import numpy as np
import torch

from src.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    cxr_train,
    cxr_val,
    derm_train,
    derm_val,
)


def _dummy_img(h=256, w=256):
    """Create a dummy uint8 image [0,255]."""
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


class TestConstants:
    """Test that constants are accessible and have correct values."""

    def test_imagenet_mean(self):
        assert IMAGENET_MEAN == (0.485, 0.456, 0.406)

    def test_imagenet_std(self):
        assert IMAGENET_STD == (0.229, 0.224, 0.225)


class TestCXRTransformsDirect:
    """Test CXR transforms by calling them directly."""

    def test_cxr_train_direct(self):
        """Test cxr_train function directly."""
        tfm = cxr_train(img_size=224)
        out = tfm(image=_dummy_img(300, 300))["image"]
        assert out.shape == (3, 224, 224)
        assert isinstance(out, torch.Tensor)

    def test_cxr_train_custom_size(self):
        """Test cxr_train with custom size."""
        tfm = cxr_train(img_size=512)
        out = tfm(image=_dummy_img(600, 600))["image"]
        assert out.shape == (3, 512, 512)

    def test_cxr_val_direct(self):
        """Test cxr_val function directly."""
        tfm = cxr_val(img_size=224)
        out = tfm(image=_dummy_img(300, 300))["image"]
        assert out.shape == (3, 224, 224)
        assert isinstance(out, torch.Tensor)

    def test_cxr_val_custom_size(self):
        """Test cxr_val with custom size."""
        tfm = cxr_val(img_size=384)
        out = tfm(image=_dummy_img(400, 400))["image"]
        assert out.shape == (3, 384, 384)


class TestDermTransformsDirect:
    """Test dermatology transforms by calling them directly."""

    def test_derm_train_direct(self):
        """Test derm_train function directly."""
        tfm = derm_train(img_size=224)
        out = tfm(image=_dummy_img(300, 300))["image"]
        assert out.shape == (3, 224, 224)
        assert isinstance(out, torch.Tensor)

    def test_derm_train_custom_size(self):
        """Test derm_train with custom size."""
        tfm = derm_train(img_size=512)
        out = tfm(image=_dummy_img(600, 600))["image"]
        assert out.shape == (3, 512, 512)

    def test_derm_val_direct(self):
        """Test derm_val function directly."""
        tfm = derm_val(img_size=224)
        out = tfm(image=_dummy_img(300, 300))["image"]
        assert out.shape == (3, 224, 224)
        assert isinstance(out, torch.Tensor)

    def test_derm_val_custom_size(self):
        """Test derm_val with custom size."""
        tfm = derm_val(img_size=384)
        out = tfm(image=_dummy_img(400, 400))["image"]
        assert out.shape == (3, 384, 384)
