# tests/test_transforms_behavior.py
import numpy as np
import torch

from src.data import transforms as T


def _dummy_img(h=256, w=256):
    """Create a dummy uint8 image [0,255]."""
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def test_cxr_train_transform_range_and_shape():
    """Test CXR training transforms produce correct output."""
    tfm = T.build_transforms(domain="cxr", split="train")
    out = tfm(image=_dummy_img(300, 300))["image"]

    assert out.shape == (3, 224, 224), "Should be 3x224x224"
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32
    assert float(out.min()) >= -5, "Min should be >= -5"
    assert float(out.max()) <= 5, "Max should be <= 5"


def test_derm_train_transform_range_and_shape():
    """Test dermatology training transforms."""
    tfm = T.build_transforms(domain="derm", split="train")
    out = tfm(image=_dummy_img(224, 224))["image"]

    assert out.shape == (3, 224, 224), "Should be 3x224x224"
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32


def test_cxr_val_is_deterministic():
    """Test that validation transforms are deterministic."""
    tfm = T.build_transforms(domain="cxr", split="val")
    img = _dummy_img(280, 280)

    a = tfm(image=img)["image"]
    b = tfm(image=img)["image"]

    # Validation should be deterministic (no random augmentations)
    assert torch.allclose(a, b, atol=1e-5), "Val transforms should be deterministic"


def test_derm_val_is_deterministic():
    """Test dermatology validation transforms are deterministic."""
    tfm = T.build_transforms(domain="derm", split="val")
    img = _dummy_img(224, 224)

    a = tfm(image=img)["image"]
    b = tfm(image=img)["image"]

    assert torch.allclose(a, b, atol=1e-5), "Val transforms should be deterministic"


def test_test_split_transforms():
    """Test that test split transforms work."""
    for domain in ["cxr", "derm"]:
        tfm = T.build_transforms(domain=domain, split="test")
        out = tfm(image=_dummy_img(256, 256))["image"]
        assert out.shape[0] == 3, f"Should have 3 channels for {domain}"


def test_transform_preserves_image_content():
    """Test that transforms preserve some image content."""
    np.random.seed(42)
    # Create an image with clear pattern
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[100:150, 100:150, :] = 255  # White square

    tfm = T.build_transforms(domain="cxr", split="val")
    out = tfm(image=img)["image"]

    # Check that transform produced valid output
    assert out.shape == (3, 224, 224), "Shape should be preserved"
    assert not torch.allclose(out, torch.zeros_like(out)), "Should not be all zeros"
