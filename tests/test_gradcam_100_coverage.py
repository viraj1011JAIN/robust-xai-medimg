"""Tests for 100% coverage of src/xai/gradcam.py - covers lines 52, 127"""

import pytest
import torch
import torch.nn as nn

from src.xai.gradcam import GradCAM, get_gradcam, gradcam


class SimpleModel(nn.Module):
    """Simple model with layer4 for testing."""

    def __init__(self):
        super().__init__()
        self.layer4 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================================
# Line 52: ValueError for invalid layer name
# ============================================================================


def test_gradcam_invalid_layer_name():
    """Test that GradCAM raises ValueError for non-existent layer - Line 52."""
    model = SimpleModel()

    with pytest.raises(ValueError, match="Layer 'nonexistent_layer' not found"):
        GradCAM(model, target_layer_name="nonexistent_layer")


def test_gradcam_invalid_layer_shows_available():
    """Test that error message shows available layers."""
    model = SimpleModel()

    with pytest.raises(ValueError, match="Available"):
        GradCAM(model, target_layer_name="invalid_layer")


# ============================================================================
# Line 127: Squeeze when cam.shape[0] == 1 and squeeze=True
# ============================================================================


def test_gradcam_generate_squeeze_single_batch():
    """Test generate() with squeeze=True and single batch - Line 127."""
    model = SimpleModel()
    gc = GradCAM(model, target_layer_name="layer4")

    # Single image input
    x = torch.randn(1, 3, 32, 32)

    # Explicitly pass squeeze=True
    cam = gc.generate(x, class_idx=0, squeeze=True)

    # Should be 2D (H, W) not 3D (1, H, W)
    assert cam.ndim == 2
    assert cam.shape == (32, 32)


def test_gradcam_generate_no_squeeze_single_batch():
    """Test generate() with squeeze=False keeps batch dimension."""
    model = SimpleModel()
    gc = GradCAM(model, target_layer_name="layer4")

    x = torch.randn(1, 3, 32, 32)

    # squeeze=False should keep batch dimension
    cam = gc.generate(x, class_idx=0, squeeze=False)
    assert cam.ndim == 3
    assert cam.shape == (1, 32, 32)


def test_gradcam_generate_two_arg_call_triggers_squeeze():
    """Test two-arg call (model, x) triggers squeeze by default - Line 127."""
    model = SimpleModel()
    gc = GradCAM(model, target_layer_name="layer4")

    x = torch.randn(1, 3, 32, 32)

    # Two-arg call should set squeeze=True by default
    cam = gc.generate(model, x, class_idx=0)

    # Should be squeezed to 2D
    assert cam.ndim == 2


def test_gradcam_functional_api_2d_unsqueeze():
    """Test gradcam() functional API unsqueezes 2D output - Line 152 in gradcam()."""
    model = SimpleModel()

    x = torch.randn(1, 3, 32, 32)

    # Call functional API
    cam = gradcam(model, x, class_idx=0, layer="layer4")

    # Should always be 3D (N, H, W)
    assert cam.ndim == 3


# ============================================================================
# Additional coverage for completeness
# ============================================================================


def test_gradcam_various_input_scenarios():
    """Test various input scenarios."""
    model = SimpleModel()
    gc = GradCAM(model, target_layer_name="layer4")

    # Multi-batch input
    x_multi = torch.randn(4, 3, 32, 32)
    cam_multi = gc.generate(x_multi, class_idx=1)
    assert cam_multi.shape == (4, 32, 32)

    # No class_idx (sum over all classes)
    cam_sum = gc.generate(x_multi, class_idx=None)
    assert cam_sum.shape == (4, 32, 32)


def test_gradcam_remove_hooks():
    """Test hook removal methods."""
    model = SimpleModel()
    gc = GradCAM(model, target_layer_name="layer4")

    # Test remove_hooks()
    gc.remove_hooks()

    # Create new instance for remove() alias
    gc2 = GradCAM(model, target_layer_name="layer4")
    gc2.remove()


def test_get_gradcam_function():
    """Test get_gradcam() convenience function."""
    model = SimpleModel()
    gc = get_gradcam(model, layer="layer4")

    assert isinstance(gc, GradCAM)

    x = torch.randn(1, 3, 32, 32)
    cam = gc.generate(x)
    assert cam.ndim in (2, 3)


def test_gradcam_target_class_kwarg():
    """Test using target_class kwarg instead of class_idx."""
    model = SimpleModel()
    gc = GradCAM(model, target_layer_name="layer4")

    x = torch.randn(2, 3, 32, 32)

    # Use target_class kwarg
    cam = gc.generate(x, target_class=3)
    assert cam.shape == (2, 32, 32)


def test_gradcam_with_model_returning_tuple():
    """Test GradCAM with model that returns tuple."""

    class TupleOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = self.layer4(x)
            x = x.view(x.size(0), -1)
            logits = self.fc(x)
            return (logits, x)  # Return tuple

    model = TupleOutputModel()
    gc = GradCAM(model, target_layer_name="layer4")

    x = torch.randn(1, 3, 32, 32)
    cam = gc.generate(x, class_idx=0)

    assert cam.shape[1:] == (32, 32)


def test_gradcam_0d_logits():
    """Test GradCAM with 0-dimensional output (scalar)."""

    class ScalarOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            x = self.layer4(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x.squeeze()  # 0D scalar

    model = ScalarOutputModel()
    gc = GradCAM(model, target_layer_name="layer4")

    x = torch.randn(1, 3, 32, 32)
    cam = gc.generate(x)

    assert cam.ndim >= 2


def test_gradcam_1d_logits():
    """Test GradCAM with 1-dimensional output."""

    class OneDOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = self.layer4(x)
            x = x.view(-1)  # Flatten to 1D
            return self.fc(x).squeeze(0)

    model = OneDOutputModel()
    gc = GradCAM(model, target_layer_name="layer4")

    x = torch.randn(1, 3, 32, 32)
    cam = gc.generate(x)

    assert cam.shape[1:] == (32, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
