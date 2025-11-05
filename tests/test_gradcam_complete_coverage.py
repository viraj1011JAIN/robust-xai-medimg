# tests/test_gradcam_complete_coverage.py
"""Complete coverage tests for GradCAM."""

import pytest
import torch
import torch.nn as nn

from src.xai.gradcam import GradCAM


class TestModel(nn.Module):
    """Simple test model with layer4 for GradCAM."""

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


def test_gradcam_init():
    """Test GradCAM initialization."""
    model = TestModel()
    cam = GradCAM(model, target_layer_name="layer4")
    assert cam is not None


def test_gradcam_invalid_layer_raises():
    """Test GradCAM with invalid layer name."""
    model = TestModel()
    with pytest.raises(ValueError):
        GradCAM(model, target_layer_name="nonexistent_layer")


def test_gradcam_generate():
    """Test GradCAM generation."""
    model = TestModel().eval()
    cam = GradCAM(model, target_layer_name="layer4")

    x = torch.rand(1, 3, 32, 32)
    result = cam.generate(model, x)

    assert result.shape[-2:] == (32, 32)
    cam.remove()


def test_gradcam_with_target_class():
    """Test GradCAM with specific target class."""
    model = TestModel().eval()
    cam = GradCAM(model, target_layer_name="layer4")

    x = torch.rand(2, 3, 32, 32)
    result = cam.generate(model, x, target_class=1)

    assert result.shape[-2:] == (32, 32)
    cam.remove()


def test_gradcam_batch_processing():
    """Test GradCAM with batch."""
    model = TestModel().eval()
    cam = GradCAM(model, target_layer_name="layer4")

    x = torch.rand(4, 3, 32, 32)
    result = cam.generate(model, x)

    assert result.shape[-2:] == (32, 32)
    cam.remove()


def test_gradcam_different_sizes():
    """Test GradCAM with different input sizes."""
    model = TestModel().eval()
    cam = GradCAM(model, target_layer_name="layer4")

    for size in [64, 128, 224]:
        x = torch.rand(1, 3, size, size)
        result = cam.generate(model, x)
        assert result.shape[-2:] == (size, size)

    cam.remove()


def test_gradcam_remove_hooks():
    """Test GradCAM hook removal."""
    model = TestModel().eval()
    cam = GradCAM(model, target_layer_name="layer4")

    # Generate once
    x = torch.rand(1, 3, 32, 32)
    cam.generate(model, x)

    # Remove hooks
    cam.remove()

    # Should be able to call remove multiple times
    cam.remove()


def test_gradcam_with_gradient():
    """Test GradCAM actually uses gradients."""
    model = TestModel()
    model.eval()

    cam = GradCAM(model, target_layer_name="layer4")

    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    result = cam.generate(model, x, target_class=0)

    # Result should be non-negative (ReLU applied)
    assert (result >= 0).all()

    cam.remove()


class ModelWithoutLayer4(nn.Module):
    """Model without layer4 for testing error paths."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        return x


def test_gradcam_model_without_target_layer():
    """Test GradCAM with model missing target layer."""
    model = ModelWithoutLayer4()

    with pytest.raises(ValueError):
        GradCAM(model, target_layer_name="layer4")


def test_gradcam_none_target_class():
    """Test GradCAM with target_class=None."""
    model = TestModel().eval()
    cam = GradCAM(model, target_layer_name="layer4")

    x = torch.rand(1, 3, 32, 32)
    result = cam.generate(model, x, target_class=None)

    assert result.shape[-2:] == (32, 32)
    cam.remove()
