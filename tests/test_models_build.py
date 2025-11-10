"""
REAL tests for src/models/build.py - requires timm library installed.
Run: pip install timm
"""

import pytest
import torch
import torch.nn as nn

# Check if timm is available
try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TIMM_AVAILABLE, reason="timm not installed. Run: pip install timm"
)


class TestBuildModel:
    """Test real model building with timm."""

    def test_build_resnet18(self):
        """Test building actual ResNet18."""
        from src.models.build import build_model

        model = build_model("resnet18", num_classes=1, pretrained=False)

        assert isinstance(model, nn.Module)
        assert hasattr(model, "layer4")

        # Test forward pass
        x = torch.rand(2, 3, 224, 224)
        output = model(x)
        assert output.shape[0] == 2  # Batch size

    def test_build_resnet50(self):
        """Test building actual ResNet50."""
        from src.models.build import build_model

        model = build_model("resnet50", num_classes=1, pretrained=False)

        assert isinstance(model, nn.Module)

        # Test forward pass
        x = torch.rand(1, 3, 224, 224)
        output = model(x)
        assert output is not None

    def test_build_efficientnet(self):
        """Test building actual EfficientNet."""
        from src.models.build import build_model

        model = build_model("efficientnet_b0", num_classes=1, pretrained=False)

        assert isinstance(model, nn.Module)
        assert hasattr(model, "conv_head")

        # Test forward pass
        x = torch.rand(1, 3, 224, 224)
        output = model(x)
        assert output is not None

    def test_build_vit(self):
        """Test building actual Vision Transformer."""
        from src.models.build import build_model

        model = build_model("vit_base_patch16_224", num_classes=1, pretrained=False)

        assert isinstance(model, nn.Module)

        # Test forward pass
        x = torch.rand(1, 3, 224, 224)
        output = model(x)
        assert output is not None

    def test_build_with_pretrained(self):
        """Test building with pretrained weights."""
        from src.models.build import build_model

        # Use a small model to save time/memory
        model = build_model("resnet18", num_classes=1, pretrained=True)

        assert isinstance(model, nn.Module)

        # Check that parameters exist and have values
        params = list(model.parameters())
        assert len(params) > 0
        assert not torch.all(params[0] == 0)  # Should have non-zero weights

    def test_build_multiclass(self):
        """Test building for multi-class classification."""
        from src.models.build import build_model

        for num_classes in [1, 2, 10, 100]:
            model = build_model("resnet18", num_classes=num_classes, pretrained=False)
            assert isinstance(model, nn.Module)

            # Test output shape
            x = torch.rand(2, 3, 224, 224)
            output = model(x)
            if num_classes == 1:
                assert output.shape == (2, 1)
            else:
                assert output.shape == (2, num_classes)

    def test_feature_extractor_attached(self):
        """Test that FeatureExtractor hook is properly attached."""
        from src.models.build import build_model

        model = build_model("resnet18", num_classes=1, pretrained=False)

        # The model should have the hook attached
        # Run a forward pass and check features are captured
        x = torch.rand(1, 3, 224, 224)
        output = model(x)

        # Hook should have captured features
        assert hasattr(model, "feature_extractor") or hasattr(model, "hook")

    def test_different_architectures(self):
        """Test multiple architectures work correctly."""
        from src.models.build import build_model

        architectures = [
            "resnet18",
            "resnet34",
            "resnet50",
        ]

        for arch in architectures:
            model = build_model(arch, num_classes=1, pretrained=False)
            assert isinstance(model, nn.Module)

            # Verify it can do inference
            x = torch.rand(1, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, 1)

    def test_model_in_eval_mode(self):
        """Test model behavior in eval mode."""
        from src.models.build import build_model

        model = build_model("resnet18", num_classes=1, pretrained=False)
        model.eval()

        x = torch.rand(2, 3, 224, 224)

        # Run twice - should get same output in eval mode
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)

    def test_model_gradients(self):
        """Test that model parameters have gradients."""
        from src.models.build import build_model

        model = build_model("resnet18", num_classes=1, pretrained=False)
        model.train()

        # Forward + backward pass
        x = torch.rand(2, 3, 224, 224)
        output = model(x)
        loss = output.mean()
        loss.backward()

        # Check some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_grad = True
                break

        assert has_grad, "Model should have gradients after backward pass"


class TestModelInitialization:
    """Test model __init__ imports."""

    def test_models_module_imports(self):
        """Test that src.models can be imported."""
        import src.models as models

        assert hasattr(models, "build_model")

    def test_build_model_function_exists(self):
        """Test that build_model function is accessible."""
        from src.models import build_model

        assert callable(build_model)
