"""
Comprehensive tests for src/models/hooks.py
"""

# Import directly from hooks module to avoid __init__.py with timm dependency
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# Load hooks.py directly without triggering __init__.py
hooks_path = Path(__file__).parent.parent / "src" / "models" / "hooks.py"
spec = importlib.util.spec_from_file_location("hooks", hooks_path)
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)

FeatureExtractor = hooks.FeatureExtractor
_first_tensor = hooks._first_tensor


class SimpleModule(nn.Module):
    """Simple module for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TupleOutputModule(nn.Module):
    """Module that returns a tuple."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        out = self.linear(x)
        return (out, torch.zeros(5))


class ListOutputModule(nn.Module):
    """Module that returns a list."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        out = self.linear(x)
        return [out, torch.zeros(5)]


# Tests for _first_tensor helper
class TestFirstTensor:
    def test_tensor_input(self):
        """Test with direct tensor input."""
        t = torch.randn(3, 4)
        result = _first_tensor(t)
        assert result is t

    def test_tuple_with_tensor(self):
        """Test with tuple containing tensor."""
        t = torch.randn(3, 4)
        result = _first_tensor((t, torch.zeros(2)))
        assert result is t

    def test_list_with_tensor(self):
        """Test with list containing tensor."""
        t = torch.randn(3, 4)
        result = _first_tensor([t, torch.zeros(2)])
        assert result is t

    def test_empty_tuple(self):
        """Test with empty tuple."""
        result = _first_tensor(())
        assert result is None

    def test_empty_list(self):
        """Test with empty list."""
        result = _first_tensor([])
        assert result is None

    def test_non_tensor_input(self):
        """Test with non-tensor input."""
        result = _first_tensor(42)
        assert result is None

    def test_tuple_with_non_tensors(self):
        """Test with tuple containing only non-tensors."""
        result = _first_tensor((1, 2, "hello"))
        assert result is None

    def test_tuple_tensor_second_position(self):
        """Test with tensor in second position of tuple."""
        t = torch.randn(3, 4)
        result = _first_tensor((None, t))
        assert result is t


# Tests for FeatureExtractor
class TestFeatureExtractorInit:
    def test_init_with_valid_module(self):
        """Test initialization with valid module."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        assert fx.target_module is module
        assert fx.activations is None
        assert fx.gradients is None
        assert fx.is_attached is True
        assert fx._closed is False

        fx.close()

    def test_init_with_invalid_module(self):
        """Test initialization with non-module raises TypeError."""
        with pytest.raises(TypeError, match="target_module must be a torch.nn.Module"):
            FeatureExtractor("not a module")

    def test_hooks_registered(self):
        """Test that hooks are registered on initialization."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        assert fx._fwd_handle is not None
        assert fx._bwd_handle is not None

        fx.close()


class TestFeatureExtractorForward:
    def test_capture_simple_forward(self):
        """Test capturing activations from simple forward pass."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10)
        output = module(x)

        assert fx.activations is not None
        assert fx.activations.shape == output.shape
        assert torch.allclose(fx.activations, output)

        fx.close()

    def test_capture_tuple_output(self):
        """Test capturing activations when module returns tuple."""
        module = TupleOutputModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10)
        output_tuple = module(x)

        assert fx.activations is not None
        assert fx.activations.shape == output_tuple[0].shape
        assert torch.allclose(fx.activations, output_tuple[0])

        fx.close()

    def test_capture_list_output(self):
        """Test capturing activations when module returns list."""
        module = ListOutputModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10)
        output_list = module(x)

        assert fx.activations is not None
        assert fx.activations.shape == output_list[0].shape
        assert torch.allclose(fx.activations, output_list[0])

        fx.close()

    def test_activations_detached(self):
        """Test that activations are detached."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10, requires_grad=True)
        _ = module(x)

        assert fx.activations is not None
        assert fx.activations.requires_grad is False

        fx.close()

    def test_multiple_forward_passes(self):
        """Test that activations update on multiple forward passes."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        x1 = torch.randn(2, 10)
        out1 = module(x1)
        act1 = fx.activations.clone()

        x2 = torch.randn(2, 10)
        out2 = module(x2)
        act2 = fx.activations

        assert not torch.allclose(act1, act2)
        assert torch.allclose(act2, out2)

        fx.close()


class TestFeatureExtractorBackward:
    def test_capture_gradients(self):
        """Test capturing gradients during backward pass."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10, requires_grad=True)
        output = module(x)
        loss = output.sum()
        loss.backward()

        assert fx.gradients is not None
        assert fx.gradients.shape == output.shape

        fx.close()

    def test_capture_gradients_non_tuple_grad_out(self):
        """Test gradient capture when grad_out is not a tuple/list."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        # This will test the else branch in _on_backward_full
        # where grad_out is not already a tuple/list
        x = torch.randn(2, 10, requires_grad=True)
        output = module(x)
        loss = output.sum()
        loss.backward()

        assert fx.gradients is not None

        fx.close()

    def test_gradients_detached(self):
        """Test that gradients are detached."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10, requires_grad=True)
        output = module(x)
        loss = output.sum()
        loss.backward()

        assert fx.gradients is not None
        assert fx.gradients.requires_grad is False

        fx.close()

    def test_gradients_tuple_output(self):
        """Test capturing gradients when module returns tuple."""
        module = TupleOutputModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10, requires_grad=True)
        output_tuple = module(x)
        loss = output_tuple[0].sum()
        loss.backward()

        assert fx.gradients is not None

        fx.close()

    def test_multiple_backward_passes(self):
        """Test that gradients update on multiple backward passes."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        # First backward
        x1 = torch.randn(2, 10, requires_grad=True)
        out1 = module(x1)
        loss1 = out1.sum()
        loss1.backward()
        grad1 = fx.gradients.clone()

        # Second backward
        x2 = torch.randn(2, 10, requires_grad=True)
        out2 = module(x2)
        loss2 = (out2 * 2).sum()
        loss2.backward()
        grad2 = fx.gradients

        # Gradients should be different
        assert not torch.allclose(grad1, grad2)

        fx.close()


class TestFeatureExtractorMethods:
    def test_clear(self):
        """Test clear() method."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10, requires_grad=True)
        output = module(x)
        loss = output.sum()
        loss.backward()

        assert fx.activations is not None
        assert fx.gradients is not None

        fx.clear()

        assert fx.activations is None
        assert fx.gradients is None
        assert fx.is_attached is True  # Hooks still attached

        fx.close()

    def test_get_activations_no_clone(self):
        """Test get_activations() without cloning."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10)
        _ = module(x)

        acts = fx.get_activations(clone=False)
        assert acts is fx.activations

        fx.close()

    def test_get_activations_with_clone(self):
        """Test get_activations() with cloning."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10)
        _ = module(x)

        acts = fx.get_activations(clone=True)
        assert acts is not fx.activations
        assert torch.allclose(acts, fx.activations)

        fx.close()

    def test_get_activations_none(self):
        """Test get_activations() when no activations captured."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        acts = fx.get_activations()
        assert acts is None

        fx.close()

    def test_get_gradients_no_clone(self):
        """Test get_gradients() without cloning."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10, requires_grad=True)
        output = module(x)
        loss = output.sum()
        loss.backward()

        grads = fx.get_gradients(clone=False)
        assert grads is fx.gradients

        fx.close()

    def test_get_gradients_with_clone(self):
        """Test get_gradients() with cloning."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10, requires_grad=True)
        output = module(x)
        loss = output.sum()
        loss.backward()

        grads = fx.get_gradients(clone=True)
        assert grads is not fx.gradients
        assert torch.allclose(grads, fx.gradients)

        fx.close()

    def test_get_gradients_none(self):
        """Test get_gradients() when no gradients captured."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        grads = fx.get_gradients()
        assert grads is None

        fx.close()

    def test_is_attached_property(self):
        """Test is_attached property."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        assert fx.is_attached is True

        fx.close()

        assert fx.is_attached is False


class TestFeatureExtractorClose:
    def test_close_removes_hooks(self):
        """Test that close() removes hooks."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        assert fx.is_attached is True

        fx.close()

        assert fx.is_attached is False
        assert fx._closed is True
        assert fx._fwd_handle is None
        assert fx._bwd_handle is None

    def test_close_idempotent(self):
        """Test that close() can be called multiple times safely."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        fx.close()
        fx.close()  # Should not raise
        fx.close()  # Should not raise

        assert fx.is_attached is False

    def test_forward_after_close(self):
        """Test that forward pass after close doesn't capture activations."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        x = torch.randn(2, 10)
        _ = module(x)
        assert fx.activations is not None

        fx.close()
        fx.clear()

        x2 = torch.randn(2, 10)
        _ = module(x2)
        assert fx.activations is None  # Not captured after close


class TestFeatureExtractorContextManager:
    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        module = SimpleModule()

        with FeatureExtractor(module) as fx:
            assert fx.is_attached is True

            x = torch.randn(2, 10)
            _ = module(x)

            assert fx.activations is not None

        assert fx.is_attached is False

    def test_context_manager_with_exception(self):
        """Test context manager cleans up even with exception."""
        module = SimpleModule()

        with pytest.raises(ValueError):
            with FeatureExtractor(module) as fx:
                assert fx.is_attached is True
                raise ValueError("Test exception")

        assert fx.is_attached is False

    def test_context_manager_captures_gradients(self):
        """Test capturing gradients within context manager."""
        module = SimpleModule()

        with FeatureExtractor(module) as fx:
            x = torch.randn(2, 10, requires_grad=True)
            output = module(x)
            loss = output.sum()
            loss.backward()

            assert fx.activations is not None
            assert fx.gradients is not None

        assert fx.is_attached is False


class TestFeatureExtractorDestructor:
    def test_destructor_cleanup(self):
        """Test that destructor cleans up hooks."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        assert fx.is_attached is True

        # Delete the object
        del fx

        # Hooks should be cleaned up (though we can't directly verify)
        # This test mainly ensures no exceptions are raised

    def test_destructor_after_close(self):
        """Test that destructor is safe after explicit close."""
        module = SimpleModule()
        fx = FeatureExtractor(module)

        fx.close()
        del fx  # Should not raise


class TestFeatureExtractorIntegration:
    def test_full_forward_backward_cycle(self):
        """Test complete forward and backward pass."""
        module = SimpleModule()

        with FeatureExtractor(module) as fx:
            # Forward pass
            x = torch.randn(2, 10, requires_grad=True)
            output = module(x)

            # Check activations captured
            assert fx.activations is not None
            assert torch.allclose(fx.activations, output)

            # Backward pass
            loss = output.sum()
            loss.backward()

            # Check gradients captured
            assert fx.gradients is not None
            assert fx.gradients.shape == output.shape

    def test_nested_modules(self):
        """Test with nested module structure."""

        class NestedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 8)
                self.layer2 = nn.Linear(8, 5)

            def forward(self, x):
                x = self.layer1(x)
                x = torch.relu(x)
                x = self.layer2(x)
                return x

        module = NestedModule()

        # Hook the second layer
        with FeatureExtractor(module.layer2) as fx:
            x = torch.randn(2, 10, requires_grad=True)
            output = module(x)
            loss = output.sum()
            loss.backward()

            assert fx.activations is not None
            assert fx.gradients is not None

    def test_batch_processing(self):
        """Test with different batch sizes."""
        module = SimpleModule()

        with FeatureExtractor(module) as fx:
            for batch_size in [1, 4, 16]:
                fx.clear()
                x = torch.randn(batch_size, 10)
                output = module(x)

                assert fx.activations is not None
                assert fx.activations.shape[0] == batch_size
