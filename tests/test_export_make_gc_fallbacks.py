# tests/test_export_make_gc_fallbacks.py
"""Tests for _make_gc fallback paths."""

import torch

from src.xai import export as E


def test_make_gc_standard():
    """Test _make_gc with standard GradCAM."""
    model = E.build_model()
    gc = E._make_gc(model)

    assert gc is not None
    assert hasattr(gc, "model")


def test_make_gc_creates_usable_object():
    """Test that _make_gc creates a usable GradCAM object."""
    model = E.build_model()
    gc = E._make_gc(model)

    # Should be able to generate with it
    x = torch.rand(1, 3, 224, 224)
    result = E._run_generate(gc, x)

    assert isinstance(result, torch.Tensor)
    assert result.shape[-2:] == (224, 224)
