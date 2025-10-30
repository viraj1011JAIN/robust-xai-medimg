import importlib

import pytest
import torch
from torchvision.models import resnet18

from src.xai import gradcam


def _make_gc(model):
    """
    Be tolerant to different Grad-CAM APIs:
      - class GradCAM(model, target_layer_name=...) or target_layer=...
      - factory get_gradcam(model, layer=...)
    """
    if hasattr(gradcam, "GradCAM"):
        try:
            return gradcam.GradCAM(model, target_layer_name="layer4")
        except TypeError:
            try:
                return gradcam.GradCAM(model, target_layer="layer4")
            except TypeError:
                return gradcam.GradCAM(model)
    if hasattr(gradcam, "get_gradcam"):
        return gradcam.get_gradcam(model, layer="layer4")  # type: ignore[attr-defined]
    raise AssertionError("No Grad-CAM entry point found in src.xai.gradcam")


def _run_generate(gc, x: torch.Tensor):
    """
    Accept several calling conventions without using hasattr(..., '__call__'):
      - gc.generate(x, class_idx=...) or gc.generate(x)
      - callable(gc)(x, class_idx=...) or callable(gc)(x)
      - module function gradcam.gradcam(gc, x, ...)
    """
    gen = getattr(gc, "generate", None)
    if callable(gen):
        try:
            return gen(x, class_idx=None)
        except TypeError:
            return gen(x)

    if callable(gc):
        try:
            return gc(x, class_idx=None)  # type: ignore[misc]
        except TypeError:
            return gc(x)  # type: ignore[misc]

    fn = getattr(gradcam, "gradcam", None)
    if callable(fn):
        try:
            return fn(gc, x, class_idx=None)
        except TypeError:
            return fn(gc, x)

    raise AssertionError("Dont know how to invoke Grad-CAM with this object")


@pytest.mark.parametrize("bs,h,w", [(1, 64, 64)])
def test_gradcam_output_shape(bs, h, w):
    model = resnet18(weights=None).eval()
    gc = _make_gc(model)
    x = torch.randn(bs, 3, h, w, requires_grad=True)
    heat = _run_generate(gc, x)

    assert torch.is_tensor(heat)
    assert list(heat.shape) == [bs, 1, h, w]


@pytest.mark.parametrize(
    "mod",
    [
        "src.train.baseline",
        "src.train.evaluate",
        "src.train.triobj_training",
        "src.xai.gradcam",
        "src.attacks.pgd",
    ],
)
def test_imports(mod):
    importlib.import_module(mod)
