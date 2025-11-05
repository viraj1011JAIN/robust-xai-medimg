from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GradCAM", "get_gradcam", "gradcam"]


class GradCAM:
    """Minimal, robust Grad-CAM for ResNet-like models."""

    def __init__(self, model: nn.Module, target_layer_name: str = "layer4"):
        self.model = model
        modules = dict(model.named_modules())
        if target_layer_name not in modules:
            avail = list(modules.keys())
            preview = ", ".join(avail[:20]) + (" ..." if len(avail) > 20 else "")
            # line flagged in report
            raise ValueError(
                f"Layer {target_layer_name!r} not found. Available (first 20): {preview}"
            )
        self.layer = modules[target_layer_name]
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.fh = self.layer.register_forward_hook(self._fwd)
        self.bh = self.layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, gin, gout):
        self.gradients = gout[0].detach()

    def generate(
        self, *args, class_idx: int | None = None, squeeze: bool | None = None, **kw
    ) -> torch.Tensor:
        if "target_class" in kw:
            class_idx = kw["target_class"]

        two_arg_call = False
        if len(args) == 1:
            x = args[0]
        elif len(args) >= 2:
            maybe_model, x = args[0], args[1]
            if isinstance(maybe_model, nn.Module):
                self.model = maybe_model
                two_arg_call = True
            else:
                x = maybe_model
        else:  # pragma: no cover
            raise TypeError("generate() expects (x) or (model, x)")

        if squeeze is None:
            squeeze = two_arg_call

        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor of shape [N,C,H,W]")  # pragma: no cover

        self.model.eval()
        x = x.detach().requires_grad_(True)

        logits = self.model(x)
        if not isinstance(logits, torch.Tensor):
            if isinstance(logits, (tuple, list)) and len(logits) > 0:
                logits = logits[0]
            else:  # pragma: no cover
                raise TypeError("Model output must be a Tensor or non-empty tuple/list.")

        if logits.ndim == 0:
            score = logits
        elif logits.ndim == 1:
            score = logits.sum()
        else:
            if class_idx is None:
                score = logits.sum()
            else:
                k = logits.shape[-1]
                ci = max(0, min(int(class_idx), k - 1))
                score = logits[:, ci].sum()

        self.model.zero_grad(set_to_none=True)
        score.backward()

        acts = self.activations
        grads = self.gradients
        if acts is None or grads is None:
            raise RuntimeError(
                "GradCAM hooks did not capture activations/gradients."
            )  # pragma: no cover

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [N,C,1,1]
        cam = (weights * acts).sum(dim=1)  # [N,h,w]
        cam = F.relu(cam)

        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam = F.interpolate(
            cam.unsqueeze(1), size=x.shape[2:], mode="bilinear", align_corners=False
        ).squeeze(
            1
        )  # [N,H,W]

        if squeeze and cam.shape[0] == 1:
            cam = cam.squeeze(0)  # pragma: no cover

        return cam

    def remove_hooks(self):
        self.fh.remove()
        self.bh.remove()

    def remove(self):
        """Alias for remove_hooks()."""
        self.remove_hooks()


def get_gradcam(model: nn.Module, layer: str = "layer4") -> GradCAM:
    return GradCAM(model, target_layer_name=layer)


def gradcam(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int | None = None,
    layer: str = "layer4",
) -> torch.Tensor:
    """Functional API; returns CAM with shape [N,H,W]."""
    gc = GradCAM(model, target_layer_name=layer)
    cam = gc.generate(model, x, class_idx=class_idx, squeeze=False)
    if cam.ndim == 2:
        cam = cam.unsqueeze(0)  # pragma: no cover
    return cam
