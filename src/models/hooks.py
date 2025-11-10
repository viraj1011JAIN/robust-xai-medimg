# src/models/hooks.py
from __future__ import annotations

from typing import Any, Optional

from torch import Tensor, nn

__all__ = ["FeatureExtractor"]


def _first_tensor(x: Any) -> Optional[Tensor]:
    """Return the first Tensor found in x (Tensor / tuple / list), else None."""
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for item in x:
            if isinstance(item, Tensor):
                return item
    return None


class FeatureExtractor:
    """
    Capture forward activations and backward gradients from a target module.

    - Context manager support: `with FeatureExtractor(mod) as fx: ...`
    - Stores forward output as `activations` (detached).
    - Stores gradient wrt module output as `gradients` (detached).
    - Idempotent `close()`; safe to call multiple times.
    - Handles tuple/list outputs gracefully.
    """

    def __init__(self, target_module: nn.Module) -> None:
        if not isinstance(target_module, nn.Module):
            raise TypeError("target_module must be a torch.nn.Module")
        self.target_module = target_module
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None

        self._closed = False
        self._fwd_handle = target_module.register_forward_hook(self._on_forward)

        # Prefer full-backward hook; fallback to legacy, using the same callback.
        if hasattr(target_module, "register_full_backward_hook"):
            self._bwd_handle = target_module.register_full_backward_hook(self._on_backward)  # type: ignore[attr-defined]
        else:  # pragma: no cover - very old PyTorch
            self._bwd_handle = target_module.register_backward_hook(self._on_backward)  # type: ignore[attr-defined]

    # ---------------- Hook callbacks ----------------

    def _on_forward(self, module: nn.Module, inp: Any, out: Any) -> None:
        t = _first_tensor(out)
        self.activations = t.detach() if isinstance(t, Tensor) else None

    def _on_backward(self, module: nn.Module, grad_in: Any, grad_out: Any) -> None:
        # grad_out mirrors forward outputs; we want gradient wrt output
        seq = grad_out if isinstance(grad_out, (tuple, list)) else (grad_out,)
        t = _first_tensor(seq)
        self.gradients = t.detach() if isinstance(t, Tensor) else None

    # ---------------- Public API --------------------

    def clear(self) -> None:
        """Clear stored tensors (does not remove hooks)."""
        self.activations = None
        self.gradients = None

    @property
    def is_attached(self) -> bool:
        """True if hooks are still registered."""
        return not self._closed

    def get_activations(self, clone: bool = False) -> Optional[Tensor]:
        if self.activations is None:
            return None
        return self.activations.clone() if clone else self.activations

    def get_gradients(self, clone: bool = False) -> Optional[Tensor]:
        if self.gradients is None:
            return None
        return self.gradients.clone() if clone else self.gradients

    def close(self) -> None:
        """Remove hooks (idempotent)."""
        if self._closed:
            return
        if getattr(self, "_fwd_handle", None) is not None:
            try:
                self._fwd_handle.remove()
            finally:
                self._fwd_handle = None
        if getattr(self, "_bwd_handle", None) is not None:
            try:
                self._bwd_handle.remove()
            finally:
                self._bwd_handle = None
        self._closed = True

    # --------------- Context manager ---------------

    def __enter__(self) -> "FeatureExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
