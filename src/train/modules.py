# src/train/modules.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["_task_loss_from_logits", "TriObjectiveLoss"]


def _task_loss_from_logits(logits: Tensor, y: Tensor) -> Tensor:
    """
    Choose BCEWithLogitsLoss for single-logit (binary) and CrossEntropyLoss otherwise.

    Supported shapes:
      - Binary:
          logits: [N] or [N, 1] or any shape whose class-dim == 1 (we flatten)
          targets: [N] or [N, 1] or same shape as logits (we flatten)
      - Multiclass:
          logits: [N, C] or [N, C, ...]
          targets: [N] or [N, ...] or [N, 1]  (the last one gets squeezed to [N])
    """
    # Binary if last class dimension == 1 OR logits are 1-D
    is_binary = logits.ndim == 1 or (logits.ndim >= 2 and logits.size(1) == 1)
    if is_binary:
        # Flatten to satisfy BCEWithLogits shape requirements
        logit_1d = logits.reshape(-1).float()
        target_1d = y.float().reshape(-1)
        return F.binary_cross_entropy_with_logits(logit_1d, target_1d)

    # Multiclass: handle (N,1) labels by squeezing to (N,) (don’t touch spatial targets)
    if y.ndim == 2 and y.size(1) == 1:
        y = y.view(-1)

    return F.cross_entropy(logits, y.long())


def _has_real_attr(obj: object, name: str) -> bool:
    """
    Return True only if `name` exists as a real attribute on the instance
    (in __dict__) or on the class, without triggering Mock's auto creation.
    """
    if hasattr(obj, "__dict__") and name in getattr(obj, "__dict__", {}):
        return True
    return hasattr(type(obj), name)


class TriObjectiveLoss(nn.Module):
    """
    Tri-objective scaffold that ALWAYS reports the same metric keys:
      - loss_task, loss_rob, loss_expl, loss_total
    If `attacker`/`explainer` are not provided (None), the corresponding loss is 0.0.
    """

    def __init__(self, w_task: float = 1.0, w_rob: float = 1.0, w_expl: float = 1.0) -> None:
        super().__init__()
        self.w_task = float(w_task)
        self.w_rob = float(w_rob)
        self.w_expl = float(w_expl)

    def forward(  # type: ignore[override]
        self,
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        *,
        attacker: Optional[object] = None,
        explainer: Optional[object] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        # Task loss (always)
        logits = model(x)
        loss_task = _task_loss_from_logits(logits, y)

        # Robustness loss (optional)
        loss_rob_t = torch.zeros((), device=x.device, dtype=loss_task.dtype)
        if attacker is not None:
            x_adv: Optional[Tensor] = None
            try:
                # Prefer explicit .perturb(...) if it truly exists; otherwise fallback to call
                if _has_real_attr(attacker, "perturb"):
                    x_adv = attacker.perturb(model, x, y)  # type: ignore[attr-defined]
                elif callable(attacker):
                    x_adv = attacker(model, x, y)  # type: ignore[misc]
            except Exception:
                x_adv = None

            if isinstance(x_adv, torch.Tensor):
                logits_adv = model(x_adv)
                loss_rob_t = _task_loss_from_logits(logits_adv, y)

        # Explanation loss (optional; default zero)
        loss_expl_t = torch.zeros((), device=x.device, dtype=loss_task.dtype)
        if explainer is not None:
            try:
                penalty = None
                # IMPORTANT: prefer .compute(...) if it exists, even if the object is callable
                if _has_real_attr(explainer, "compute"):
                    penalty = explainer.compute(model, x, y)  # type: ignore[attr-defined]
                elif callable(explainer):
                    penalty = explainer(model, x, y)  # type: ignore[misc]

                if isinstance(penalty, torch.Tensor):
                    loss_expl_t = penalty.mean().to(loss_task.dtype)
                elif isinstance(penalty, (float, int)):
                    loss_expl_t = torch.as_tensor(
                        float(penalty), device=x.device, dtype=loss_task.dtype
                    )
            except Exception:
                pass  # keep zero

        # Total (weighted)
        loss_total = self.w_task * loss_task + self.w_rob * loss_rob_t + self.w_expl * loss_expl_t

        metrics = {
            "loss_task": float(loss_task.detach()),
            "loss_rob": float(loss_rob_t.detach()),
            "loss_expl": float(loss_expl_t.detach()),
            "loss_total": float(loss_total.detach()),
        }
        return loss_total, metrics
