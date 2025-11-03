from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

__all__ = ["PGDAttack", "pgd_attack"]


class PGDAttack:
    """
    Basic untargeted PGD attack.
    - Guarantees ||x_adv - x||_∞ ≤ eps.
    - Does NOT clamp to a value range unless `clamp=(low, high)` is provided.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float,
        alpha: float,
        steps: int,
        *,
        random_start: bool = True,
        clamp: Optional[Tuple[float, float]] = None,  # default: no clamping
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.random_start = bool(random_start)
        self.clamp = clamp
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return pgd_attack(
            self.model,
            x,
            y,
            eps=self.eps,
            alpha=self.alpha,
            steps=self.steps,
            random_start=self.random_start,
            clamp=self.clamp,
            loss_fn=self.loss_fn,
        )


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
    clamp: Optional[Tuple[float, float]] = None,  # default: no clamping
    loss_fn: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Projected Gradient Descent (untargeted).
    - If eps == 0 or steps == 0 → returns x unchanged (no-op).
    - Preserves model train/eval state.
    - Always projects back to the L∞ ball around the ORIGINAL x.
    - Optional clamping only if `clamp=(low, high)` is provided.
    """
    loss_fn = loss_fn or nn.CrossEntropyLoss()

    # Accept one-hot labels as well
    if y.dim() > 1:
        y = y.argmax(dim=-1)

    was_training = model.training
    try:
        model.eval()

        x_orig = x.detach()
        x_adv = x_orig.clone()

        if eps == 0.0 or steps == 0:
            return x_adv  # no-op

        if random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)

        for _ in range(steps):
            x_adv.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(x_adv)
            loss = loss_fn(logits, y)
            loss.backward()
            grad = x_adv.grad.detach()

            # gradient sign step
            x_adv = x_adv.detach() + alpha * grad.sign()

            # project back to eps-ball around original x
            eta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
            x_adv = (x_orig + eta).detach()

            # optional clamping
            if clamp is not None:
                lo, hi = clamp
                x_adv = x_adv.clamp(lo, hi)

        return x_adv.detach()
    finally:
        model.train(was_training)
