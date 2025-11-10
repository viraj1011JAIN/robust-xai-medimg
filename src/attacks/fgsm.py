"""FGSM Attack implementation."""
from __future__ import annotations

import torch
import torch.nn as nn


class FGSMAttack:
    """
    Fast Gradient Sign Method.
    
    Args:
        epsilon: step size in input space (assumes inputs are scaled to [0, 1])
        loss_fn: optional criterion; defaults to BCEWithLogitsLoss for binary tasks
    """
    
    def __init__(self, epsilon: float = 0.03, loss_fn: nn.Module | None = None):
        self.epsilon = float(epsilon)
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()
    
    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples.
        
        Does not change the model's mode; clamps outputs to [0, 1].
        """
        # If epsilon is 0, return clamped input (no perturbation needed)
        if self.epsilon == 0.0:
            return torch.clamp(x.detach(), 0.0, 1.0)
        
        x_adv = x.detach().clone().requires_grad_(True)
        
        logits = model(x_adv)
        
        # Align shapes for common binary heads
        if logits.ndim > 1 and logits.size(-1) == 1:
            logits_for_loss = logits.squeeze(-1)
        else:
            logits_for_loss = logits
        
        y_float = y.to(dtype=logits_for_loss.dtype)
        
        model.zero_grad(set_to_none=True)
        loss = self.loss_fn(logits_for_loss, y_float)
        loss.backward()
        
        grad = x_adv.grad
        if grad is None:
            # Defensive: rare, but avoid crashing
            grad_sign = torch.zeros_like(x_adv)
        else:
            grad_sign = grad.sign()
        
        # Apply perturbation
        x_adv = x_adv.detach() + self.epsilon * grad_sign
        
        # Clamp to valid range
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        return x_adv.detach()