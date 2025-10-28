import torch
import torch.nn as nn

class FGSMAttack:
    """
    One-step FGSM attack: x_adv = x + epsilon * sign(grad_x L)
    Args:
        epsilon: max Linf perturbation (0..1 scale)
        loss_fn: default BCEWithLogitsLoss (binary)
    """
    def __init__(self, epsilon: float, loss_fn: nn.Module | None = None):
        self.epsilon = float(epsilon)
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()

    def __call__(self, model, x, y):
        was_training = model.training
        model.eval()

        x_adv = x.detach().clone().requires_grad_(True)
        logits = model(x_adv).squeeze()
        loss = self.loss_fn(logits, y.float())
        loss.backward()

        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + self.epsilon * grad_sign
            x_adv.clamp_(0.0, 1.0)

        if was_training:
            model.train()
        return x_adv.detach()
