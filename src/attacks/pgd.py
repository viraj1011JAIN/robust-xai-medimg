import torch
import torch.nn as nn


class PGDAttack:
    """
    Projected Gradient Descent (Linf).
    Args:
        epsilon: Linf budget (0..1)
        alpha: step size (0..1)
        num_steps: iterations
        random_start: if True, start within epsilon-ball
        loss_fn: default BCEWithLogitsLoss (binary)
    """

    def __init__(
        self,
        epsilon: float,
        alpha: float,
        num_steps: int,
        random_start: bool = True,
        loss_fn: nn.Module | None = None,
    ):
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.num_steps = int(num_steps)
        self.random_start = bool(random_start)
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()

    def __call__(self, model, x, y):
        was_training = model.training
        model.eval()

        x0 = x.detach()
        if self.random_start and self.epsilon > 0:
            delta = torch.empty_like(x0).uniform_(-self.epsilon, self.epsilon)
            x_adv = (x0 + delta).clamp(0.0, 1.0)
        else:
            x_adv = x0.clone()

        for _ in range(self.num_steps):
            x_adv = x_adv.detach().clone().requires_grad_(True)
            logits = model(x_adv).squeeze()
            loss = self.loss_fn(logits, y.float())
            loss.backward()

            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + self.alpha * grad_sign
                # project to Linf-ball around x0
                x_adv = torch.max(
                    torch.min(x_adv, x0 + self.epsilon), x0 - self.epsilon
                )
                x_adv.clamp_(0.0, 1.0)
            x_adv.grad = None

        if was_training:
            model.train()
        return x_adv.detach()
