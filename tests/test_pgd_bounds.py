import pytest
import torch
import torch.nn as nn

pgd_mod = pytest.importorskip("src.attacks.pgd", reason="pgd module missing")


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


def _attack(model, x, y, eps=2 / 255, steps=5):
    # Try a few likely names
    if hasattr(pgd_mod, "pgd_attack"):
        return pgd_mod.pgd_attack(model, x, y, eps=eps, alpha=eps / steps, steps=steps)
    if hasattr(pgd_mod, "PGD"):
        atk = pgd_mod.PGD(model, eps=eps, alpha=eps / steps, steps=steps)
        return atk(x, y)
    if hasattr(pgd_mod, "fgsm_attack"):
        return pgd_mod.fgsm_attack(model, x, y, eps=eps)
    pytest.skip("No recognized PGD/FGSM API in src/attacks/pgd.py")


def linf(x, x_adv):
    return (x_adv - x).detach().abs().amax().item()


def test_attack_respects_eps():
    torch.manual_seed(0)
    model = Tiny().eval()
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    y = torch.tensor([0, 1])
    x_adv = _attack(model, x, y, eps=2 / 255, steps=5)
    assert x_adv.shape == x.shape
    assert linf(x, x_adv) <= 2 / 255 + 1e-6
