import torch

from src.attacks import pgd


def test_pgd_zero_eps_noop():
    x = torch.rand(2, 3, 8, 8)
    y = torch.tensor([0, 1])
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    out = pgd.pgd_attack(model, x, y, eps=0.0, alpha=0.1, steps=5, random_start=False)
    assert torch.allclose(out, x)  # no change


def test_pgd_respects_bounds():
    x = torch.zeros(1, 3, 8, 8)
    y = torch.tensor([1])
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    adv = pgd.pgd_attack(
        model, x, y, eps=0.2, alpha=0.1, steps=3, random_start=True, clamp=(0.0, 1.0)
    )
    assert adv.min() >= 0.0 and adv.max() <= 1.0
