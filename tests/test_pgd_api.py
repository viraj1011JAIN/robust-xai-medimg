# tests/test_pgd_api.py
import torch

from src.attacks import pgd


def test_pgd_zero_eps_noop():
    """Test that PGD with eps=0 returns unchanged input."""
    x = torch.rand(2, 3, 8, 8)
    y = torch.tensor([0, 1])
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))

    out = pgd.pgd_attack(model, x, y, eps=0.0, alpha=0.1, steps=5, random_start=False)
    assert torch.allclose(out, x, atol=1e-6), "Output should match input when eps=0"


def test_pgd_respects_bounds():
    """Test that PGD respects clamp bounds."""
    x = torch.zeros(1, 3, 8, 8)
    y = torch.tensor([1])
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))

    adv = pgd.pgd_attack(
        model, x, y, eps=0.2, alpha=0.1, steps=3, random_start=True, clamp=(0.0, 1.0)
    )
    assert adv.min() >= 0.0, "Values should be >= 0.0"
    assert adv.max() <= 1.0, "Values should be <= 1.0"


def test_pgd_creates_perturbation():
    """Test that PGD creates non-trivial perturbations."""
    torch.manual_seed(42)
    x = torch.ones(1, 3, 8, 8) * 0.5
    y = torch.tensor([1])
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))

    adv = pgd.pgd_attack(model, x, y, eps=0.1, alpha=0.03, steps=10, random_start=True)

    # Should create some perturbation
    diff = (adv - x).abs().max()
    assert diff > 0, "Should create non-zero perturbation"
    assert diff <= 0.1 + 1e-5, "Perturbation should respect eps bound"
