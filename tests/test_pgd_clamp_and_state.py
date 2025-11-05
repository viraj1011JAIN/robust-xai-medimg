# tests/test_pgd_clamp_and_state.py
import torch

from src.attacks import pgd


class Tiny(torch.nn.Module):
    """Tiny model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 8 * 8, 2)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


def test_pgd_clamp_and_state_restore():
    """Test PGD clamping and model state restoration."""
    torch.manual_seed(0)
    model = Tiny()
    model.train(True)  # Set to training mode

    x = torch.rand(2, 3, 8, 8)
    y_onehot = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

    # Use clamp to cover optional clamping branch
    x_adv = pgd.pgd_attack(
        model,
        x,
        y_onehot,  # One-hot labels should be converted via argmax
        eps=2 / 255,
        alpha=1 / 255,
        steps=3,
        random_start=True,
        clamp=(0.0, 1.0),
    )

    # Verify shape is preserved
    assert x_adv.shape == x.shape, "Output shape should match input"

    # Verify model state is restored to training mode
    assert model.training is True, "Model should be in training mode after attack"

    # Verify clamping was applied
    assert x_adv.min() >= 0.0, "Values should be >= 0.0"
    assert x_adv.max() <= 1.0, "Values should be <= 1.0"


def test_pgd_with_label_indices():
    """Test PGD with label indices instead of one-hot."""
    torch.manual_seed(1)
    model = Tiny()
    x = torch.rand(2, 3, 8, 8)
    y = torch.tensor([0, 1])  # Label indices

    x_adv = pgd.pgd_attack(model, x, y, eps=4 / 255, alpha=1 / 255, steps=5, random_start=False)

    assert x_adv.shape == x.shape, "Shape should be preserved"
    assert not torch.equal(x_adv, x), "Should create perturbation"
