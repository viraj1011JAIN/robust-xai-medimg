# tests/test_pgd_wrapper_class.py
import torch

from src.attacks.pgd import PGDAttack


def test_pgdattack_init_and_call_hits_wrapper_lines():
    """Test PGDAttack wrapper class initialization and call."""
    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2)
    ).eval()
    x = torch.rand(2, 3, 8, 8)
    y = torch.tensor([0, 1])

    # Initialize with all parameters to cover attribute assignment
    atk = PGDAttack(
        model=model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=3,
        random_start=True,
        clamp=(0.0, 1.0),
        loss_fn=None,  # Test default CrossEntropyLoss path
    )

    # Verify attributes are set
    assert atk.eps == 8 / 255, "eps should be set"
    assert atk.alpha == 2 / 255, "alpha should be set"
    assert atk.steps == 3, "steps should be set"
    assert atk.random_start is True, "random_start should be set"
    assert atk.clamp == (0.0, 1.0), "clamp should be set"

    # Test __call__ delegation
    out = atk(x, y)
    assert out.shape == x.shape, "Output shape should match input"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Should respect clamp bounds"


def test_pgdattack_with_custom_loss():
    """Test PGDAttack with custom loss function."""
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    x = torch.rand(1, 3, 8, 8)
    y = torch.tensor([1])

    custom_loss = torch.nn.CrossEntropyLoss()
    atk = PGDAttack(
        model=model,
        eps=0.05,
        alpha=0.01,
        steps=5,
        random_start=False,
        loss_fn=custom_loss,
    )

    out = atk(x, y)
    assert out.shape == x.shape, "Shape should be preserved"


def test_pgdattack_no_clamp():
    """Test PGDAttack without clamping."""
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2))
    x = torch.rand(1, 3, 8, 8)
    y = torch.tensor([0])

    atk = PGDAttack(
        model=model,
        eps=0.1,
        alpha=0.03,
        steps=3,
        random_start=True,
        clamp=None,  # No clamping
    )

    out = atk(x, y)
    assert out.shape == x.shape, "Shape should be preserved"
