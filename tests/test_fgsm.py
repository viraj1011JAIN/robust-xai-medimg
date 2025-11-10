"""
Complete tests for src/attacks/fgsm.py to achieve ~100% coverage.
Covers: train/eval, eps=0, large eps, clipping, shapes (N,1) and (N,),
custom loss_fn path, determinism, gradient bound.
"""
import torch
import torch.nn as nn
import pytest

from src.attacks.fgsm import FGSMAttack


class SimpleNetCol(nn.Module):
    """Linear head returning shape (N, 1)."""
    def __init__(self, in_features=4):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x)


class SimpleNetFlat(nn.Module):
    """Linear head returning shape (N,) to cover logits.squeeze path."""
    def __init__(self, in_features=4):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)  # shape (N,)


class DummyLoss(nn.Module):
    """Custom loss to ensure the custom loss_fn branch is executed."""
    def __init__(self):
        super().__init__()
        self.called = False

    def forward(self, logits, targets):
        self.called = True
        # simple MSE so no logits constraints
        return ((logits - targets) ** 2).mean()


class TestFGSMAttack:
    def test_basic_perturbation_train_mode(self):
        model = SimpleNetCol().train()
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.tensor([0.0, 1.0, 0.0])

        adv = FGSMAttack(epsilon=0.1)(model, x, y)
        assert isinstance(adv, torch.Tensor)
        assert adv.shape == x.shape
        assert not adv.requires_grad
        assert torch.any((adv - x).abs() > 0)

    def test_eval_mode(self):
        model = SimpleNetCol().eval()
        x = torch.randn(2, 4, requires_grad=True)
        y = torch.tensor([1.0, 0.0])

        adv = FGSMAttack(epsilon=0.05)(model, x, y)
        assert adv.shape == x.shape
        assert torch.any((adv - x).abs() > 0)

    def test_zero_epsilon(self):
        model = SimpleNetCol()
        # Use in-range inputs so clamp doesn't change x when eps=0
        x = torch.rand(3, 4, requires_grad=True) * 0.8 + 0.1  # range [0.1, 0.9]
        y = torch.tensor([0.0, 1.0, 0.0])

        adv = FGSMAttack(epsilon=0.0)(model, x, y)
        assert torch.allclose(adv, x, atol=1e-6)

    def test_large_epsilon_creates_change(self):
        model = SimpleNetCol()
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.tensor([0.0, 1.0, 0.0])

        adv = FGSMAttack(epsilon=1.0)(model, x, y)
        assert (adv - x).abs().max() > 0

    def test_clipping_to_unit_interval(self):
        model = SimpleNetCol()
        x = torch.tensor([[0.01, 0.01, 0.99, 0.99],
                          [0.02, 0.98, 0.03, 0.97]], requires_grad=True)
        y = torch.tensor([0.0, 1.0])
        adv = FGSMAttack(epsilon=0.5)(model, x, y)
        assert torch.all(adv >= 0.0)
        assert torch.all(adv <= 1.0)

    def test_single_sample_and_batch(self):
        model = SimpleNetCol()
        x1 = torch.randn(1, 4, requires_grad=True)
        y1 = torch.tensor([0.0])
        adv1 = FGSMAttack(0.1)(model, x1, y1)
        assert adv1.shape == (1, 4)

        xN = torch.randn(16, 4, requires_grad=True)
        yN = torch.rand(16)
        advN = FGSMAttack(0.1)(model, xN, yN)
        assert advN.shape == (16, 4)

    def test_deterministic_for_same_inputs(self):
        torch.manual_seed(42)
        model = SimpleNetCol()
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.tensor([0.0, 1.0, 0.0])

        atk = FGSMAttack(0.1)
        a1 = atk(model, x.clone(), y)
        a2 = atk(model, x.clone(), y)
        assert torch.allclose(a1, a2)

    def test_gradient_bound_by_epsilon(self):
        model = SimpleNetCol().train()
        # Use in-range inputs so clamp doesn't add extra displacement
        x = torch.rand(2, 4, requires_grad=True) * 0.8 + 0.1  # range [0.1, 0.9]
        y = torch.tensor([1.0, 0.0])

        eps = 0.1
        adv = FGSMAttack(epsilon=eps)(model, x, y)
        delta = (adv - x).abs().max()
        assert delta <= eps + 1e-5

    def test_non_column_logits_shape_path(self):
        """Covers branch where logits are already shape (N,) not (N,1)."""
        model = SimpleNetFlat()
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        adv = FGSMAttack(0.1)(model, x, y)
        assert adv.shape == x.shape

    def test_custom_loss_fn_branch(self):
        """Ensure provided loss_fn is actually used."""
        model = SimpleNetCol()
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.tensor([0.1, 0.9, 0.2])  # arbitrary floats OK for DummyLoss
        custom = DummyLoss()
        _ = FGSMAttack(0.05, loss_fn=custom)(model, x, y)
        assert custom.called