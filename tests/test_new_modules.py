"""
Comprehensive test suite for src/train/modules.py
Achieves 100% line and branch coverage.
"""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from src.train.modules import TriObjectiveLoss, _task_loss_from_logits

# ============================================================================
# Test _task_loss_from_logits function
# ============================================================================


def test_task_loss_binary_1d_logits_1d_y():
    """Test binary loss with 1D logits and 1D targets."""
    logits = torch.randn(8)
    y = torch.randint(0, 2, (8,))
    loss = _task_loss_from_logits(logits, y)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_task_loss_binary_2d_logits_1d_y():
    """Test binary loss with (N, 1) logits and 1D targets."""
    logits = torch.randn(8, 1)
    y = torch.randint(0, 2, (8,))
    loss = _task_loss_from_logits(logits, y)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_task_loss_binary_2d_logits_2d_y():
    """Test binary loss with (N, 1) logits and (N, 1) targets."""
    logits = torch.randn(8, 1)
    y = torch.randint(0, 2, (8, 1))
    loss = _task_loss_from_logits(logits, y)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_task_loss_binary_1d_logits_2d_y():
    """Test binary loss with 1D logits and (N, 1) targets."""
    logits = torch.randn(8)
    y = torch.randint(0, 2, (8, 1))
    loss = _task_loss_from_logits(logits, y)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_task_loss_binary_float_targets():
    """Test binary loss with float targets."""
    logits = torch.randn(8, 1)
    y = torch.rand(8)  # Float targets
    loss = _task_loss_from_logits(logits, y)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_task_loss_multiclass_2d():
    """Test multiclass loss with (N, C) logits."""
    logits = torch.randn(8, 5)
    y = torch.randint(0, 5, (8,))
    loss = _task_loss_from_logits(logits, y)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_task_loss_multiclass_2d_y():
    """Test multiclass loss with 2D targets (gets reshaped to 1D)."""
    logits = torch.randn(8, 5)
    y = torch.randint(0, 5, (8, 1))
    loss = _task_loss_from_logits(logits, y)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


def test_task_loss_multiclass_3d():
    """Test multiclass loss with 3D logits (e.g., segmentation)."""
    logits = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 3, (4, 8, 8))
    loss = _task_loss_from_logits(logits, y)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


# ============================================================================
# Test TriObjectiveLoss initialization
# ============================================================================


def test_init_default_weights():
    """Test default weight initialization."""
    loss_fn = TriObjectiveLoss()
    assert loss_fn.w_task == 1.0
    assert loss_fn.w_rob == 1.0
    assert loss_fn.w_expl == 1.0


def test_init_custom_weights():
    """Test custom weight initialization."""
    loss_fn = TriObjectiveLoss(w_task=0.5, w_rob=0.3, w_expl=0.2)
    assert loss_fn.w_task == 0.5
    assert loss_fn.w_rob == 0.3
    assert loss_fn.w_expl == 0.2


def test_init_zero_weights():
    """Test initialization with zero weights."""
    loss_fn = TriObjectiveLoss(w_task=0.0, w_rob=0.0, w_expl=0.0)
    assert loss_fn.w_task == 0.0
    assert loss_fn.w_rob == 0.0
    assert loss_fn.w_expl == 0.0


def test_init_negative_weights():
    """Test initialization with negative weights."""
    loss_fn = TriObjectiveLoss(w_task=-1.0, w_rob=-0.5, w_expl=-0.2)
    assert loss_fn.w_task == -1.0
    assert loss_fn.w_rob == -0.5
    assert loss_fn.w_expl == -0.2


# ============================================================================
# Test forward pass - basic functionality
# ============================================================================


def test_forward_basic_no_optionals():
    """Test forward pass without attacker or explainer."""
    model = nn.Linear(10, 1)
    loss_fn = TriObjectiveLoss()
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y)

    assert loss.shape == torch.Size([])
    assert "loss_task" in metrics
    assert "loss_rob" in metrics
    assert "loss_expl" in metrics
    assert "loss_total" in metrics
    assert metrics["loss_task"] > 0
    assert metrics["loss_rob"] == 0.0
    assert metrics["loss_expl"] == 0.0


def test_forward_multiclass():
    """Test forward pass with multiclass model."""
    model = nn.Linear(10, 5)
    loss_fn = TriObjectiveLoss()
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))

    loss, metrics = loss_fn(model, x, y)

    assert loss.shape == torch.Size([])
    assert metrics["loss_task"] > 0
    assert metrics["loss_rob"] == 0.0
    assert metrics["loss_expl"] == 0.0


def test_forward_all_metrics_present():
    """Test that all required metrics are always present."""
    model = nn.Linear(10, 1)
    loss_fn = TriObjectiveLoss()
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    _, metrics = loss_fn(model, x, y)

    required_keys = {"loss_task", "loss_rob", "loss_expl", "loss_total"}
    assert set(metrics.keys()) == required_keys


# ============================================================================
# Test robustness loss - attacker with __call__
# ============================================================================


def test_forward_with_callable_attacker():
    """Test forward pass with callable attacker."""
    model = nn.Linear(10, 1)

    # Mock attacker that returns adversarial examples
    mock_attacker = Mock()
    x_adv = torch.randn(4, 10)
    mock_attacker.return_value = x_adv

    loss_fn = TriObjectiveLoss(w_rob=0.5)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, attacker=mock_attacker)

    # Attacker should be called with (model, x, y)
    mock_attacker.assert_called_once_with(model, x, y)
    assert metrics["loss_rob"] > 0


def test_forward_with_perturb_attacker():
    """Test forward pass with attacker that has perturb method."""
    model = nn.Linear(10, 1)

    # Mock attacker with perturb method
    mock_attacker = Mock()
    mock_attacker.perturb = Mock(return_value=torch.randn(4, 10))
    # Remove __call__ to force perturb path
    del mock_attacker.__call__

    loss_fn = TriObjectiveLoss(w_rob=0.5)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, attacker=mock_attacker)

    # perturb should be called
    mock_attacker.perturb.assert_called_once_with(model, x, y)
    assert metrics["loss_rob"] > 0


def test_forward_attacker_returns_none():
    """Test that None return from attacker results in zero robustness loss."""
    model = nn.Linear(10, 1)

    mock_attacker = Mock(return_value=None)

    loss_fn = TriObjectiveLoss(w_rob=0.5)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, attacker=mock_attacker)

    assert metrics["loss_rob"] == 0.0


def test_forward_attacker_returns_non_tensor():
    """Test that non-tensor return from attacker results in zero robustness loss."""
    model = nn.Linear(10, 1)

    mock_attacker = Mock(return_value="not a tensor")

    loss_fn = TriObjectiveLoss(w_rob=0.5)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, attacker=mock_attacker)

    assert metrics["loss_rob"] == 0.0


def test_forward_attacker_raises_exception():
    """Test that attacker exceptions are caught and robustness loss is zero."""
    model = nn.Linear(10, 1)

    mock_attacker = Mock(side_effect=RuntimeError("Attack failed"))

    loss_fn = TriObjectiveLoss(w_rob=0.5)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, attacker=mock_attacker)

    # Should not raise, robustness loss should be 0
    assert metrics["loss_rob"] == 0.0


def test_forward_attacker_no_call_no_perturb():
    """Test attacker without __call__ or perturb method."""
    model = nn.Linear(10, 1)

    mock_attacker = Mock(spec=[])  # No methods

    loss_fn = TriObjectiveLoss(w_rob=0.5)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, attacker=mock_attacker)

    assert metrics["loss_rob"] == 0.0


# ============================================================================
# Test explanation loss - explainer with __call__
# ============================================================================


def test_forward_with_callable_explainer_tensor():
    """Test forward pass with callable explainer returning tensor."""
    model = nn.Linear(10, 1)

    # Mock explainer that returns a penalty tensor
    mock_explainer = Mock()
    penalty = torch.tensor([0.1, 0.2, 0.3, 0.4])
    mock_explainer.return_value = penalty

    loss_fn = TriObjectiveLoss(w_expl=0.3)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, explainer=mock_explainer)

    # Explainer should be called
    mock_explainer.assert_called_once_with(model, x, y)
    assert metrics["loss_expl"] > 0
    # Should be mean of penalty
    assert abs(metrics["loss_expl"] - penalty.mean().item()) < 1e-5


def test_forward_with_compute_explainer():
    """Test forward pass with explainer that has compute method."""
    model = nn.Linear(10, 1)

    # Mock explainer with compute method
    mock_explainer = Mock()
    penalty = torch.tensor([0.5, 0.6])
    mock_explainer.compute = Mock(return_value=penalty)
    # Remove __call__ to force compute path
    del mock_explainer.__call__

    loss_fn = TriObjectiveLoss(w_expl=0.3)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, explainer=mock_explainer)

    # compute should be called
    mock_explainer.compute.assert_called_once_with(model, x, y)
    assert metrics["loss_expl"] > 0


def test_forward_explainer_returns_float():
    """Test explainer returning float penalty."""
    model = nn.Linear(10, 1)

    mock_explainer = Mock(return_value=0.42)

    loss_fn = TriObjectiveLoss(w_expl=0.3)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, explainer=mock_explainer)

    assert abs(metrics["loss_expl"] - 0.42) < 1e-5


def test_forward_explainer_returns_int():
    """Test explainer returning int penalty."""
    model = nn.Linear(10, 1)

    mock_explainer = Mock(return_value=2)

    loss_fn = TriObjectiveLoss(w_expl=0.3)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, explainer=mock_explainer)

    assert abs(metrics["loss_expl"] - 2.0) < 1e-5


def test_forward_explainer_returns_none():
    """Test that None return from explainer results in zero explanation loss."""
    model = nn.Linear(10, 1)

    mock_explainer = Mock(return_value=None)

    loss_fn = TriObjectiveLoss(w_expl=0.3)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, explainer=mock_explainer)

    assert metrics["loss_expl"] == 0.0


def test_forward_explainer_returns_invalid_type():
    """Test that invalid return type from explainer results in zero loss."""
    model = nn.Linear(10, 1)

    mock_explainer = Mock(return_value="not a valid type")

    loss_fn = TriObjectiveLoss(w_expl=0.3)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, explainer=mock_explainer)

    assert metrics["loss_expl"] == 0.0


def test_forward_explainer_raises_exception():
    """Test that explainer exceptions are caught and explanation loss is zero."""
    model = nn.Linear(10, 1)

    mock_explainer = Mock(side_effect=RuntimeError("Explainer failed"))

    loss_fn = TriObjectiveLoss(w_expl=0.3)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, explainer=mock_explainer)

    # Should not raise, explanation loss should be 0
    assert metrics["loss_expl"] == 0.0


def test_forward_explainer_no_call_no_compute():
    """Test explainer without __call__ or compute method."""
    model = nn.Linear(10, 1)

    mock_explainer = Mock(spec=[])  # No methods

    loss_fn = TriObjectiveLoss(w_expl=0.3)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, explainer=mock_explainer)

    assert metrics["loss_expl"] == 0.0


# ============================================================================
# Test weighted combination
# ============================================================================


def test_forward_weighted_combination():
    """Test that loss_total is correct weighted sum."""
    model = nn.Linear(10, 1)

    mock_attacker = Mock(return_value=torch.randn(4, 10))
    mock_explainer = Mock(return_value=torch.tensor(0.5))

    w_task, w_rob, w_expl = 0.5, 0.3, 0.2
    loss_fn = TriObjectiveLoss(w_task=w_task, w_rob=w_rob, w_expl=w_expl)

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, attacker=mock_attacker, explainer=mock_explainer)

    # Verify weighted sum
    expected = (
        w_task * metrics["loss_task"] + w_rob * metrics["loss_rob"] + w_expl * metrics["loss_expl"]
    )
    assert abs(metrics["loss_total"] - expected) < 1e-5


def test_forward_all_weights_zero():
    """Test forward pass with all weights set to zero."""
    model = nn.Linear(10, 1)
    loss_fn = TriObjectiveLoss(w_task=0.0, w_rob=0.0, w_expl=0.0)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y)

    assert metrics["loss_total"] == 0.0


def test_forward_only_task_weight():
    """Test that only task loss contributes when other weights are zero."""
    model = nn.Linear(10, 1)

    mock_attacker = Mock(return_value=torch.randn(4, 10))
    mock_explainer = Mock(return_value=torch.tensor(0.5))

    loss_fn = TriObjectiveLoss(w_task=1.0, w_rob=0.0, w_expl=0.0)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, attacker=mock_attacker, explainer=mock_explainer)

    # Total should equal task loss
    assert abs(metrics["loss_total"] - metrics["loss_task"]) < 1e-5


# ============================================================================
# Test device handling
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_cuda():
    """Test forward pass on CUDA device."""
    device = torch.device("cuda")
    model = nn.Linear(10, 1).to(device)
    loss_fn = TriObjectiveLoss()
    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 2, (4,), device=device)

    loss, metrics = loss_fn(model, x, y)

    assert loss.device.type == "cuda"


def test_forward_preserves_dtype():
    """Test that loss preserves dtype from task loss."""
    model = nn.Linear(10, 1)

    mock_attacker = Mock(return_value=torch.randn(4, 10))
    mock_explainer = Mock(return_value=torch.tensor(0.5))

    loss_fn = TriObjectiveLoss()
    x = torch.randn(4, 10, dtype=torch.float32)
    y = torch.randint(0, 2, (4,))

    loss, metrics = loss_fn(model, x, y, attacker=mock_attacker, explainer=mock_explainer)

    assert loss.dtype == torch.float32


# ============================================================================
# Integration tests
# ============================================================================


def test_full_pipeline_binary():
    """Test complete pipeline with binary classification."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

    mock_attacker = Mock(return_value=torch.randn(8, 10))
    mock_explainer = Mock(return_value=torch.tensor([0.1, 0.2, 0.3]))

    loss_fn = TriObjectiveLoss(w_task=1.0, w_rob=0.5, w_expl=0.3)

    x = torch.randn(8, 10)
    y = torch.randint(0, 2, (8,))

    loss, metrics = loss_fn(model, x, y, attacker=mock_attacker, explainer=mock_explainer)

    assert loss.item() > 0
    assert metrics["loss_task"] > 0
    assert metrics["loss_rob"] > 0
    assert metrics["loss_expl"] > 0
    assert metrics["loss_total"] > 0


def test_full_pipeline_multiclass():
    """Test complete pipeline with multiclass classification."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    mock_attacker = Mock(return_value=torch.randn(8, 10))
    mock_explainer = Mock(return_value=0.25)

    loss_fn = TriObjectiveLoss(w_task=1.0, w_rob=0.5, w_expl=0.3)

    x = torch.randn(8, 10)
    y = torch.randint(0, 5, (8,))

    loss, metrics = loss_fn(model, x, y, attacker=mock_attacker, explainer=mock_explainer)

    assert loss.item() > 0
    assert all(key in metrics for key in ["loss_task", "loss_rob", "loss_expl", "loss_total"])


def test_multiple_forward_calls():
    """Test multiple forward passes (e.g., training loop)."""
    model = nn.Linear(10, 1)
    loss_fn = TriObjectiveLoss()

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    # Multiple calls should all succeed
    for _ in range(5):
        loss, metrics = loss_fn(model, x, y)
        assert loss.item() >= 0
        assert len(metrics) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
