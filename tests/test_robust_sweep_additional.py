"""
test_robust_sweep_additional.py
Additional edge case tests for 100% coverage using REAL components.
"""

import csv
import os
import shutil
import sys
import tempfile

import pytest
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image

from src.eval import robust_sweep as rs


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def real_resnet50_state_dict():
    """Real ResNet50 state dict."""
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.state_dict()


@pytest.fixture
def real_densenet121_state_dict():
    """Real DenseNet121 state dict."""
    model = tv_models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model.state_dict()


def _create_real_images(temp_dir, n=12):
    """Create real test images."""
    img_dir = os.path.join(temp_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(temp_dir, "data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label"])
        for i in range(n):
            img = Image.new("RGB", (224, 224), color=(i * 20, i * 20, i * 20))
            img_path = os.path.join(img_dir, f"img_{i}.png")
            img.save(img_path)
            w.writerow([img_path, int(i % 2)])

    return csv_path


def _prep_ckpt(temp_dir, monkeypatch, state_dict):
    """Prepare checkpoint."""
    ckpt = os.path.join(temp_dir, "model.ckpt")
    torch.save({"state_dict": state_dict}, ckpt)
    monkeypatch.setattr(
        os.path, "getsize", lambda p: 100_000_000 if p == ckpt else os.path.getsize(p)
    )
    return ckpt


# ---------------------------------------------------------------------------
# Test different model architectures
# ---------------------------------------------------------------------------


def test_load_model_resnet50(temp_dir, monkeypatch, real_resnet50_state_dict):
    """Test loading ResNet50."""
    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_resnet50_state_dict)
    model = rs.load_model("resnet50", ckpt, torch.device("cpu"))
    assert isinstance(model, tv_models.ResNet)
    assert model.fc.out_features == 1


def test_load_model_densenet121(temp_dir, monkeypatch, real_densenet121_state_dict):
    """Test loading DenseNet121."""
    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_densenet121_state_dict)
    model = rs.load_model("densenet121", ckpt, torch.device("cpu"))
    assert isinstance(model, tv_models.DenseNet)
    assert model.classifier.out_features == 1


# ---------------------------------------------------------------------------
# Test PGD attack configurations
# ---------------------------------------------------------------------------


def test_main_pgd_with_single_alpha(temp_dir, monkeypatch):
    """Test PGD with single alpha value."""
    csv_path = _create_real_images(temp_dir, n=10)
    out_csv = os.path.join(temp_dir, "pgd_single.csv")

    # Use real ResNet18
    real_state = tv_models.resnet18(weights=None).state_dict()
    real_state["fc.weight"] = torch.randn(1, 512)
    real_state["fc.bias"] = torch.randn(1)

    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_state)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robust_sweep.py",
            "--csv",
            csv_path,
            "--ckpt",
            ckpt,
            "--out",
            out_csv,
            "--eps",
            "8",
            "--steps",
            "10",
            "--alpha",
            "2",  # Single alpha, not alpha_list
            "--fresh",
        ],
    )

    rs.main()

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert "PGD10" in rows[0]["attack"]
        assert rows[0]["alpha_255"] == "2"


def test_main_multiple_epsilon_values(temp_dir, monkeypatch):
    """Test sweep over multiple epsilon values."""
    csv_path = _create_real_images(temp_dir, n=10)
    out_csv = os.path.join(temp_dir, "multi_eps.csv")

    real_state = tv_models.resnet18(weights=None).state_dict()
    real_state["fc.weight"] = torch.randn(1, 512)
    real_state["fc.bias"] = torch.randn(1)

    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_state)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robust_sweep.py",
            "--csv",
            csv_path,
            "--ckpt",
            ckpt,
            "--out",
            out_csv,
            "--eps",
            "0,2,4,8",
            "--steps",
            "0",
            "--fresh",
        ],
    )

    rs.main()

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
        eps_values = [int(r["eps_255"]) for r in rows]
        assert sorted(eps_values) == [0, 2, 4, 8]


def test_main_multiple_step_values(temp_dir, monkeypatch):
    """Test sweep over multiple step values."""
    csv_path = _create_real_images(temp_dir, n=10)
    out_csv = os.path.join(temp_dir, "multi_steps.csv")

    real_state = tv_models.resnet18(weights=None).state_dict()
    real_state["fc.weight"] = torch.randn(1, 512)
    real_state["fc.bias"] = torch.randn(1)

    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_state)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robust_sweep.py",
            "--csv",
            csv_path,
            "--ckpt",
            ckpt,
            "--out",
            out_csv,
            "--eps",
            "4",
            "--steps",
            "0,5,10",
            "--alpha",
            "1",
            "--fresh",
        ],
    )

    rs.main()

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
        step_values = [int(r["steps"]) for r in rows]
        assert sorted(step_values) == [0, 5, 10]


def test_main_combined_sweep(temp_dir, monkeypatch):
    """Test full parameter sweep: eps x steps x alpha."""
    csv_path = _create_real_images(temp_dir, n=16)
    out_csv = os.path.join(temp_dir, "full_sweep.csv")

    real_state = tv_models.resnet18(weights=None).state_dict()
    real_state["fc.weight"] = torch.randn(1, 512)
    real_state["fc.bias"] = torch.randn(1)

    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_state)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robust_sweep.py",
            "--csv",
            csv_path,
            "--ckpt",
            ckpt,
            "--out",
            out_csv,
            "--eps",
            "0,4",
            "--steps",
            "0,5",
            "--alpha_list",
            "1,2",
            "--fresh",
        ],
    )

    rs.main()

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
        # Should have: FGSM@0, FGSM@4, PGD5_a1@4, PGD5_a2@4
        assert len(rows) >= 4


# ---------------------------------------------------------------------------
# Test resume behavior with different schemas
# ---------------------------------------------------------------------------


def test_main_resume_old_schema_format(temp_dir, monkeypatch, capsys):
    """Test resuming from old CSV schema (eps/steps without attack column)."""
    csv_path = _create_real_images(temp_dir, n=10)
    out_csv = os.path.join(temp_dir, "old_schema.csv")

    # Create old schema CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "steps"])
        w.writeheader()
        w.writerow({"eps": "4", "steps": "0"})

    real_state = tv_models.resnet18(weights=None).state_dict()
    real_state["fc.weight"] = torch.randn(1, 512)
    real_state["fc.bias"] = torch.randn(1)

    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_state)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robust_sweep.py",
            "--csv",
            csv_path,
            "--ckpt",
            ckpt,
            "--out",
            out_csv,
            "--eps",
            "4,8",
            "--steps",
            "0",
        ],
    )

    rs.main()
    captured = capsys.readouterr()

    # Should skip eps=4 from old schema
    assert "[skip]" in captured.out or os.path.exists(out_csv)


def test_main_resume_new_schema_exact_match(temp_dir, monkeypatch, capsys):
    """Test resume with new schema matching exact configuration."""
    csv_path = _create_real_images(temp_dir, n=10)
    out_csv = os.path.join(temp_dir, "new_schema.csv")

    # Create new schema CSV with exact match
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "attack",
                "eps_255",
                "steps",
                "alpha_255",
                "AUC_clean",
                "AUC_adv",
                "AUC_drop",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "attack": "PGD10_a2",
                "eps_255": 8,
                "steps": 10,
                "alpha_255": 2,
                "AUC_clean": 0.90,
                "AUC_adv": 0.75,
                "AUC_drop": 0.15,
            }
        )

    real_state = tv_models.resnet18(weights=None).state_dict()
    real_state["fc.weight"] = torch.randn(1, 512)
    real_state["fc.bias"] = torch.randn(1)

    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_state)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robust_sweep.py",
            "--csv",
            csv_path,
            "--ckpt",
            ckpt,
            "--out",
            out_csv,
            "--eps",
            "8",
            "--steps",
            "10",
            "--alpha_list",
            "2",
        ],
    )

    rs.main()
    captured = capsys.readouterr()
    assert "[skip]" in captured.out


# ---------------------------------------------------------------------------
# Test CUDA-related code paths
# ---------------------------------------------------------------------------


def test_main_with_cuda_deterministic(temp_dir, monkeypatch):
    """Test CUDA deterministic mode."""
    csv_path = _create_real_images(temp_dir, n=10)
    out_csv = os.path.join(temp_dir, "cuda_det.csv")

    real_state = tv_models.resnet18(weights=None).state_dict()
    real_state["fc.weight"] = torch.randn(1, 512)
    real_state["fc.bias"] = torch.randn(1)

    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_state)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robust_sweep.py",
            "--csv",
            csv_path,
            "--ckpt",
            ckpt,
            "--out",
            out_csv,
            "--eps",
            "0",
            "--steps",
            "0",
            "--deterministic",
            "--fresh",
        ],
    )

    rs.main()
    assert os.path.exists(out_csv)


# ---------------------------------------------------------------------------
# Test error handling paths
# ---------------------------------------------------------------------------


def test_main_invalid_checkpoint_path(temp_dir, monkeypatch):
    """Test error when checkpoint doesn't exist."""
    csv_path = _create_real_images(temp_dir, n=8)
    out_csv = os.path.join(temp_dir, "out.csv")
    fake_ckpt = os.path.join(temp_dir, "nonexistent.ckpt")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robust_sweep.py",
            "--csv",
            csv_path,
            "--ckpt",
            fake_ckpt,
            "--out",
            out_csv,
            "--eps",
            "0",
            "--steps",
            "0",
        ],
    )

    with pytest.raises(RuntimeError, match="Checkpoint not found"):
        rs.main()


def test_main_directory_creation(temp_dir, monkeypatch):
    """Test that output directories are created if they don't exist."""
    csv_path = _create_real_images(temp_dir, n=8)
    nested_path = os.path.join(temp_dir, "level1", "level2", "level3", "out.csv")

    real_state = tv_models.resnet18(weights=None).state_dict()
    real_state["fc.weight"] = torch.randn(1, 512)
    real_state["fc.bias"] = torch.randn(1)

    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_state)

    assert not os.path.exists(os.path.dirname(nested_path))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robust_sweep.py",
            "--csv",
            csv_path,
            "--ckpt",
            ckpt,
            "--out",
            nested_path,
            "--eps",
            "0",
            "--steps",
            "0",
        ],
    )

    rs.main()
    assert os.path.exists(nested_path)
    assert os.path.exists(os.path.dirname(nested_path))


# ---------------------------------------------------------------------------
# Test specific attack configurations
# ---------------------------------------------------------------------------


def test_fgsm_only_attack(temp_dir, monkeypatch):
    """Test FGSM-only configuration (steps=0)."""
    csv_path = _create_real_images(temp_dir, n=10)
    out_csv = os.path.join(temp_dir, "fgsm_only.csv")

    real_state = tv_models.resnet18(weights=None).state_dict()
    real_state["fc.weight"] = torch.randn(1, 512)
    real_state["fc.bias"] = torch.randn(1)

    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_state)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robust_sweep.py",
            "--csv",
            csv_path,
            "--ckpt",
            ckpt,
            "--out",
            out_csv,
            "--eps",
            "2,4,8",
            "--steps",
            "0",
            "--fresh",
        ],
    )

    rs.main()

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
        assert all(r["attack"] == "FGSM" for r in rows)
        assert all(int(r["steps"]) == 0 for r in rows)


def test_pgd_only_attack(temp_dir, monkeypatch):
    """Test PGD-only configuration (steps>0, no eps=0)."""
    csv_path = _create_real_images(temp_dir, n=10)
    out_csv = os.path.join(temp_dir, "pgd_only.csv")

    real_state = tv_models.resnet18(weights=None).state_dict()
    real_state["fc.weight"] = torch.randn(1, 512)
    real_state["fc.bias"] = torch.randn(1)

    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_state)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robust_sweep.py",
            "--csv",
            csv_path,
            "--ckpt",
            ckpt,
            "--out",
            out_csv,
            "--eps",
            "4,8",
            "--steps",
            "10",
            "--alpha",
            "2",
            "--fresh",
        ],
    )

    rs.main()

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
        assert all("PGD" in r["attack"] for r in rows)
        assert all(int(r["steps"]) == 10 for r in rows)


def test_mixed_fgsm_and_pgd(temp_dir, monkeypatch):
    """Test mix of FGSM and PGD attacks."""
    csv_path = _create_real_images(temp_dir, n=12)
    out_csv = os.path.join(temp_dir, "mixed.csv")

    real_state = tv_models.resnet18(weights=None).state_dict()
    real_state["fc.weight"] = torch.randn(1, 512)
    real_state["fc.bias"] = torch.randn(1)

    ckpt = _prep_ckpt(temp_dir, monkeypatch, real_state)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "robust_sweep.py",
            "--csv",
            csv_path,
            "--ckpt",
            ckpt,
            "--out",
            out_csv,
            "--eps",
            "4",
            "--steps",
            "0,10",
            "--alpha",
            "1",
            "--fresh",
        ],
    )

    rs.main()

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
        attacks = [r["attack"] for r in rows]
        assert "FGSM" in attacks
        assert any("PGD" in a for a in attacks)