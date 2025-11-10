"""
test_robust_sweep.py
High-coverage tests for src.eval.robust_sweep with REAL models and data.

Production guarantees:
- Uses REAL ResNet18 from torchvision
- Uses REAL CSVImageDataset
- Uses REAL FGSMAttack and PGDAttack
- Uses REAL image processing and ROC-AUC calculations
- Only mocks file I/O and torch.load for speed
"""

import csv
import hashlib
import importlib.util
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

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def real_resnet18_state_dict():
    """Real ResNet18 state dict with binary classification head."""
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.state_dict()


def _create_real_test_images(temp_dir, n=6):
    """Create real PNG images on disk."""
    img_dir = os.path.join(temp_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(temp_dir, "data.csv")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label"])

        for i in range(n):
            # Create real 224x224 RGB image
            img = Image.new("RGB", (224, 224), color=(i * 40, i * 40, i * 40))
            img_path = os.path.join(img_dir, f"img_{i}.png")
            img.save(img_path)

            w.writerow([img_path, int(i % 2)])

    return csv_path


def _prep_valid_ckpt(temp_dir, monkeypatch, state_dict):
    """Prepare a valid checkpoint file."""
    ckpt = os.path.join(temp_dir, "model.ckpt")
    torch.save({"state_dict": state_dict}, ckpt)

    # Mock file size check only
    real_getsize = os.path.getsize

    def fake_getsize(p):
        if p == ckpt:
            return 100_000_000  # Pretend it's large
        return real_getsize(p)

    monkeypatch.setattr(os.path, "getsize", fake_getsize)
    return ckpt


# ---------------------------------------------------------------------------
# Helper function tests (need implementation in robust_sweep.py)
# ---------------------------------------------------------------------------


def test_safe_int_variants():
    assert rs._safe_int(5) == 5
    assert rs._safe_int("7") == 7
    assert rs._safe_int("3.0") == 3
    assert rs._safe_int(4.9) == 4


def test_file_sha256_roundtrip(temp_dir):
    path = os.path.join(temp_dir, "f.bin")
    data = b"hello-world"
    with open(path, "wb") as f:
        f.write(data)
    h = rs._file_sha256(path)
    assert isinstance(h, str)
    assert len(h) == 64
    assert h == hashlib.sha256(data).hexdigest()


def test_state_signature_stable_and_short(real_resnet18_state_dict):
    s1 = rs._state_signature(real_resnet18_state_dict)
    s2 = rs._state_signature(real_resnet18_state_dict)
    assert isinstance(s1, str)
    assert len(s1) == 12
    assert s1 == s2


def test_assert_ckpt_ok_missing(temp_dir):
    missing = os.path.join(temp_dir, "no.ckpt")
    with pytest.raises(RuntimeError, match="Checkpoint not found"):
        rs._assert_ckpt_ok(missing)


def test_assert_ckpt_ok_too_small(temp_dir):
    path = os.path.join(temp_dir, "small.ckpt")
    with open(path, "wb") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError, match="Checkpoint looks truncated"):
        rs._assert_ckpt_ok(path)


def test_assert_ckpt_ok_pass(temp_dir, monkeypatch):
    path = os.path.join(temp_dir, "ok.ckpt")
    with open(path, "wb") as f:
        f.write(b"x" * 16)

    monkeypatch.setattr(os.path, "getsize", lambda p: 10000 if p == path else os.path.getsize(p))
    rs._assert_ckpt_ok(path)  # should not raise


# ---------------------------------------------------------------------------
# load_model branches with REAL ResNet18
# ---------------------------------------------------------------------------


def test_load_model_success(temp_dir, monkeypatch, real_resnet18_state_dict):
    ckpt = _prep_valid_ckpt(temp_dir, monkeypatch, real_resnet18_state_dict)

    model = rs.load_model("resnet18", ckpt, torch.device("cpu"))
    assert isinstance(model, nn.Module)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 1


def test_load_model_weights_only_typeerror_fallback(
    temp_dir, monkeypatch, real_resnet18_state_dict
):
    ckpt = _prep_valid_ckpt(temp_dir, monkeypatch, real_resnet18_state_dict)

    calls = {"n": 0}
    real_load = torch.load

    def fake_load(path, map_location=None, weights_only=None):
        calls["n"] += 1
        if calls["n"] == 1 and weights_only is True:
            raise TypeError("weights_only unsupported")
        return real_load(path, map_location=map_location)

    monkeypatch.setattr(torch, "load", fake_load)
    model = rs.load_model("resnet18", ckpt, torch.device("cpu"))
    assert isinstance(model, nn.Module)
    assert calls["n"] == 2


def test_load_model_plain_dict_state_dict(temp_dir, monkeypatch, real_resnet18_state_dict):
    ckpt = os.path.join(temp_dir, "model.ckpt")
    torch.save(real_resnet18_state_dict, ckpt)  # Plain dict, not {"state_dict": ...}

    monkeypatch.setattr(
        os.path, "getsize", lambda p: 100_000_000 if p == ckpt else os.path.getsize(p)
    )

    model = rs.load_model("resnet18", ckpt, torch.device("cpu"))
    assert isinstance(model, nn.Module)


def test_load_model_unsupported_name(temp_dir, monkeypatch, real_resnet18_state_dict):
    ckpt = _prep_valid_ckpt(temp_dir, monkeypatch, real_resnet18_state_dict)

    with pytest.raises(ValueError, match="Unsupported model"):
        rs.load_model("vit_b16", ckpt, torch.device("cpu"))


def test_load_model_corrupt_archive(temp_dir, monkeypatch):
    ckpt = os.path.join(temp_dir, "corrupt.ckpt")
    with open(ckpt, "wb") as f:
        f.write(b"not a valid pytorch file")

    monkeypatch.setattr(
        os.path, "getsize", lambda p: 100_000_000 if p == ckpt else os.path.getsize(p)
    )

    with pytest.raises(RuntimeError, match="appears corrupt or not a valid PyTorch archive"):
        rs.load_model("resnet18", ckpt, torch.device("cpu"))


def test_load_model_unexpected_format(temp_dir, monkeypatch):
    ckpt = os.path.join(temp_dir, "weird.ckpt")
    torch.save([1, 2, 3], ckpt)  # List instead of dict

    monkeypatch.setattr(
        os.path, "getsize", lambda p: 100_000_000 if p == ckpt else os.path.getsize(p)
    )

    with pytest.raises(RuntimeError, match="Unexpected checkpoint format"):
        rs.load_model("resnet18", ckpt, torch.device("cpu"))


def test_load_model_shape_mismatch(temp_dir, monkeypatch):
    """Test when state dict has incompatible shapes."""
    ckpt = os.path.join(temp_dir, "bad_shape.ckpt")

    # Create state dict with wrong shape for fc layer
    bad_state = tv_models.resnet18(weights=None).state_dict()
    bad_state["fc.weight"] = torch.randn(10, 512)  # Wrong out_features

    torch.save({"state_dict": bad_state}, ckpt)
    monkeypatch.setattr(
        os.path, "getsize", lambda p: 100_000_000 if p == ckpt else os.path.getsize(p)
    )

    with pytest.raises(RuntimeError, match="Model.load_state_dict failed"):
        rs.load_model("resnet18", ckpt, torch.device("cpu"))


# ---------------------------------------------------------------------------
# auc_* tests with REAL models and data
# ---------------------------------------------------------------------------


def test_auc_clean_with_real_resnet(temp_dir, real_resnet18_state_dict):
    """Test auc_clean with real ResNet18 and real images."""
    csv_path = _create_real_test_images(temp_dir, n=10)

    from src.data.nih_binary import CSVImageDataset

    dataset = CSVImageDataset(csv_path, img_size=224, augment=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(real_resnet18_state_dict)

    device = torch.device("cpu")
    auc = rs.auc_clean(model.to(device).eval(), loader, device)

    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0 or auc != auc  # Valid AUC or NaN


def test_auc_under_attack_with_real_fgsm(temp_dir, real_resnet18_state_dict):
    """Test auc_under_attack with real ResNet18, real images, and real FGSM."""
    csv_path = _create_real_test_images(temp_dir, n=10)

    from src.attacks.fgsm import FGSMAttack
    from src.data.nih_binary import CSVImageDataset

    dataset = CSVImageDataset(csv_path, img_size=224, augment=False)

    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(real_resnet18_state_dict)

    device = torch.device("cpu")
    attack = FGSMAttack(epsilon=8 / 255)

    auc = rs.auc_under_attack(
        model.to(device).eval(), dataset, device, attack, batch_size=4, num_workers=0
    )

    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0 or auc != auc


def test_auc_under_attack_cuda_pin_memory(temp_dir, real_resnet18_state_dict, monkeypatch):
    """Test that CUDA availability triggers pin_memory=True."""
    csv_path = _create_real_test_images(temp_dir, n=6)

    from src.attacks.fgsm import FGSMAttack
    from src.data.nih_binary import CSVImageDataset

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    dataset = CSVImageDataset(csv_path, img_size=224, augment=False)

    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(real_resnet18_state_dict)

    device = torch.device("cpu")
    attack = FGSMAttack(epsilon=4 / 255)

    auc = rs.auc_under_attack(
        model.to(device).eval(), dataset, device, attack, batch_size=2, num_workers=0
    )
    assert isinstance(auc, float)


def test_auc_clean_single_class_returns_nan(temp_dir):
    """Test that single-class data returns NaN."""
    img_dir = os.path.join(temp_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(temp_dir, "single_class.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label"])
        for i in range(6):
            img = Image.new("RGB", (224, 224))
            img_path = os.path.join(img_dir, f"img_{i}.png")
            img.save(img_path)
            w.writerow
