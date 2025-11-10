"""
test_robust_sweep.py
High-coverage tests for src.eval.robust_sweep with REAL models and data.

Tests the actual robust_sweep.py implementation that exists,
not hypothetical functions.
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
# Helper function tests (only test what exists)
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

    # Will raise RuntimeError regardless of PyTorch version
    with pytest.raises(RuntimeError, match="corrupt|archive"):
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

    with pytest.raises(RuntimeError):
        rs.load_model("resnet18", ckpt, torch.device("cpu"))


# ---------------------------------------------------------------------------
# main(): Test actual main() function
# ---------------------------------------------------------------------------


def test_main_with_real_components(temp_dir, monkeypatch, real_resnet18_state_dict):
    """Test main() with real ResNet18, real dataset, real attacks."""
    csv_path = _create_real_test_images(temp_dir, n=20)
    out_csv = os.path.join(temp_dir, "robust_sweep.csv")
    ckpt = _prep_valid_ckpt(temp_dir, monkeypatch, real_resnet18_state_dict)

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
            "0",
            "--fresh",
        ],
    )

    rs.main()

    assert os.path.exists(out_csv)
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
        assert len(rows) >= 2  # At least eps=0 and eps=4


def test_main_pgd_with_alpha_list(temp_dir, monkeypatch, real_resnet18_state_dict):
    """Test PGD with alpha_list parameter."""
    csv_path = _create_real_test_images(temp_dir, n=12)
    out_csv = os.path.join(temp_dir, "pgd_sweep.csv")
    ckpt = _prep_valid_ckpt(temp_dir, monkeypatch, real_resnet18_state_dict)

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
            "10",
            "--alpha_list",
            "1,2",
            "--fresh",
        ],
    )

    rs.main()

    assert os.path.exists(out_csv)
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
        assert any("PGD10_a1" in r["attack"] for r in rows)
        assert any("PGD10_a2" in r["attack"] for r in rows)


def test_main_deterministic_mode(temp_dir, monkeypatch, real_resnet18_state_dict):
    """Test --deterministic flag."""
    csv_path = _create_real_test_images(temp_dir, n=10)
    out_csv = os.path.join(temp_dir, "deterministic.csv")
    ckpt = _prep_valid_ckpt(temp_dir, monkeypatch, real_resnet18_state_dict)

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


def test_main_resume_skip_existing(temp_dir, monkeypatch, real_resnet18_state_dict, capsys):
    """Test resume logic that skips already-completed configurations."""
    csv_path = _create_real_test_images(temp_dir, n=10)
    out_csv = os.path.join(temp_dir, "resume.csv")

    # Pre-create output with completed config
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
            dict(
                attack="FGSM",
                eps_255=4,
                steps=0,
                alpha_255=0,
                AUC_clean=0.85,
                AUC_adv=0.75,
                AUC_drop=0.10,
            )
        )

    ckpt = _prep_valid_ckpt(temp_dir, monkeypatch, real_resnet18_state_dict)

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
            "0",
        ],
    )

    rs.main()
    captured = capsys.readouterr()
    assert "[skip]" in captured.out


def test_main_fresh_flag_removes_existing(temp_dir, monkeypatch, real_resnet18_state_dict):
    """Test --fresh flag removes existing output file."""
    csv_path = _create_real_test_images(temp_dir, n=8)
    out_csv = os.path.join(temp_dir, "fresh_test.csv")

    # Create pre-existing file
    with open(out_csv, "w") as f:
        f.write("OLD,DATA,SHOULD,BE,DELETED\n")

    assert os.path.exists(out_csv)
    old_content = open(out_csv).read()
    assert "OLD" in old_content

    ckpt = _prep_valid_ckpt(temp_dir, monkeypatch, real_resnet18_state_dict)

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
            "--fresh",
        ],
    )

    rs.main()

    # Verify old content is gone
    new_content = open(out_csv).read()
    assert "OLD" not in new_content
    assert "attack" in new_content
    assert "AUC_clean" in new_content


def test_main_nested_output_directory(temp_dir, monkeypatch, real_resnet18_state_dict):
    """Test that nested output directories are created."""
    csv_path = _create_real_test_images(temp_dir, n=8)
    nested_out = os.path.join(temp_dir, "a", "b", "c", "output.csv")
    ckpt = _prep_valid_ckpt(temp_dir, monkeypatch, real_resnet18_state_dict)

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
            nested_out,
            "--eps",
            "0",
            "--steps",
            "0",
        ],
    )

    rs.main()
    assert os.path.exists(nested_out)


def test_main_guard_exec_module(temp_dir, monkeypatch, real_resnet18_state_dict):
    """Execute module as __main__ to cover if __name__ == '__main__' guard."""
    csv_path = _create_real_test_images(temp_dir, n=10)
    out_csv = os.path.join(temp_dir, "exec_main.csv")
    ckpt = _prep_valid_ckpt(temp_dir, monkeypatch, real_resnet18_state_dict)

    module_path = rs.__file__
    spec = importlib.util.spec_from_file_location("__main__", module_path)
    module = importlib.util.module_from_spec(spec)

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
        ],
    )

    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass

    assert os.path.exists(out_csv)
