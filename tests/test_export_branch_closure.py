import types
from pathlib import Path

import torch

import src.xai.export as E


def test_make_gc_raises_when_no_entrypoints(monkeypatch):
    # Remove all entry points so _make_gc raises
    real = E.gradcam
    dummy = types.SimpleNamespace(GradCAM=None, get_gradcam=None, gradcam=None)
    monkeypatch.setattr(E, "gradcam", dummy, raising=True)
    try:
        m = E.build_model()
        try:
            E._make_gc(m)
        except RuntimeError:
            pass  # expected
        else:
            assert False, "Expected RuntimeError when no gradcam entrypoints"
    finally:
        monkeypatch.setattr(E, "gradcam", real, raising=True)


def test_save_gradcam_png_hits_2d_avg_and_interp(monkeypatch, tmp_path):
    # Force _run_generate to return a 3-channel heatmap with wrong size (triggers avg + interpolate)
    def _run_gen(_gc, x):
        # return [1, 3, h, w] that differ from input size
        return torch.rand(1, 3, 12, 9)

    monkeypatch.setattr(E, "_run_generate", _run_gen, raising=True)

    model = E.build_model()
    x = torch.rand(1, 3, 16, 10)
    out = tmp_path / "cam.png"
    E.save_gradcam_png(model, x, out)
    assert out.exists()


def test_save_gradcam_png_hits_2d_branch(monkeypatch, tmp_path):
    # Now return 2D heatmap [H, W] to hit the 2D -> unsqueeze twice branch
    def _run_gen(_gc, x):
        return torch.rand(7, 5)

    monkeypatch.setattr(E, "_run_generate", _run_gen, raising=True)

    model = E.build_model()
    x = torch.rand(1, 3, 7, 5)
    out = tmp_path / "cam2.png"
    E.save_gradcam_png(model, x, out)
    assert out.exists()
