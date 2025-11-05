from pathlib import Path

import torch

import src.xai.export as E


def test_save_gradcam_png_exact_shape_no_avg_no_interp(monkeypatch, tmp_path):
    # Force _run_generate to return [1,1,H,W] exactly matching input size
    def _run_gen(_gc, x):
        N, C, H, W = x.shape
        return torch.rand(1, 1, H, W)

    monkeypatch.setattr(E, "_run_generate", _run_gen, raising=True)

    model = E.build_model()
    x = torch.rand(1, 3, 19, 13)
    out = tmp_path / "exact.png"
    E.save_gradcam_png(model, x, out)
    assert out.exists() and out.stat().st_size > 0
