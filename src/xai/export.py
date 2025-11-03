# src/xai/export.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models as tv_models

# Optional pandas import (CSV helpers)
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

from src.data.nih_binary import CSVImageDataset


# --- small utilities expected by tests -------------------------------------------
def ensure_dir(p) -> Path:
    """Ensure directory p exists (mkdir -p) and return Path."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_npy(arr: np.ndarray, path: Path | str) -> Path:
    """Save numpy array to .npy and return the path."""
    path = Path(path)
    ensure_dir(path.parent)
    np.save(str(path), arr)
    return path


def load_npy(path: Path | str) -> np.ndarray:
    """Load numpy array from .npy."""
    return np.load(str(path))


def save_heatmap(hm: np.ndarray | torch.Tensor, path: Path | str) -> Path:
    """
    Save a float heatmap as 8-bit grayscale PNG.
    Accepts (H,W) or (1,H,W); min–max normalizes to [0,1].
    """
    path = Path(path)
    ensure_dir(path.parent)

    if isinstance(hm, torch.Tensor):
        hm = hm.detach().cpu().float().numpy()

    hm = np.squeeze(hm)
    if hm.ndim != 2:
        raise ValueError("save_heatmap expects (H,W) or (1,H,W)")

    hm = hm - hm.min()
    denom = float(hm.max()) if float(hm.max()) > 1e-12 else 1.0
    hm = hm / denom

    img = (hm * 255.0).astype("uint8")
    Image.fromarray(img, mode="L").save(path)
    return path


def save_csv(df, path: Path | str, index: bool = False) -> Path:
    """Save a pandas DataFrame to CSV; returns the path."""
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for save_csv but is not installed.")
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    return path


def load_csv(path: Path | str):
    """Load a pandas DataFrame from CSV."""
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for load_csv but is not installed.")
    return pd.read_csv(path)


def save_json(obj, path: Path | str, *, indent: int = 2) -> Path:
    """Save a Python object as JSON; returns the path."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    return path


def load_json(path: Path | str):
    """Load a Python object from JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --- flexible Grad-CAM helpers ---------------------------------------------------
def _make_gc(mod):
    from src.xai import gradcam

    if hasattr(gradcam, "GradCAM"):
        try:
            return gradcam.GradCAM(mod, target_layer_name="layer4")
        except TypeError:
            try:
                return gradcam.GradCAM(mod, target_layer="layer4")
            except TypeError:
                return gradcam.GradCAM(mod)
    if hasattr(gradcam, "get_gradcam"):
        return gradcam.get_gradcam(mod, layer="layer4")  # type: ignore[attr-defined]
    raise RuntimeError("No Grad-CAM entry point found (src.xai.gradcam)")


def _run_generate(gc, x: torch.Tensor) -> torch.Tensor:
    from src.xai import gradcam

    gen = getattr(gc, "generate", None)
    if callable(gen):
        try:
            return gen(x, class_idx=None)
        except TypeError:
            return gen(x)

    if callable(gc):
        try:
            return gc(x, class_idx=None)  # type: ignore[misc]
        except TypeError:
            return gc(x)  # type: ignore[misc]

    fn = getattr(gradcam, "gradcam", None)
    if callable(fn):
        try:
            return fn(gc, x, class_idx=None)
        except TypeError:
            return fn(gc, x)

    raise RuntimeError("Don't know how to invoke Grad-CAM for this object")


def build_model(name: str = "resnet18") -> torch.nn.Module:
    m = tv_models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 1)
    m.eval()
    return m


def save_gradcam_png(model: torch.nn.Module, x: torch.Tensor, out_path: Path) -> None:
    """
    Generate a Grad-CAM heatmap for the first image in a batch and save as PNG.
    Saves a single-channel grayscale heatmap (no matplotlib dependency).
    """
    model.eval()
    if x.dim() != 4:
        raise ValueError("x must be NCHW")

    with torch.no_grad():
        _ = model(x)  # some impls set up hooks on first forward

    gc = _make_gc(model)
    heat = _run_generate(gc, x)  # normalize to [N,1,H,W]

    if heat.dim() == 3:
        heat = heat.unsqueeze(1)  # [N,1,H,W]
    elif heat.dim() == 2:
        heat = heat.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    h = heat[0]  # [1,H,W]
    if h.shape[0] != 1:
        h = h.mean(dim=0, keepdim=True)

    _, _, H, W = x.shape
    if h.shape[-2:] != (H, W):
        h = F.interpolate(
            h.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)

    h2d = h.squeeze(0)
    h2d = h2d - h2d.min()
    denom = torch.clamp(h2d.max(), min=1e-8)
    h2d = h2d / denom

    arr = (h2d.cpu().numpy() * 255.0).astype("uint8")
    ensure_dir(Path(out_path).parent)
    Image.fromarray(arr, mode="L").save(out_path)


# --- tiny CLI used by tools/smoke_gradcam.py ------------------------------------
def _load_one_from_cfg(cfg_path: str, split: str = "val") -> torch.Tensor:
    """Load a single image tensor from CSV defined in config."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cfg_path)
    img_size = int(cfg.data.img_size)
    csv_path = cfg.data.val_csv if split == "val" else cfg.data.train_csv
    ds = CSVImageDataset(csv_path, img_size, augment=False)
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty.")
    x0, _ = ds[0]
    return x0.unsqueeze(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny.yaml", help="Path to YAML config")
    p.add_argument(
        "--out", default="results/figures/gradcam_tiny.png", help="Output PNG path"
    )
    p.add_argument("--split", choices=["val", "train"], default="val")
    args = p.parse_args()

    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    model = build_model("resnet18")
    x = _load_one_from_cfg(args.config, split=args.split)
    save_gradcam_png(model, x, out_path)
    print(f"[GRADCAM] wrote {out_path.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
