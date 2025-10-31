from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models as tv_models

from src.data.nih_binary import CSVImageDataset


# --- flexible Grad-CAM helpers (compatible with variants already handled in tests) ---
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
        # forward once to ensure hooks (some impls need this)
        _ = model(x)

    gc = _make_gc(model)
    heat = _run_generate(gc, x)  # shape can vary; normalize to [N,1,H,W]

    if heat.dim() == 3:
        # Assume [N,H,W] -> add channel
        heat = heat.unsqueeze(1)
    elif heat.dim() == 2:
        # Assume [H,W] -> add batch+channel
        heat = heat.unsqueeze(0).unsqueeze(0)

    # Take first item
    h = heat[0]  # [1,h,w]
    if h.shape[0] != 1:
        # if channel != 1, average
        h = h.mean(dim=0, keepdim=True)

    # Resize to input spatial size
    _, _, H, W = x.shape
    if h.shape[-2:] != (H, W):
        h = F.interpolate(h.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

    h2d = h.squeeze(0)  # [H,W]
    # Normalize to 0..1
    h2d = h2d - h2d.min()
    denom = torch.clamp(h2d.max(), min=1e-8)
    h2d = h2d / denom

    arr = (h2d.cpu().numpy() * 255.0).astype("uint8")
    Image.fromarray(arr, mode="L").save(out_path)


def _load_one_from_cfg(cfg_path: str, split: str = "val") -> torch.Tensor:
    """
    Load a single image tensor from CSV defined in config (val split by default).
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cfg_path)
    img_size = int(cfg.data.img_size)
    csv_path = cfg.data.val_csv if split == "val" else cfg.data.train_csv
    ds = CSVImageDataset(csv_path, img_size, augment=False)
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty.")
    # get first item, add batch dim
    x0, _ = ds[0]
    return x0.unsqueeze(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny.yaml", help="Path to YAML config")
    p.add_argument("--out", default="results/figures/gradcam_tiny.png", help="Output PNG path")
    p.add_argument("--split", choices=["val", "train"], default="val")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model("resnet18")
    x = _load_one_from_cfg(args.config, split=args.split)
    save_gradcam_png(model, x, out_path)
    print(f"[GRADCAM] wrote {out_path.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
