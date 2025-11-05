from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from src.data.nih_binary import CSVImageDataset


def _build_model(name: str, num_out: int = 1) -> nn.Module:
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_out)
    return m


def _build_loaders(cfg) -> Tuple[DataLoader, DataLoader]:
    train_ds = CSVImageDataset(cfg.data.train_csv, cfg.data.img_size, augment=True)
    val_ds = CSVImageDataset(cfg.data.val_csv, cfg.data.img_size, augment=False)

    pin_mem = bool((cfg.device == "cuda") and torch.cuda.is_available())
    common = dict(
        num_workers=int(cfg.train.num_workers),
        pin_memory=pin_mem,
        persistent_workers=False,
    )
    if common["num_workers"] > 0:
        common["prefetch_factor"] = 2

    train_ld = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True, **common)
    val_ld = DataLoader(val_ds, batch_size=max(1, cfg.data.batch_size * 2), shuffle=False, **common)
    return train_ld, val_ld


def evaluate(cfg_path: str, ckpt: str | None = None, dry_run: bool = False) -> tuple[float, float]:
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(
        {
            "device": "cuda",
            "model": {"name": "resnet18"},
            "data": {"batch_size": 8, "img_size": 224, "train_csv": "", "val_csv": ""},
            "train": {"num_workers": 0, "seed": 42},
        },
        cfg,
    )

    use_cuda = (cfg.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    _, val_ld = _build_loaders(cfg)
    model = _build_model(cfg.model.name, num_out=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    model.eval()
    logits_all, y_all, tot, n = [], [], 0.0, 0

    with torch.no_grad():
        for xb, yb in val_ld:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).unsqueeze(1)

            out = model(xb)
            loss = loss_fn(out, yb)

            tot += loss.item() * xb.size(0)
            n += xb.size(0)

            logits_all.append(out.detach().cpu())
            y_all.append(yb.detach().cpu())

            if dry_run:
                break

    logits = torch.cat(logits_all).squeeze(1)
    ytrue = torch.cat(y_all).squeeze(1)
    try:
        auc = roc_auc_score(ytrue.numpy(), logits.sigmoid().numpy())
    except Exception:
        auc = float("nan")

    avg_loss = tot / max(1, n)
    print(f"[EVAL] loss={avg_loss:.4f}  auroc={auc:.3f}  (batches={'1' if dry_run else 'all'})")
    return avg_loss, float(auc)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config (e.g., configs/tiny.yaml)")
    p.add_argument("--ckpt", default=None, help="Path to weights (optional)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one validation batch without requiring a checkpoint",
    )
    p.add_argument("--out", default=None, help="Write CSV summary to this path (loss,auroc)")
    return p.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    loss, auc = evaluate(args.config, ckpt=args.ckpt, dry_run=args.dry_run)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_path.exists()
        with out_path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["loss", "auroc"])
            w.writerow([f"{loss:.6f}", f"{auc:.6f}"])
        print(f"[EVAL] wrote CSV -> {out_path}")
