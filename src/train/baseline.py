# src/train/baseline.py
import argparse
import datetime
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models as tv_models

from src.data.nih_binary import CSVImageDataset


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # keep deterministic False for speed; flip True only if you must
    torch.use_deterministic_algorithms(False)


def build_model(name: str, num_out: int = 1):
    name = (name or "resnet18").lower()
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_out)
    return m


def make_sampler(y, balance: bool):
    if not balance:
        return None
    y = np.asarray(y, dtype=np.float32)
    p = max(y.mean(), 1e-6)
    w_pos = 0.5 / p
    w_neg = 0.5 / (1.0 - p)
    weights = torch.as_tensor([w_pos if t > 0.5 else w_neg for t in y], dtype=torch.float)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def main(cfg_path="configs/base.yaml"):
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(
        {
            "device": "cuda",
            "model": {"name": "resnet18"},
            "data": {
                "batch_size": 8,
                "img_size": 224,
                "train_csv": "",
                "val_csv": "",
            },
            "train": {
                "epochs": 5,
                "num_workers": 0,
                "amp": True,
                "patience": 2,
                "balance": True,
                "warmup_epochs": 0,
                "seed": 42,
            },
            "optim": {"lr": 1e-3, "weight_decay": 1e-4, "scheduler": "plateau"},
            "log": {"outdir": "results/runs"},
            "ckpt": {"dir": "results/checkpoints"},
        },
        cfg,
    )

    set_seed(int(cfg.train.seed))
    use_cuda = (cfg.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # --- logging: unique run dir ---
    run_dir = os.path.join(cfg.log.outdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    os.makedirs(cfg.ckpt.dir, exist_ok=True)

    # --- datasets ---
    train_ds = CSVImageDataset(cfg.data.train_csv, cfg.data.img_size, augment=True)
    val_ds = CSVImageDataset(cfg.data.val_csv, cfg.data.img_size, augment=False)

    # --- imbalance ---
    sampler = make_sampler(train_ds.y, bool(cfg.train.balance))

    # --- dataloaders ---
    common = dict(
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        persistent_workers=False,
    )
    if common["num_workers"] > 0:
        common["prefetch_factor"] = 2

    train_ld = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        **common,
    )
    val_ld = DataLoader(val_ds, batch_size=cfg.data.batch_size * 2, shuffle=False, **common)

    # --- model / opt / loss / amp ---
    m = build_model(cfg.model.name, num_out=1).to(device)
    p = max(float(np.mean(train_ds.y)), 1e-6)
    pos_weight = torch.tensor((1.0 - p) / p, dtype=torch.float, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.AdamW(m.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and cfg.train.amp))

    # --- schedulers ---
    sched = None
    if str(cfg.optim.scheduler).lower() == "plateau":
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
    elif str(cfg.optim.scheduler).lower() == "cosine":
        total_ep = int(cfg.train.epochs)
        warm = max(int(cfg.train.warmup_epochs), 0)

        def lr_lambda(epoch):
            if epoch < warm:
                return (epoch + 1) / max(warm, 1)
            t = (epoch - warm) / max(total_ep - warm, 1)
            return 0.5 * (1 + math.cos(math.pi * t))

        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    # --- epoch loop ---
    def run_epoch(loader, train=True, step0=0):
        m.train(train)
        tot, n = 0.0, 0
        logits_all, y_all = [], []
        step = step0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).unsqueeze(1)
            with torch.set_grad_enabled(train), torch.amp.autocast("cuda", enabled=(use_cuda and cfg.train.amp)):
                out = m(xb)
                loss = crit(out, yb)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                writer.add_scalar("train/loss", loss.item(), step)
                step += 1
            tot += loss.item() * xb.size(0)
            n += xb.size(0)
            logits_all.append(out.detach().cpu())
            y_all.append(yb.detach().cpu())
        logits = torch.cat(logits_all).squeeze(1)
        ytrue = torch.cat(y_all).squeeze(1)
        try:
            auc = roc_auc_score(ytrue.numpy(), logits.sigmoid().numpy())
        except Exception:
            auc = float("nan")
        return tot / max(n, 1), auc, step

    best_auc, wait, g = -math.inf, 0, 0
    for ep in range(int(cfg.train.epochs)):
        t0 = time.time()
        tr_loss, tr_auc, g = run_epoch(train_ld, True, g)
        va_loss, va_auc, _ = run_epoch(val_ld, False, 0)
        if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(va_loss)
        elif sched is not None:
            sched.step()
        writer.add_scalar("val/loss", va_loss, ep)
        writer.add_scalar("val/auroc", va_auc, ep)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], ep)
        print(
            f"Epoch {ep}: train {tr_loss:.4f}/{tr_auc:.3f} | " f"val {va_loss:.4f}/{va_auc:.3f} ({time.time()-t0:.1f}s)"
        )

        # save weights only
        last_path = os.path.join(cfg.ckpt.dir, "last.pt")
        best_path = os.path.join(cfg.ckpt.dir, "best.pt")
        torch.save(m.state_dict(), last_path)
        if va_auc > best_auc:
            best_auc = va_auc
            wait = 0
            torch.save(m.state_dict(), best_path)
            print(f"[ckpt] saved best (AUROC {best_auc:.3f}) -> {best_path}")
        else:
            wait += 1
            if wait > int(cfg.train.patience):
                print(f"[early-stop] patience reached (best AUROC {best_auc:.3f})")
                break

    writer.close()


# ------------------------ Tiny CLI smoke path (fast) -------------------------
def _smoke_run() -> None:
    """CPU-only, one mini-batch synthetic train step. Runs in a few seconds."""
    device = torch.device("cpu")
    model = tv_models.resnet18(weights=None).to(device)
    model.fc = nn.Linear(model.fc.in_features, 1).to(device)
    model.train()

    # Tiny fake data: 4 RGB 64x64, binary labels
    x = torch.randn(4, 3, 64, 64)
    y = torch.randint(0, 2, (4,)).float()
    dl = DataLoader(TensorDataset(x, y), batch_size=4)

    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()

    xb, yb = next(iter(dl))
    opt.zero_grad()
    logits = model(xb).squeeze(1)
    loss = loss_fn(logits, yb)
    loss.backward()
    opt.step()

    print(f"[SMOKE] loss={loss.item():.4f}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Run 1-batch synthetic training")
    p.add_argument("--config", default="configs/base.yaml", help="Path to OmegaConf config")
    return p.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    if args.smoke:
        _smoke_run()
    else:
        main(args.config)
