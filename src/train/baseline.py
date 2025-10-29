# src/train/baseline.py
import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models as tv_models
from torchvision.models import ResNet18_Weights

try:
    import timm
except Exception:
    timm = None

from src.data.nih_binary import CSVImageDataset


def build_model(name: str, num_out: int = 1, pretrained: bool = True):
    """
    Build a model with optional ImageNet pretraining.
    - torchvision path for resnet18
    - timm path for anything else (efficientnet_b0, vit_tiny_patch16_224, …)
    """
    name = (name or "resnet18").lower()
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = tv_models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_out)
        return m

    if timm is None:
        raise ValueError(
            f"Requested model '{name}' but timm is not installed. " f"Either install timm or set model.name: resnet18."
        )
    return timm.create_model(name, pretrained=pretrained, num_classes=num_out)


def make_sampler(y, balance: bool):
    if not balance:
        return None
    import numpy as np

    y = np.asarray(y, dtype=np.float32)
    p = max(y.mean(), 1e-6)  # positive rate
    w_pos = 0.5 / p
    w_neg = 0.5 / (1.0 - p)
    weights = torch.as_tensor([w_pos if t > 0.5 else w_neg for t in y], dtype=torch.float)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def main(cfg_path="configs/base.yaml"):
    # -------- Config with safe defaults --------
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(
        {
            "device": "cuda",
            "model": {"name": "resnet18", "pretrained": True},
            "data": {"batch_size": 8, "img_size": 224, "train_csv": "", "val_csv": ""},
            "train": {
                "epochs": 5,
                "num_workers": 0,
                "amp": True,
                "patience": 2,
                "balance": True,
                "warmup_epochs": 0,
            },
            "optim": {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "scheduler": "plateau",  # "plateau" or "cosine"
            },
            "log": {"outdir": "results/runs"},
            "ckpt": {"dir": "results/checkpoints"},
        },
        cfg,
    )

    use_cuda = (cfg.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # -------- Datasets --------
    train_ds = CSVImageDataset(cfg.data.train_csv, cfg.data.img_size, augment=True)
    val_ds = CSVImageDataset(cfg.data.val_csv, cfg.data.img_size, augment=False)

    # -------- Loss with pos_weight (imbalance) --------
    import numpy as np

    p = max(float(np.mean(train_ds.y)), 1e-6)
    pos_weight = torch.tensor((1.0 - p) / p, dtype=torch.float, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -------- Sampler / DataLoaders (Windows-friendly) --------
    sampler = make_sampler(train_ds.y, bool(cfg.train.balance))
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
    val_ld = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size * 2,
        shuffle=False,
        **common,
    )

    # -------- Model / Optim / AMP --------
    m = build_model(
        name=str(cfg.model.name),
        num_out=1,
        pretrained=bool(cfg.model.get("pretrained", True)),
    ).to(device)

    opt = optim.AdamW(m.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and cfg.train.amp))

    # -------- Scheduler --------
    sched = None
    if str(cfg.optim.scheduler).lower() == "plateau":
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
    elif str(cfg.optim.scheduler).lower() == "cosine":
        total_ep = int(cfg.train.epochs)
        warmup = max(int(cfg.train.warmup_epochs), 0)

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            t = (epoch - warmup) / max(total_ep - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * t))

        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    writer = SummaryWriter(cfg.log.outdir)
    os.makedirs(cfg.ckpt.dir, exist_ok=True)

    # -------- Train / Val epoch --------
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
                step += xb.size(0)

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

    # -------- Loop with early stop + ckpts (weights only) --------
    best_auc = float("-inf")
    wait = 0
    g = 0
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
        print(f"Epoch {ep}: train {tr_loss:.4f}/{tr_auc:.3f} | val {va_loss:.4f}/{va_auc:.3f} ({time.time()-t0:.1f}s)")

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


if __name__ == "__main__":
    main()
