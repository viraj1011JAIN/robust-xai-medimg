import argparse
import csv
import math
import os

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack
from src.data.nih_binary import CSVImageDataset


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return int(float(x))


def load_model(name: str, ckpt: str, device: torch.device) -> nn.Module:
    name = (name or "resnet18").lower()
    if name != "resnet18":
        raise ValueError(f"Only resnet18 supported here (got {name})")
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    # silence future warning if available
    try:
        sd = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(ckpt, map_location=device)
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


@torch.no_grad()
def auc_clean(model: nn.Module, loader, device):
    logits, labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits.append(model(xb).sigmoid().squeeze().cpu())
        labels.append(yb.cpu())
    import torch as T

    L, Y = T.cat(logits), T.cat(labels)
    try:
        return roc_auc_score(Y.numpy(), L.numpy())
    except Exception:
        return float("nan")


def auc_under_attack(model: nn.Module, dataset, device, attack, attack_bs: int, eval_bs: int):
    # 1) generate adversarials in micro-batches
    atk_loader = DataLoader(dataset, batch_size=attack_bs, shuffle=False, num_workers=0, pin_memory=True)
    adv_x, adv_y = [], []
    for xb, yb in atk_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        x_adv = attack(model, xb, yb)  # fp32, eval mode inside
        adv_x.append(x_adv.cpu())
        adv_y.append(yb.cpu())
        torch.cuda.empty_cache()

    import torch as T

    X = T.cat(adv_x).to(device, non_blocking=True)
    Y = T.cat(adv_y).to(device, non_blocking=True)

    # 2) evaluate logits in larger eval batches
    logits = []
    for i in range(0, X.size(0), eval_bs):
        xb = X[i : i + eval_bs]
        logits.append(model(xb).sigmoid().squeeze().detach().cpu())
    L = T.cat(logits)

    try:
        return roc_auc_score(Y.cpu().numpy(), L.numpy())
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--attack_bs", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--eps", default="0,2,4,6")
    ap.add_argument("--steps", default="0,5,10")
    ap.add_argument("--alpha", type=float, default=1.0, help="alpha in /255 units")
    ap.add_argument("--out", default="results/metrics/robust_sweep.csv")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing CSV and start fresh")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CSVImageDataset(args.csv, img_size=args.img_size, augment=False)
    eval_loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=True)

    model = load_model(args.model, args.ckpt, device)
    clean = auc_clean(model, eval_loader, device)
    print(f"[clean] AUROC={clean:.3f}")

    # resume logic: support both old schema (eps,steps,auroc) and new schema (attack,eps_255,steps,AUC_*)
    done = set()
    if (not args.fresh) and os.path.exists(args.out):
        with open(args.out, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "attack" in r and "eps_255" in r and "steps" in r:
                    eps255 = _safe_int(r["eps_255"])
                    steps = _safe_int(r["steps"])
                    attack_name = r["attack"]
                    done.add((attack_name, eps255, steps))
                elif "eps" in r and "steps" in r:  # old schema
                    eps255 = _safe_int(r["eps"])
                    steps = _safe_int(r["steps"])
                    name = "FGSM" if steps == 0 else f"PGD{steps}"
                    done.add((f"{name}_{eps255}", eps255, steps))

    # sweep
    for e_str in args.eps.split(","):
        eps255 = _safe_int(e_str.strip())
        eps = eps255 / 255.0
        for s_str in args.steps.split(","):
            steps = _safe_int(s_str.strip())
            base = "FGSM" if steps == 0 else f"PGD{steps}"
            key = (f"{base}_{eps255}", eps255, steps)

            if key in done:
                print(f"[skip] already done -> {base}@eps{eps255}, steps={steps}")
                continue

            attack = (
                FGSMAttack(epsilon=eps)
                if steps == 0
                else PGDAttack(epsilon=eps, alpha=(args.alpha / 255.0), num_steps=steps)
            )

            try:
                adv_auc = auc_under_attack(model, ds, device, attack, args.attack_bs, args.bs)
                drop = (clean - adv_auc) if (not math.isnan(clean) and not math.isnan(adv_auc)) else float("nan")
                row = {
                    "attack": f"{base}_{eps255}",
                    "eps_255": eps255,
                    "steps": steps,
                    "AUC_clean": clean,
                    "AUC_adv": adv_auc,
                    "AUC_drop": drop,
                }
                # append progressively (resume-safe)
                write_mode = "a" if (os.path.exists(args.out) and not args.fresh) else "w"
                with open(args.out, write_mode, newline="") as f:
                    w = csv.DictWriter(f, fieldnames=row.keys())
                    if write_mode == "w":
                        w.writeheader()
                    w.writerow(row)
                print(f"{base:>8s}@{eps255:>2}/255: clean {clean:.3f} | adv {adv_auc:.3f} | drop {drop:.3f}")

            except RuntimeError as ex:
                print(f"[warn] {base}@eps{eps255} failed: {ex}")
                torch.cuda.empty_cache()

    print(f"[metrics] wrote/updated: {args.out}")


if __name__ == "__main__":
    main()
