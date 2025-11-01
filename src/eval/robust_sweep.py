import argparse
import csv
import hashlib
import json
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


def _file_sha256(path: str, nbytes: int = 1 << 20) -> str:
    """SHA256 of file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(nbytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _state_signature(state_dict: dict) -> str:
    """
    Lightweight signature of keys, shapes, dtypes to verify which weights were used
    without hashing full tensor contents (fast + deterministic).
    """
    items = []
    for k, v in state_dict.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else None
        dtype = str(v.dtype) if hasattr(v, "dtype") else None
        items.append((k, shape, dtype))
    s = json.dumps(
        sorted(items, key=lambda t: t[0]), separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _assert_ckpt_ok(path: str, min_bytes: int = 4096):
    """Fail fast if the checkpoint is missing or obviously truncated."""
    if not os.path.exists(path):
        raise RuntimeError(f"Checkpoint not found: {path}")
    sz = os.path.getsize(path)
    if sz < min_bytes:
        raise RuntimeError(
            f"Checkpoint looks truncated (size={sz} bytes < {min_bytes}): {path}"
        )


def load_model(name: str, ckpt: str, device: torch.device) -> nn.Module:
    name = (name or "resnet18").lower()
    if name != "resnet18":
        raise ValueError(f"Only resnet18 supported here (got {name})")

    # Preflight validation (existence + min size)
    _assert_ckpt_ok(ckpt)

    m = tv_models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)

    # Load the checkpoint robustly with clearer errors for corrupt zip
    try:
        try:
            raw = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:
            raw = torch.load(ckpt, map_location=device)
    except Exception as e:
        msg = str(e)
        if "PytorchStreamReader" in msg or "central directory" in msg:
            raise RuntimeError(
                f"Checkpoint appears corrupt or not a valid PyTorch archive: {ckpt}\n{e}"
            )
        raise RuntimeError(f"Failed to load checkpoint file (torch.load): {ckpt}\n{e}")

    # Unwrap common formats
    if (
        isinstance(raw, dict)
        and "state_dict" in raw
        and isinstance(raw["state_dict"], dict)
    ):
        sd = raw["state_dict"]
    elif isinstance(raw, dict):
        # Heuristic: treat the dict as a state_dict if it looks like one (Tensor leaves)
        sd = raw
    else:
        raise RuntimeError(
            f"Unexpected checkpoint format. Expected dict or dict['state_dict'], got: {type(raw)}"
        )

    # Load weights (strict True so corrupt/truncated mismatches fail loudly)
    try:
        missing, unexpected = m.load_state_dict(sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Missing keys: {missing}, Unexpected keys: {unexpected}"
            )
    except Exception as e:
        raise RuntimeError(f"Model.load_state_dict failed for {ckpt}: {e}")

    # Report fingerprinting info
    file_sha = _file_sha256(ckpt)[:12]
    sig = _state_signature(sd)
    print(
        f"[ckpt] path={ckpt} | file_sha256={file_sha} | tensors={len(sd)} | sig={sig}"
    )

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


def auc_under_attack(
    model: nn.Module, dataset, device, attack, attack_bs: int, eval_bs: int
):
    # pin_memory only if CUDA is available
    pin = device.type == "cuda"
    # 1) generate adversarials in micro-batches
    atk_loader = DataLoader(
        dataset, batch_size=attack_bs, shuffle=False, num_workers=0, pin_memory=pin
    )
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
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--attack_bs", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--eps", default="0,2,4,6")
    ap.add_argument("--steps", default="0,5,10")
    ap.add_argument(
        "--alpha", type=float, default=1.0, help="alpha in /255 units (single value)"
    )
    ap.add_argument(
        "--alpha_list",
        type=str,
        default="",
        help="comma-separated alphas in /255 units, e.g. '1,2,3'",
    )
    ap.add_argument("--out", default="results/metrics/robust_sweep.csv")
    ap.add_argument(
        "--fresh", action="store_true", help="ignore any existing CSV and start fresh"
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="cudnn deterministic mode & fixed seeds",
    )
    args = ap.parse_args()

    # helpful echo
    print(f"[args] ckpt={args.ckpt}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determinism (optional)
    if args.deterministic:
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    # figure alphas to sweep
    if args.alpha_list.strip():
        alphas_255 = [
            _safe_int(x.strip()) for x in args.alpha_list.split(",") if x.strip()
        ]
    else:
        alphas_255 = [_safe_int(args.alpha)]

    # CSV header/append handling
    header = [
        "attack",
        "eps_255",
        "steps",
        "alpha_255",
        "AUC_clean",
        "AUC_adv",
        "AUC_drop",
    ]
    first_write = True
    if args.fresh and os.path.exists(args.out):
        os.remove(args.out)
    elif os.path.exists(args.out):
        first_write = False

    ds = CSVImageDataset(args.csv, img_size=args.img_size, augment=False)
    pin = device.type == "cuda"
    eval_loader = DataLoader(
        ds, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=pin
    )

    model = load_model(args.model, args.ckpt, device)
    clean = auc_clean(model, eval_loader, device)
    print(f"[clean] AUROC={clean:.3f}")

    # resume logic: support old & new schema
    done = set()
    if (not args.fresh) and os.path.exists(args.out):
        with open(args.out, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "attack" in r and "eps_255" in r and "steps" in r:
                    eps255 = _safe_int(r["eps_255"])
                    steps = _safe_int(r["steps"])
                    attack_name = r["attack"]
                    # prefer alpha_255 if present
                    a255 = (
                        _safe_int(r["alpha_255"])
                        if "alpha_255" in r and r["alpha_255"] != ""
                        else None
                    )
                    done.add((attack_name, eps255, steps, a255))
                elif "eps" in r and "steps" in r:  # very old schema
                    eps255 = _safe_int(r["eps"])
                    steps = _safe_int(r["steps"])
                    name = "FGSM" if steps == 0 else f"PGD{steps}"
                    done.add((f"{name}_{eps255}", eps255, steps, None))

    # sweep
    for e_str in args.eps.split(","):
        eps255 = _safe_int(e_str.strip())
        eps = eps255 / 255.0
        for s_str in args.steps.split(","):
            steps = _safe_int(s_str.strip())
            base = "FGSM" if steps == 0 else f"PGD{steps}"
            for alpha255 in alphas_255:
                # label: PGD includes alpha tag to uniquely identify
                attack_label = base if steps == 0 else f"{base}_a{alpha255}"
                key = (attack_label, eps255, steps, alpha255 if steps != 0 else None)

                if key in done:
                    print(
                        f"[skip] already done -> {attack_label}@eps{eps255}, steps={steps}"
                    )
                    continue

                attack = (
                    FGSMAttack(epsilon=eps)
                    if steps == 0
                    else PGDAttack(
                        epsilon=eps, alpha=(alpha255 / 255.0), num_steps=steps
                    )
                )

                try:
                    adv_auc = auc_under_attack(
                        model, ds, device, attack, args.attack_bs, args.bs
                    )
                    drop = (
                        (clean - adv_auc)
                        if (not math.isnan(clean) and not math.isnan(adv_auc))
                        else float("nan")
                    )
                    row = {
                        "attack": attack_label,
                        "eps_255": eps255,
                        "steps": steps,
                        "alpha_255": (0 if steps == 0 else _safe_int(alpha255)),
                        "AUC_clean": clean,
                        "AUC_adv": adv_auc,
                        "AUC_drop": drop,
                    }

                    # write header once per run, then append rows
                    mode = "w" if first_write else "a"
                    with open(args.out, mode, newline="") as f:
                        w = csv.DictWriter(f, fieldnames=header)
                        if first_write:
                            w.writeheader()
                            first_write = False
                        w.writerow(row)

                    print(
                        f"{attack_label:>10s}@{eps255:>2}/255: "
                        f"clean {clean:.3f} | adv {adv_auc:.3f} | drop {drop:.3f}"
                    )

                except RuntimeError as ex:
                    print(f"[warn] {attack_label}@eps{eps255} failed: {ex}")
                    torch.cuda.empty_cache()

    print(f"[metrics] wrote/updated: {args.out}")


if __name__ == "__main__":
    main()
