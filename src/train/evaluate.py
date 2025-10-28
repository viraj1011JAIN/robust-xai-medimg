import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from src.data.nih_binary import CSVImageDataset


def build_model(name: str):
    name = (name or "resnet18").lower()
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--roc_png", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CSVImageDataset(args.csv, img_size=args.img_size, augment=False)
    ld = DataLoader(
        ds, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=True
    )

    m = build_model(args.model).to(device)
    # PyTorch 2.5 supports weights_only=True (experimental); keep False fallback for compatibility.
    try:
        sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()

    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in ld:
            xb = xb.to(device, non_blocking=True)
            logits = m(xb).squeeze(1).cpu()
            all_logits.append(logits)
            all_y.append(yb)

    logits = torch.cat(all_logits).numpy()
    ytrue = torch.cat(all_y).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    auc = roc_auc_score(ytrue, probs)
    print("AUROC:", auc)

    if args.out_csv:
        import pandas as pd

        pd.DataFrame({"prob": probs, "y": ytrue}).to_csv(args.out_csv, index=False)
    if args.roc_png:
        fpr, tpr, _ = roc_curve(ytrue, probs)
        plt.figure()
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC (AUC={auc:.3f})")
        os.makedirs(os.path.dirname(args.roc_png), exist_ok=True)
        plt.savefig(args.roc_png, bbox_inches="tight")


if __name__ == "__main__":
    main()
