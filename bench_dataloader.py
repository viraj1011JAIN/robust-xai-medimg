# bench_dataloader.py
import math
import time

import torch
from torch.utils.data import DataLoader

from src.data.nih_binary import CSVImageDataset


def main(csv, bs=16, workers=2, iters=200, drop_last=False):
    ds = CSVImageDataset(csv)
    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset is empty after file existence filtering.")

    eff_bs = min(bs, n)  # ensure we can yield at least 1 batch
    print(f"[bench] rows: {n} | requested bs={bs} -> eff_bs={eff_bs} | workers={workers}")

    kwargs = dict(
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        shuffle=True,
        drop_last=drop_last,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = 2

    ld = DataLoader(ds, batch_size=eff_bs, **kwargs)
    it = iter(ld)

    # warmup: at most one full pass
    warmup_iters = min(10, max(1, math.ceil(n / eff_bs)))
    for _ in range(warmup_iters):
        try:
            next(it)
        except StopIteration:
            it = iter(ld)
            next(it)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(ld)
            xb, yb = next(it)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt = time.time() - t0
    print(f"[bench] {iters} batches in {dt:.2f}s -> {dt/iters:.4f}s/batch")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--drop_last", action="store_true")
    args = ap.parse_args()
    main(args.csv, args.bs, args.workers, args.iters, args.drop_last)
