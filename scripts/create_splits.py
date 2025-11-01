from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def make_splits(csv_path, label_col, out_json, seed=42, train=0.7, val=0.15, test=0.15):
    assert abs((train + val + test) - 1.0) < 1e-8
    df = pd.read_csv(csv_path)
    y = df[label_col]
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    idx_trainval, idx_test = next(sss1.split(df, y))
    df_trainval, y_trainval = df.iloc[idx_trainval], y.iloc[idx_trainval]
    val_ratio = val / (train + val)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    idx_train, idx_val = next(sss2.split(df_trainval, y_trainval))
    splits = {
        "seed": seed,
        "counts": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "indices": {
            "train": df_trainval.index[idx_train].tolist(),
            "val": df_trainval.index[idx_val].tolist(),
            "test": df.index[idx_test].tolist(),
        },
    }
    _write_json(out_json, splits)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("label")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    make_splits(a.csv, a.label, a.out, seed=a.seed)
