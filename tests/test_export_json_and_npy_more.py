# tests/test_export_json_and_npy_more.py
from pathlib import Path

import numpy as np

from src.xai import export as E


def test_npy_roundtrip_and_dirs(tmp_path):
    arr = np.arange(9, dtype="float32").reshape(3, 3)
    p = tmp_path / "sub" / "a.npy"
    E.save_npy(arr, p)
    got = E.load_npy(p)
    assert got.shape == (3, 3) and float(got[0, 1]) == 1.0


def test_json_roundtrip_and_indent(tmp_path):
    obj = {"a": 1, "b": {"c": 2}}
    p = tmp_path / "out" / "x.json"
    E.save_json(obj, p, indent=4)
    got = E.load_json(p)
    assert got == obj
