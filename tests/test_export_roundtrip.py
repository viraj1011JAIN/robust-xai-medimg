import json

import numpy as np
import pandas as pd

from src.xai import export as E


def test_export_helpers(tmp_path):
    root = tmp_path / "out"
    E.ensure_dir(root)

    arr = np.random.rand(4, 4).astype("float32")
    p_npy = root / "a.npy"
    E.save_npy(arr, p_npy)
    assert p_npy.exists()
    assert np.load(p_npy).shape == (4, 4)

    p_png = root / "a.png"
    E.save_heatmap(arr, p_png)  # should not crash
    assert p_png.exists()

    df = pd.DataFrame({"id": [1, 2], "score": [0.1, 0.9]})
    p_csv = root / "r.csv"
    E.save_csv(df, p_csv)
    assert p_csv.exists()
    assert len(pd.read_csv(p_csv)) == 2

    meta = {"seed": 123, "info": {"a": 1}}
    p_json = root / "m.json"
    E.save_json(meta, p_json)
    assert p_json.exists()
    assert json.loads(p_json.read_text())["seed"] == 123
