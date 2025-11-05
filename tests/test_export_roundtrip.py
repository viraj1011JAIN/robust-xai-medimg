# tests/test_export_roundtrip.py
import json

import numpy as np
import pandas as pd

from src.xai import export as E


def test_export_helpers(tmp_path):
    """Test all export helper functions for roundtrip correctness."""
    root = tmp_path / "out"
    E.ensure_dir(root)
    assert root.exists(), "Directory should be created"

    # Test numpy array save/load
    arr = np.random.rand(4, 4).astype("float32")
    p_npy = root / "a.npy"
    E.save_npy(arr, p_npy)
    assert p_npy.exists(), "NPY file should exist"
    loaded = np.load(p_npy)
    assert loaded.shape == (4, 4), "Loaded array should be 4x4"
    assert np.allclose(arr, loaded), "Loaded array should match original"

    # Test heatmap generation
    p_png = root / "a.png"
    E.save_heatmap(arr, p_png)
    assert p_png.exists(), "PNG file should exist"

    # Test CSV save/load
    df = pd.DataFrame({"id": [1, 2], "score": [0.1, 0.9]})
    p_csv = root / "r.csv"
    E.save_csv(df, p_csv)
    assert p_csv.exists(), "CSV file should exist"
    loaded_df = pd.read_csv(p_csv)
    assert len(loaded_df) == 2, "Loaded dataframe should have 2 rows"
    assert list(loaded_df.columns) == ["id", "score"], "Columns should match"

    # Test JSON save/load
    meta = {"seed": 123, "info": {"a": 1}}
    p_json = root / "m.json"
    E.save_json(meta, p_json)
    assert p_json.exists(), "JSON file should exist"
    loaded_json = json.loads(p_json.read_text())
    assert loaded_json["seed"] == 123, "Seed should match"
    assert loaded_json["info"]["a"] == 1, "Nested info should match"


def test_load_csv_roundtrip(tmp_path):
    """Test CSV load function."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    p = tmp_path / "test.csv"
    E.save_csv(df, p)

    loaded = E.load_csv(p)
    assert len(loaded) == 3, "Should load 3 rows"
    assert list(loaded.columns) == ["x", "y"], "Should have correct columns"
