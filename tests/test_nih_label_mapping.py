import numpy as np
import pandas as pd
import torch

import src.data.nih_binary as nih


def test_nih_uncertain_mapping(monkeypatch, tmp_path):
    # fake csv
    df = pd.DataFrame(
        {
            "Image": ["x1.png", "x2.png"],
            "Finding": ["Pneumonia", "Effusion"],
            "Pneumonia": [1, -1],  # second is uncertain
            "Effusion": [0, 1],
            "PatientID": [111, 222],
            "Site": ["A", "B"],
        }
    )
    csv = tmp_path / "nih.csv"
    df.to_csv(csv, index=False)

    # fake image loader returns 224x224 gray
    def _load_img(_):
        arr = (np.random.rand(224, 224) * 255).astype(np.uint8)
        return arr

    monkeypatch.setattr(nih, "_imread_gray", _load_img)

    ds = nih.NIHBinarizedDataset(
        csv_path=str(csv),
        classes=["Pneumonia", "Effusion"],
        uncertain_to=0,  # map -1 to 0
        transform=None,  # uses default in __getitem__ if any
    )

    x0, y0, meta0 = ds[0]
    x1, y1, meta1 = ds[1]

    assert y0.shape[-1] == 2
    assert (y0.numpy() == np.array([1, 0])).all()
    # uncertain -1 mapped to 0
    assert (y1.numpy() == np.array([0, 1])).all()
    assert "patient_id" in meta0 and "site" in meta0
    assert isinstance(x0, torch.Tensor)
