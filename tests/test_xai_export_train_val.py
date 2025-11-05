import os

import pytest
import torch


# Skip locally if the tiny config isn't present; will run in CI if available
@pytest.mark.skipif(
    not os.path.exists("configs/test_phase1.yaml"),
    reason="Local test config not available",
)
def test_load_one_from_cfg_train_and_val():
    from src.xai.export import _load_one_from_cfg

    cfg_path = "configs/test_phase1.yaml"

    x_train = _load_one_from_cfg(cfg_path, split="train")
    assert isinstance(x_train, torch.Tensor)
    assert x_train.ndim == 4 and x_train.shape[0] == 1

    x_val = _load_one_from_cfg(cfg_path, split="val")
    assert isinstance(x_val, torch.Tensor)
    assert x_val.ndim == 4 and x_val.shape[0] == 1
