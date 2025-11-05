# tests/test_config_build.py
from omegaconf import OmegaConf

from src.train.baseline import build_model


def test_config_loads_and_model_builds():
    """Test that config loads and model can be built."""
    cfg = OmegaConf.load("configs/tiny.yaml")
    m = build_model(cfg.model.name, num_out=1)
    assert m is not None
    assert hasattr(m, "forward"), "Model should have forward method"
