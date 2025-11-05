import csv
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from PIL import Image

from src.xai.export import _load_one_from_cfg


def _write_csv_and_img(tmp_path: Path, csv_name: str) -> str:
    imgs = tmp_path / "imgs"
    imgs.mkdir(exist_ok=True)
    Image.new("RGB", (3, 3)).save(imgs / "a.png")
    csv_path = tmp_path / csv_name
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerow({"image_path": "imgs/a.png", "label": "1"})
    return str(csv_path)


def test_load_one_from_cfg_fallback_to_val_when_train_missing(tmp_path):
    val_csv = _write_csv_and_img(tmp_path, "val.csv")
    cfg = {"data": {"img_size": 8, "val_csv": val_csv}}
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(OmegaConf.create(cfg), cfg_path)

    # Ask for 'train' – code should fall back to val_csv branch
    x = _load_one_from_cfg(str(cfg_path), split="train")
    assert x.shape[0] == 1 and x.shape[1] == 3


def test_load_one_from_cfg_raises_when_no_csv_keys(tmp_path):
    cfg = {"data": {"img_size": 8}}
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(OmegaConf.create(cfg), cfg_path)

    with pytest.raises(RuntimeError, match="missing.*train_csv.*val_csv|Missing"):
        _load_one_from_cfg(str(cfg_path), split="val")
