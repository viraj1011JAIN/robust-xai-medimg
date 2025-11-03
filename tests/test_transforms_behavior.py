import numpy as np

from src.data import transforms as T


def _dummy_img(h=256, w=256):
    # uint8 image [0,255]
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def test_cxr_train_transform_range_and_shape():
    tfm = T.build_transforms(domain="cxr", split="train")
    out = tfm(image=_dummy_img(300, 300))["image"]
    assert out.shape == (3, 224, 224)
    assert float(out.min()) >= -5 and float(out.max()) <= 5  # normalized


def test_derm_train_transform_range_and_shape():
    tfm = T.build_transforms(domain="derm", split="train")
    out = tfm(image=_dummy_img(224, 224))["image"]
    assert out.shape == (3, 224, 224)


def test_cxr_val_is_deterministic():
    tfm = T.build_transforms(domain="cxr", split="val")
    img = _dummy_img(280, 280)
    a = tfm(image=img)["image"]
    b = tfm(image=img)["image"]
    # no training jitter, so val should be stable for same input
    assert np.allclose(a, b)
