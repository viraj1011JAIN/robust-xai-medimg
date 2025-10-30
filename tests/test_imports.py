import importlib

import pytest


@pytest.mark.parametrize(
    "mod",
    [
        "src.train.baseline",
        "src.train.evaluate",
        "src.train.triobj_training",
        "src.xai.gradcam",
        "src.attacks.pgd",
    ],
)
def test_imports(mod):
    importlib.import_module(mod)
