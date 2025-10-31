from __future__ import annotations

import os
import platform
import random

import numpy as np
import torch

_DEFAULT_SEED = 42


def set_seed(seed: int = _DEFAULT_SEED) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return seed


def get_reproducibility_info() -> dict:
    return {
        "seed": _DEFAULT_SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "devices": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algs": True,
    }
