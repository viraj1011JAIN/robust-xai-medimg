from typing import Any, List, Tuple

from torch.utils.data import Dataset


class CSVImageDataset(Dataset):
    """
    Minimal stub so training modules can import without requiring real data code.
    Replace later with the real implementation.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._items: List[Tuple[Any, Any]] = []

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        raise RuntimeError("CSVImageDataset stub — real implementation not added yet.")
