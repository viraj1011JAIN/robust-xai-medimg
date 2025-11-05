# tests/test_transforms_bad_args.py
import pytest

from src.data.transforms import build_transforms


def test_build_transforms_bad_domain_and_split():
    """Test that invalid domain or split raises ValueError."""
    # Test invalid domain
    with pytest.raises(ValueError, match="domain|nope|invalid"):
        build_transforms(domain="nope", split="train")

    # Test invalid split
    with pytest.raises(ValueError, match="split|nope|invalid"):
        build_transforms(domain="cxr", split="nope")


def test_build_transforms_edge_cases():
    """Test additional edge cases for transforms."""
    # Valid domains should work
    for domain in ["cxr", "derm"]:
        for split in ["train", "val", "test"]:
            tfm = build_transforms(domain=domain, split=split)
            assert tfm is not None, f"Should create transform for {domain}/{split}"
