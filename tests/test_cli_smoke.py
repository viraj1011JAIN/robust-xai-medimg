# tests/test_cli_smoke.py
import subprocess
import sys


def test_cli_smoke():
    """Test that baseline training runs in smoke mode without errors."""
    res = subprocess.run(
        [sys.executable, "-m", "src.train.baseline", "--smoke"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, f"CLI failed with stderr: {res.stderr}"
    assert "[SMOKE] loss=" in res.stdout, "Expected smoke test output not found"
