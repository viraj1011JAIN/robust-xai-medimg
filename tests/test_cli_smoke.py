# tests/test_cli_smoke.py
import subprocess
import sys


def test_cli_smoke():
    res = subprocess.run(
        [sys.executable, "-m", "src.train.baseline", "--smoke"], capture_output=True, text=True, timeout=60
    )
    assert res.returncode == 0
    assert "[SMOKE] loss=" in res.stdout
