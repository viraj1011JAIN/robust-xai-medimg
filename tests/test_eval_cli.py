import subprocess
import sys


def test_eval_cli_dry_run():
    cmd = [sys.executable, "-m", "src.train.evaluate", "--config", "configs/tiny.yaml", "--dry-run"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0
    # Should print the summary line once it finishes a batch
    assert "[EVAL] loss=" in res.stdout
