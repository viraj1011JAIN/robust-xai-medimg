import subprocess
import sys


def test_eval_cli_writes_csv(tmp_path):
    out_csv = tmp_path / "eval_out.csv"
    cmd = [
        sys.executable,
        "-m",
        "src.train.evaluate",
        "--config",
        "configs/tiny.yaml",
        "--dry-run",
        "--out",
        str(out_csv),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0
    assert out_csv.exists()
    txt = out_csv.read_text().strip()
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    # Expect header + at least one row
    assert len(lines) >= 2
    assert lines[0].split(",") == ["loss", "auroc"]
