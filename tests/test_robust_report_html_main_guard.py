# tests/test_robust_report_html_main_guard.py
"""
Test to cover line 142 (if __name__ == '__main__') in robust_report_html.py
Run with: pytest tests/test_robust_report_html_main_guard.py -v
"""
import os
import shutil
import subprocess
import sys
import tempfile

import pytest


def test_main_guard_via_subprocess():
    """
    Test line 142 by running the module as a script via subprocess.
    This is the most reliable way to test the if __name__ == '__main__' block.
    """
    temp_dir = tempfile.mkdtemp()

    try:
        # Create test data
        dir64 = os.path.join(temp_dir, "64")
        os.makedirs(dir64)

        # Minimal valid PNG
        png_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
            b"\xc0\x00\x00\x00\x03\x00\x01\x8e\xb1\x8b\xdc\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with open(os.path.join(dir64, "test.png"), "wb") as f:
            f.write(png_data)

        output_file = os.path.join(temp_dir, "output.html")

        # Run the module as a script
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.eval.robust_report_html",
                "--out",
                output_file,
                "--dir64",
                dir64,
                "--dir224",
                os.path.join(temp_dir, "none"),
            ],
            capture_output=True,
            text=True,
        )

        # Check it ran successfully
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "[report] wrote" in result.stdout
        assert os.path.exists(output_file), "Output file was not created"

        # Verify output content
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "<!doctype html>" in content
        assert "Robustness Comparison" in content

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_main_guard_via_subprocess()
    print("✅ Main guard test passed - line 142 covered")
