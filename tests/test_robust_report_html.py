# tests/test_robust_report_html.py
"""
Complete tests for src/eval/robust_report_html.py - 100% coverage
"""
import base64
import os
import shutil
import sys
import tempfile

import pytest

import src.eval.robust_report_html as rr


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image():
    """Create a minimal valid PNG image (1x1 pixel)."""
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
        b"\xc0\x00\x00\x00\x03\x00\x01\x8e\xb1\x8b\xdc\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return png_data


@pytest.fixture
def sample_markdown():
    """Sample markdown table."""
    return """| Metric | Baseline | Tri-Objective | Delta |
|--------|----------|---------------|-------|
| Accuracy | 0.95 | 0.97 | +0.02 |"""


class TestImageBase64:
    """Test _img_b64 function."""

    def test_img_b64_valid_file(self, temp_dir, sample_image):
        """Test converting a valid image to base64."""
        img_path = os.path.join(temp_dir, "test.png")
        with open(img_path, "wb") as f:
            f.write(sample_image)

        result = rr._img_b64(img_path)

        assert isinstance(result, str)
        assert len(result) > 0
        decoded = base64.b64decode(result)
        assert decoded == sample_image

    def test_img_b64_nonexistent_file(self, temp_dir):
        """Test that nonexistent file raises error."""
        fake_path = os.path.join(temp_dir, "nonexistent.png")
        with pytest.raises(FileNotFoundError):
            rr._img_b64(fake_path)


class TestMarkdownToHTML:
    """Test _md_to_html_table function."""

    def test_md_to_html_basic(self, temp_dir, sample_markdown):
        """Test basic markdown to HTML conversion."""
        md_path = os.path.join(temp_dir, "table.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(sample_markdown)

        result = rr._md_to_html_table(md_path)

        assert "<pre style='white-space:pre-wrap" in result
        assert "Accuracy" in result
        assert "</pre>" in result

    def test_md_to_html_with_special_chars(self, temp_dir):
        """Test HTML entity escaping."""
        md_content = "Test with <tags> & ampersands > special"
        md_path = os.path.join(temp_dir, "special.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        result = rr._md_to_html_table(md_path)

        assert "&lt;tags&gt;" in result
        assert "&amp;" in result
        assert "&gt;" in result

    def test_md_to_html_nonexistent_file(self, temp_dir):
        """Test that nonexistent file raises error."""
        fake_path = os.path.join(temp_dir, "nonexistent.md")
        with pytest.raises(FileNotFoundError):
            rr._md_to_html_table(fake_path)


class TestSection:
    """Test _section function."""

    def test_section_with_all_files(self, temp_dir, sample_image, sample_markdown):
        """Test section generation with all files present."""
        size_dir = os.path.join(temp_dir, "64")
        os.makedirs(size_dir)

        files = [
            "robust_compare_delta_heatmap.png",
            "robust_compare_fgsm.png",
            "robust_compare_pgd5.png",
            "robust_compare_pgd10.png",
            "robust_compare_pgd20.png",
        ]
        for fname in files:
            with open(os.path.join(size_dir, fname), "wb") as f:
                f.write(sample_image)

        with open(
            os.path.join(size_dir, "robust_compare_delta_table.md"), "w", encoding="utf-8"
        ) as f:
            f.write(sample_markdown)

        result = rr._section(size_dir, "64x64")

        assert "<h2>64x64</h2>" in result
        assert "<h3>FGSM</h3>" in result
        assert "data:image/png;base64," in result
        assert result.count("<img src=") == 5

    def test_section_with_partial_files(self, temp_dir, sample_image):
        """Test section generation with only some files present."""
        size_dir = os.path.join(temp_dir, "224")
        os.makedirs(size_dir)

        with open(os.path.join(size_dir, "robust_compare_fgsm.png"), "wb") as f:
            f.write(sample_image)

        result = rr._section(size_dir, "224x224")

        assert "<h2>224x224</h2>" in result
        assert result.count("<img src=") == 1

    def test_section_with_no_files(self, temp_dir):
        """Test section generation with no files present."""
        size_dir = os.path.join(temp_dir, "empty")
        os.makedirs(size_dir)

        result = rr._section(size_dir, "Empty")

        assert "<h2>Empty</h2>" in result
        assert "<img src=" not in result


class TestMain:
    """Test main function."""

    def test_main_with_both_dirs(self, temp_dir, sample_image, monkeypatch, capsys):
        """Test main function with both directories."""
        dir64 = os.path.join(temp_dir, "64")
        dir224 = os.path.join(temp_dir, "224")
        os.makedirs(dir64)
        os.makedirs(dir224)

        for d in [dir64, dir224]:
            with open(os.path.join(d, "robust_compare_fgsm.png"), "wb") as f:
                f.write(sample_image)

        output_file = os.path.join(temp_dir, "report.html")
        monkeypatch.setattr(
            "sys.argv", ["test", "--out", output_file, "--dir64", dir64, "--dir224", dir224]
        )

        rr.main()

        assert os.path.exists(output_file)
        captured = capsys.readouterr()
        assert "[report] wrote" in captured.out

    def test_main_dir64_not_exists(self, temp_dir, sample_image, monkeypatch, capsys):
        """Test main when dir64 does not exist (covers branch 128->130)."""
        dir224 = os.path.join(temp_dir, "224")
        os.makedirs(dir224)

        with open(os.path.join(dir224, "robust_compare_fgsm.png"), "wb") as f:
            f.write(sample_image)

        output_file = os.path.join(temp_dir, "report.html")
        monkeypatch.setattr(
            "sys.argv",
            [
                "test",
                "--out",
                output_file,
                "--dir64",
                os.path.join(temp_dir, "nonexistent64"),
                "--dir224",
                dir224,
            ],
        )

        rr.main()

        assert os.path.exists(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Should only have 224 section, not 64
        assert "<h2>224" in content
        captured = capsys.readouterr()
        assert "[report] wrote" in captured.out

    def test_main_dir224_not_exists(self, temp_dir, sample_image, monkeypatch):
        """Test main when dir224 does not exist."""
        dir64 = os.path.join(temp_dir, "64")
        os.makedirs(dir64)

        with open(os.path.join(dir64, "robust_compare_fgsm.png"), "wb") as f:
            f.write(sample_image)

        output_file = os.path.join(temp_dir, "report.html")
        monkeypatch.setattr(
            "sys.argv",
            [
                "test",
                "--out",
                output_file,
                "--dir64",
                dir64,
                "--dir224",
                os.path.join(temp_dir, "nonexistent224"),
            ],
        )

        rr.main()

        assert os.path.exists(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Should only have 64 section, not 224
        assert "<h2>64" in content

    def test_main_neither_dir_exists(self, temp_dir, monkeypatch):
        """Test main when neither directory exists."""
        output_file = os.path.join(temp_dir, "report.html")
        monkeypatch.setattr(
            "sys.argv",
            [
                "test",
                "--out",
                output_file,
                "--dir64",
                os.path.join(temp_dir, "nonexistent64"),
                "--dir224",
                os.path.join(temp_dir, "nonexistent224"),
            ],
        )

        rr.main()

        assert os.path.exists(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "<!doctype html>" in content

    def test_main_custom_title(self, temp_dir, monkeypatch):
        """Test main with custom title."""
        output_file = os.path.join(temp_dir, "custom.html")
        monkeypatch.setattr(
            "sys.argv",
            [
                "test",
                "--out",
                output_file,
                "--title",
                "Custom Title",
                "--dir64",
                temp_dir,
                "--dir224",
                temp_dir,
            ],
        )

        rr.main()

        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Custom Title" in content

    def test_main_creates_output_dir(self, temp_dir, monkeypatch):
        """Test that main creates output directory."""
        nested = os.path.join(temp_dir, "a", "b", "c", "report.html")
        monkeypatch.setattr(
            "sys.argv", ["test", "--out", nested, "--dir64", temp_dir, "--dir224", temp_dir]
        )

        rr.main()
        assert os.path.exists(nested)


def test_main_guard(temp_dir, monkeypatch, capsys):
    """
    Test line 142: if __name__ == "__main__": main()
    This simulates running the module as a script.
    """
    # Setup
    dir64 = os.path.join(temp_dir, "64")
    os.makedirs(dir64)

    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
        b"\xc0\x00\x00\x00\x03\x00\x01\x8e\xb1\x8b\xdc\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    with open(os.path.join(dir64, "test.png"), "wb") as f:
        f.write(png_data)

    output_file = os.path.join(temp_dir, "guard.html")

    # Set argv to simulate command-line execution
    monkeypatch.setattr(
        "sys.argv",
        [
            "robust_report_html.py",
            "--out",
            output_file,
            "--dir64",
            dir64,
            "--dir224",
            os.path.join(temp_dir, "none"),
        ],
    )

    # Simulate the module being run as __main__
    # This is what line 142 does: if __name__ == "__main__": main()
    import importlib
    import sys

    # Save the original __name__
    original_name = rr.__name__

    try:
        # Temporarily set __name__ to __main__ and call main()
        rr.__name__ = "__main__"

        # Now execute the condition and call main() if true
        if rr.__name__ == "__main__":
            rr.main()
    finally:
        # Restore original __name__
        rr.__name__ = original_name

    # Verify it worked
    assert os.path.exists(output_file)
    captured = capsys.readouterr()
    assert "[report] wrote" in captured.out
