# scripts/heal-opencv.ps1
$ErrorActionPreference = "Stop"
Write-Host "Enforcing headless OpenCV under constraints..."

# Always remove GUI variants if present
pip uninstall -y opencv-python opencv-contrib-python | Out-Null

# Reinstall headless under our constraints
pip install -c ".\constraints.txt" --force-reinstall opencv-python-headless==4.10.0.84

# Reinstall grad-cam WITHOUT deps so it doesn't try to pull GUI OpenCV
pip install --no-deps "git+https://github.com/jacobgil/pytorch-grad-cam.git"

# PowerShell-safe embedded Python (use a temp .py file)
$code = @"
from importlib.metadata import version, PackageNotFoundError
import numpy, scipy, cv2

def has_dist(name: str) -> bool:
    try:
        version(name); return True
    except PackageNotFoundError:
        return False

is_headless = has_dist("opencv-python-headless") \
    and not has_dist("opencv-python") \
    and not has_dist("opencv-contrib-python")

print("NumPy", numpy.__version__, "SciPy", scipy.__version__)
print("cv2 module path:", cv2.__file__)
print("installed dists:",
      "opencv-python-headless" if has_dist("opencv-python-headless") else "-",
      "opencv-python" if has_dist("opencv-python") else "-",
      "opencv-contrib-python" if has_dist("opencv-contrib-python") else "-")
print("OpenCV headless (by dist metadata):", is_headless)
"@

$tmp = [System.IO.Path]::GetTempFileName() + '.py'
Set-Content -LiteralPath $tmp -Value $code -Encoding UTF8
python $tmp
Remove-Item $tmp -Force
