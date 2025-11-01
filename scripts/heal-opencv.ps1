$ErrorActionPreference = "Stop"
Write-Host "Enforcing headless OpenCV under constraints..."
pip uninstall -y opencv-python | Out-Null
pip install -c ".\constraints.txt" --force-reinstall opencv-python-headless==4.10.0.84
python - <<'PY'
import numpy, scipy, cv2
print("NumPy", numpy.__version__, "SciPy", scipy.__version__, "OpenCV OK:", hasattr(cv2, "getBuildInformation"))
PY
