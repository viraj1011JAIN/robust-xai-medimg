# Robust XAI for Medical Imaging — Debug & Baseline

![CI](https://github.com/viraj1011JAIN/robust-xai-medimg/actions/workflows/ci.yml/badge.svg)

This repo contains a minimal, exam-ready pipeline to (a) run fast smoke checks, (b) do a CPU-only sanity training pass on a toy dataset, and (c) keep CI green. It also includes debug notes for robustness sweeps and figure outputs you can drop into the thesis.

---

## Quick start (no data, <10s)

```bash
# 1-batch synthetic train step; proves the training loop compiles & runs
python -m src.train.baseline --smoke

# Windows PowerShell
python -m venv .venv && . .venv/Scripts/activate
python -m pip install -U pip wheel
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 torchvision==0.24.0
pip install -r requirements.txt

# 12-image toy dataset + 1-epoch CPU config
python -m src.train.baseline --config configs/tiny.yaml
# or positional form:
python -m src.train.baseline configs/tiny.yaml
# or just the synthetic smoke path again:
python -m src.train.baseline --smoke


---

# 2) Run the debug robustness sweeps

> These commands assume you’re still in the repo root and that `tools\run_debug_sweeps.ps1` exists.

```powershell
# Allow scripts for this PowerShell process only (safe/temporary)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# (Optional) activate your Python/conda venv if your script uses it
# .\.venv\Scripts\Activate.ps1
# or: conda activate robust-xai

# Quick start: uses the newest checkpoint it finds in results\checkpoints
powershell -ExecutionPolicy Bypass -File tools\run_debug_sweeps.ps1

