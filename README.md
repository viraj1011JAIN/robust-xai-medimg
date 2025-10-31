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
rn

## Robustness reports
- [64×64](results/metrics/64/index.md)
- [224×224](results/metrics/224/index.md)

- [Single-file HTML robustness report](results/metrics/robust_report.html)

