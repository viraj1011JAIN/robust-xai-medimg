# AI Agent Instructions for robust-xai-medimg

This project focuses on robust explainable AI (XAI) for medical imaging, particularly chest X-rays and dermatology images. Here's what you need to know to be productive:

## Project Architecture

- **Core Components**:
  - `src/data/`: Dataset loaders (e.g. `nih_binary.py` for chest X-rays)
  - `src/models/`: Model architectures
  - `src/train/`: Training loops and utilities
  - `src/xai/`: Explainability methods
  - `configs/`: YAML configuration files

## Key Patterns & Conventions

1. **Data Loading**:
   - Uses CSV files with `filepath` column and target labels
   - Images are loaded via `CSVImageDataset` which handles both absolute and relative paths
   - Standard ImageNet normalization is applied (see `IMAGENET_MEAN`/`IMAGENET_STD` in `src/data/nih_binary.py`)

2. **Configuration**:
   - All hyperparameters/settings in YAML files under `configs/`
   - Uses OmegaConf for config management
   - See `configs/base.yaml` for template structure

3. **Training Workflow**:
   - Models use PyTorch with mixed precision training support
   - TensorBoard logging for metrics
   - Train/val split handled by separate CSV files
   - Example: `src/train/baseline.py` shows standard training loop

## Development Setup

1. **Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Setup**:
   ```bash
   python smoke.py  # Checks PyTorch/CUDA availability
   ```

3. **Data**:
   - Expected at path specified in `configs/base.yaml` under `data.root`
   - Each dataset needs train/val CSVs with `filepath` and label columns

## Common Operations

- Run training: `python -m src.train.baseline configs/base.yaml`
- Monitor training: TensorBoard logs in `results/runs/`
- Model outputs: Binary classification with BCEWithLogitsLoss