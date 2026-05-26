# ECG-Uncertainty-Detection

Uncertainty-Aware ECG Arrhythmia Detection using Deep Ensembles and Entropy-Based Calibration for Clinical Decision Support.

## Project Overview

This project aims to improve trust in AI-based ECG classification systems by integrating uncertainty estimation methods including:

- Deep Ensembles
- Monte Carlo Dropout
- Predictive Entropy
- Cluster-Based Entropy (Novel Approach)

The goal is to flag uncertain predictions for human review, enhancing clinical decision support reliability.

## Project Structure

- `src/` – Core implementation
- `scripts/` – Data processing and training utilities
- `configs/` – Experiment configurations
- `results/` – Evaluation outputs (ignored in Git)
- `data/` – ECG datasets (ignored in Git)

## Local environment (recommended)

This repository uses a Python virtual environment for development. To keep the repository small and reproducible:

- Create your virtual environment outside the repository root (for example `../ecg-venv`).
- Ensure `/.venv/` or `venv/` is listed in `.gitignore` (already present).
- Close VS Code and any running Python processes before moving or deleting the `.venv` folder on Windows to avoid file-lock errors.
- See `docs/venv.md` for quick commands and guidance.

## Status

### Phase 1 – Environment Setup ✅
- Python virtual environment
- Required libraries installed
- Project structure initialized

### Phase 2 – Data Acquisition & Exploration ✅
- MIT-BIH dataset downloaded
- ECG records loaded successfully
- Data exploration notebook created
- ECG waveform visualization completed
- Annotation distribution analyzed

### Phase 3 – Preprocessing Pipeline 🔄 (Next)
Planned components:

- ECG bandpass filtering (0.5–40 Hz)
- R-peak segmentation (heartbeat extraction)
- 5-class arrhythmia mapping
- Patient-wise dataset split
- Dataset preparation for CNN training