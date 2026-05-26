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

### Phase 3 – Preprocessing Pipeline ✅
- ECG bandpass filtering (0.5–40 Hz)
- R-peak segmentation (heartbeat extraction)
- 5-class arrhythmia mapping
- Patient-wise dataset split
- Dataset preparation for CNN training
- Processed datasets saved to `data/processed/`

### Phase 4 – Model Training & Evaluation ✅
- CNN and deep ensemble checkpoints are present in `models/saved_models/`
- Ensemble evaluation results saved to `results/ensemble_results.pkl`
- Current test metrics:
	- Accuracy: `0.8892`
	- F1 score: `0.9157`

### Phase 5 – Uncertainty Quantification ✅
- Monte Carlo Dropout completed
- Predictive entropy completed
- Cluster-based entropy completed
- Results saved to `results/uncertainty_results.pkl`
- Human-readable run summary saved to `logs/run_summary_20260526_104638.txt`

### Phase 6 – Reporting ✅
- Compact report generated in `results/ensemble_report.md`
- CSV summary generated in `results/ensemble_report.csv`

### Phase 7 – Workspace Cleanup 🔄
- Recommended: keep `.venv` outside the repository root
- `.venv` relocation is pending because Windows file locks prevented moving it while Python/VS Code was active