# Virtual Environment (venv) recommendations

This project uses a Python virtual environment for development. For a clean repository and to avoid committing large environment files, follow these recommendations:

- Keep your virtual environment outside the repository root (recommended). For example, create it in a sibling folder:

  - Windows PowerShell:

    Move-Item -Path .venv -Destination ..\envs\ECG-Uncertainty-venv

  - Or create a new venv outside the repo:

    python -m venv ../ecg-venv

- If the venv is already in the repo, add `/.venv/` to `.gitignore` (this repo already ignores `.venv`).
- Stop any running processes (close VS Code or any Python sessions) before attempting to move or delete the `.venv` directory to avoid file-lock errors on Windows.
- If you want me to move the `.venv` directory for you, close editors and confirm and I'll attempt the move.

Why this helps:

- Reduces repository size and accidental commits of binary files.
- Makes backups and sharing the repository faster.
- Keeps environment management explicit (use `requirements.txt` or `pyproject.toml`).
