> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

# Relevant commands

* To rerun the CPU-only LSH experiment, use the notebook command under **LSH retrieval benchmark** in `README.md`. Its dependency list is `notebooks/requirements-lsh.txt`.

* The project uses `uv` for management of virtual environments. This means:
  * To install packages, use `uv add <package-name>`.
  * To run Python scripts, use `uv run <script-name>.py`.
  * To run other commands related to Python, prefix them with `uv run `, e.g., `uv run <command>`.
* The project uses `pytest` for testing. To run tests, use `uv run pytest tests/`.
* The project uses `ruff` for linting and formatting:
    * To format code, use `uv run ruff format .`.
    * To lint code, use `uv run ruff check . --fix`.
* To download and process the dataset, use `uv run python -m afp.data --subset small [--limit N]`
    (or `uv run invoke data`). It is resumable; do not run it unless asked, it downloads about 8 GB (`small`).
* To make fingerprints, use `uv run python -m afp.fingerprint cache|train|encode` (see `README.md`). `cache` downloads
    MuQ weights (about 1.2 GB) on first use and runs the model over about 62,000 windows for `small`; do not run it
    unless asked.
* The project uses `invoke` for task management (`test`, `data`, `fmt`). To see available tasks, use
    `uv run invoke --list` or refer to the `tasks.py` file.
* The project uses `pre-commit` for managing pre-commit hooks. To run all hooks on all files, use
    `uv run pre-commit run --all-files`. For more information, refer to the `.pre-commit-config.yaml` file.

# Layout

* Source lives in `src/afp/` (`afp` is a placeholder package name). See `README.md` for the module map.
* Tests must not need audio data or a downloaded model.
* Audio, cached embeddings and fingerprints live under `data/` and are git-ignored.

# Code style

* Follow existing code style.
* Keep line length within 120 characters.
* Use f-strings for formatting.
* Use type hints
* Do not add inline comments unless absolutely necessary.

# Documentation

* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.
