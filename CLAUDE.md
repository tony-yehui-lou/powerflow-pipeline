# CLAUDE.md — powerflow-pipeline

## Stack
- Python 3.12 managed by `uv`; use `uv sync` and `uv run`, never `pip` or system Python.
- Core: numpy, opencv-contrib-python, av, scipy, pandas, pydantic, typer, tqdm, torch.
- Dev: pytest, pytest-cov, ruff, mypy. Ask before adding dependencies.

## Data pipeline structure
All data pipelines live under `src/powerflow_pipeline/data/`:
```
data/
  cli.py
  core/                 # shared context, runner, discovery, filesystem, manifest, errors
  <pipeline>/
    pipeline.py         # ordered orchestration only
    config.py           # pipeline-specific configuration
    models.py           # optional pipeline-specific contracts
    steps/              # one dedicated file per logical step
```
- Each pipeline owns its folder and step files; reusable orchestration and safe I/O belong in `core/`.
- Support exactly one output mode per run: `--output <folder>` or `--in-place`; also provide `--dry-run`.
- Work in staging, validate the complete result, then publish or commit in place; never leave partial scan changes.
- Every run writes a manifest of steps, file operations, derived values, warnings, and rejected scans.
- Keep pure geometry separate from I/O. Sidecars carry `px_per_mm`, `crop_bounds`, and provenance downstream.

## Quality gates
TDD is mandatory: failing test first, then minimum implementation. Coverage must remain above 90%.
Verify geometry numerically with synthetic fixtures; never accept a transform merely because it ran.
CI: `ruff check` → `ruff format --check` → `mypy src` → `pytest --cov --cov-fail-under=90`.
