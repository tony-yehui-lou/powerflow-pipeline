# CLAUDE.md — powerflow-pipeline

## Stack
- Python 3.12 managed by `uv`; use `uv sync` and `uv run`, never `pip` or system Python.
- Core: prefect, numpy, opencv-contrib-python, av, scipy, pandas, pydantic, typer, tqdm, torch.
- Dev: pytest, pytest-cov, ruff, mypy. Ask before adding dependencies.

## Data pipeline structure
```
data/
  cli.py                # typer entrypoint: builds config, invokes the flow
  common/               # shared @tasks and helpers: discovery, filesystem, manifest, errors
  <pipeline>/
    flow.py             # @flow, ordered orchestration only
    config.py           # pydantic config for this pipeline
    tasks/              # one @task per logical step
prefect.yaml            # deployments
```
- Each pipeline owns its folder; reusable `@task`s and safe I/O belong in `common/`. Retries and caching are `@task` arguments, never hand-rolled.
- `flow.py` wires `@task`s and nothing else. Keep pure geometry inside tasks and separate from I/O; sidecars carry `px_per_mm`, `crop_bounds`, and provenance downstream.
- Support exactly one output mode per run: `--output <folder>` or `--in-place`; also provide `--dry-run`.
- Work in staging, validate the complete result, then publish or commit in place; never leave partial scan changes.
- Every run emits a manifest as a Prefect markdown artifact: steps, file operations, derived values, warnings, rejected scans.

## Data Repository

- **Location**: All the data are located in `../data` folder. 
- **Access Rule**: All the data files are write once by one task, read many times by others
- **Source**: The source data for the entry point of all the pipeline, the raw data: `../data/raw/`
- **Output**: Each task of the pipeline will output to the folder `../data/{task}_output`. They can be the input of the next task, the next task may access other folders in `../data` too.

## Quality gates
TDD is mandatory: failing test first, then minimum implementation. Coverage must remain above 90%.
Verify geometry numerically with synthetic fixtures; never accept a transform merely because it ran.
CI: `ruff check` → `ruff format --check` → `mypy src` → `pytest --cov --cov-fail-under=90`.
