# CLAUDE.md — powerflow-pipeline

## Stack
- Python 3.12 managed by `uv`; use `uv sync` and `uv run`, never `pip` or system Python.
- Core: numpy, opencv-contrib-python, av, scipy, pandas, pydantic, typer, tqdm, torch.
- Dev: pytest, pytest-cov, ruff, mypy. Ask before adding dependencies.

## Data pipeline structure


## Quality gates
TDD is mandatory: failing test first, then minimum implementation. Coverage must remain above 90%.
Verify geometry numerically with synthetic fixtures; never accept a transform merely because it ran.
CI: `ruff check` → `ruff format --check` → `mypy src` → `pytest --cov --cov-fail-under=90`.
 