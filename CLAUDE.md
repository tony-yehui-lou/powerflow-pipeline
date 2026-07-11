# CLAUDE.md — powerflow-pipeline

The projects will have 2 different types of pipleine
 - Data pipeline: Preprocssing and PostProcsssing
 - Training pipeline: Train or finetune models when needed

## Stack
- **Python 3.12**, managed by **uv**. `uv sync` to install, `uv run <cmd>` to run. Never invoke `pip` or system Python.
- **Core:** numpy · opencv-contrib-python (must be `-contrib`: `cv2.aruco` is not in plain opencv) · av (frame-accurate timestamps; OpenCV's are unreliable) · scipy · pandas · pydantic · typer · tqdm · torch.
- **Dev:** pytest · pytest-cov · ruff · mypy.
- No ML model libraries yet (no Lightning / timm / ultralytics). Ask before adding any dependency.

## Structure
```
src/powerflow/
  common/     config.py · sidecar.py            # pydantic; shared by both pipelines
  data/       io/ · geometry/ · stages/ (s0…s6) · calib/ · preflight/ · validate/
  training/   datasets.py · models/ · train.py  # stays empty until the data pipeline lands
configs/      default.yaml                      # every threshold lives here, not in literals
tests/        synthetic/ · unit/ · integration/
data/         raw/ · interim/ · processed/      # gitignored
```
`data/geometry/` is pure `array → array` with no I/O; `data/stages/` is thin (load → call geometry → gate → write sidecar).
The sidecar is the contract between the two pipelines: without `px_per_mm` and `crop_bounds`, a model prediction cannot be converted back into metres.

## TDD is mandatory
Write a failing test → run it, confirm it fails for the right reason → write the minimum code to pass → refactor.
Never write implementation before its test. Never claim work is done without pasting passing `pytest` output.
**Coverage must stay >90%**, enforced by `--cov-fail-under=90` in `pyproject.toml`.
Geometry is verified against synthetic fixtures with known ground truth (`tests/synthetic/`): a wrongly-tilted or
mirrored image looks completely fine as an array — assert on numbers, never assume a transform is correct because it ran.

## CI (GitHub Actions, on push + PR)
`ruff check` → `ruff format --check` → `mypy src` → `pytest --cov --cov-fail-under=90`. All must pass to merge.
No CD — this is a local batch pipeline with no deployment target.
