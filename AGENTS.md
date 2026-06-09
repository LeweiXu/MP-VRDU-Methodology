# Repository Guidelines

## Project Structure & Module Organization

`mpvrdu/` contains the Python package. The pipeline entry point is
`mpvrdu/pipeline.py`; supporting stages are grouped into `data/`, `represent/`,
`retrieve/`, `generate/`, `eval/`, and `analysis/`. Keep reusable logic in these
modules and use `scripts/` for command-line orchestration and inspection tools.

Experiment definitions live in `configs/` (single runs and component variants)
and `experiments/` (grid suites). Tests are under `tests/` and mirror package
features with files such as `test_retrievers.py`. Documentation belongs in
`docs/`. Treat `data/`, `.cache/`, `results/`, `fig/`, and `logs/` as generated
or local artifacts; do not commit large datasets, model weights, or run outputs.

## Build, Test, and Development Commands

Use Python 3.10 or newer. Install the CPU development stack with:

```bash
pip install -r requirements-light.txt
```

Install `requirements-gpu.txt` only for real VLM, dense, or visual retrieval
runs. Common commands are:

```bash
make test       # run the offline pytest suite
make smoke      # execute the synthetic end-to-end pipeline
make grid-dry   # preview all local experiment conditions
make report     # regenerate tables and figures from results/grid
```

Run one condition with
`python -m mpvrdu.pipeline --config configs/smoke.yaml`. Use `make grid
MAXQ=4` for a short grid machinery check.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, type hints for public
interfaces, concise docstrings, and imports grouped as standard library,
third-party, then local modules. Use `snake_case` for functions, variables,
modules, and YAML fields; `PascalCase` for classes; and `UPPER_CASE` for
constants. Prefer `pathlib.Path`, dataclasses, and config-driven behavior over
hard-coded experiment branches. No formatter or linter is currently enforced,
so keep changes consistent with nearby code.

## Testing Guidelines

Tests use `pytest`, configured in `pyproject.toml`. Name files `test_*.py` and
test functions `test_*`. Keep default tests offline and deterministic by using
the synthetic dataset fixtures in `tests/conftest.py`; optional dependencies
should be guarded with `pytest.importorskip`. Add focused regression tests for
config validation, retrieval behavior, result schemas, and pipeline changes.

## Commit & Pull Request Guidelines

History uses short, imperative subjects such as `add context.md`; keep commits
focused and avoid bundling generated outputs. Pull requests should explain the
behavioral change, name affected configs or sub-studies, list verification
commands, and link relevant issues. Include sample metrics or screenshots only
when analysis output or rendered evidence changes.
