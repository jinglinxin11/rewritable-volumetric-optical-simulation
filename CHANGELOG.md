# Changelog

## 0.2.0 - 2026-11-24

- Package the independent residual CNN under `src/` with `pyproject.toml` and
  the `residual-material-cnn` console entry point.
- Separate CLI, package resources and shared preprocessing from numerical code.
- Add validated training configuration, stderr logging and atomic JSON records.
- Separate unit/integration tests; configure Ruff, branch coverage, wheel smoke
  checks and a Linux/Windows CI matrix.
- Keep the numerical algorithm, saved model bytes, schema-1 NPZ format and
  historical manuscript scripts unchanged.

### Migration

Install with `python -m pip install -e .` before running the module. Development
tools use `python -m pip install -e ".[dev,legacy]"`. The former package-local
requirements file is replaced by project metadata; root legacy requirements
remain available for manuscript workflows.

The Chinese guide is now `docs/residual_material_cnn/README_CN.md`; recorded
reports are `examples/residual_material_cnn/`. Use `--example-seed 0` instead of
the old `--model residual_material_cnn/examples/seed0_compiled.npz` path.
External `--model` paths remain supported without model conversion.

The data default is `data/MNIST/raw` relative to the caller's working directory;
use an absolute `--data` path outside the repository. Packaged models and fixed
split indices no longer depend on the caller's working directory.
