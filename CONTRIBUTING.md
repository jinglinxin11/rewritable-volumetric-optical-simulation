# Contributing

## Scope and compatibility

The installable residual experiment lives in `src/residual_material_cnn/`.
Root `scheme_a_*` modules, archived manuscript artifacts and historical tests
remain source-compatible; do not format or migrate them as incidental cleanup.
Do not replace formal experiment parameters or overwrite past result directories.

Preserve schema-1 compiled models, fixed data splits, residual encoding and
electronic dense biases unless a separately documented scientific change is
intended. Numerical changes need paired seeded results, not only a passing test.
Keep encoding/readout independent of CLI and filesystem I/O. Validate external
data at boundaries and keep progress logs off JSON stdout.

## Local checks

Use Python 3.10 or newer in an isolated environment:

```bash
python -m pip install -e ".[dev,legacy]"
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=residual_material_cnn --cov-report=term-missing
python -m build
python tools/check_wheel.py dist/rewritable_volumetric_residual_cnn-0.2.0-py3-none-any.whl
```

`legacy` supplies dependencies for the historical tests. To test only the new
package, install `.[dev]` and run `pytest tests/unit tests/integration`. Default
tests do not download MNIST or require a GPU. The package coverage floor is 85%
including branches; assertions and boundary cases matter more than the number.

Ruff checks Python syntax, imports and selected bug patterns with a 100-character
formatter width. Its configured scope excludes the archived manuscript scripts.
Use `ruff format .` to apply formatting. Add deterministic unit tests for math
and integration tests for files, CLI, package resources and workflows. Review
the final diff for accidental artifacts, stale paths and compatibility changes.

CI runs Python 3.10/3.12 on Linux and 3.11 on Windows, tests all preserved and new
tests, builds wheel/sdist, and installs the wheel outside the checkout. The wheel
contains only the new package and its small NPZ resources, not MNIST or legacy
scripts. GitHub Actions is a quality gate, not a hardware validation.

## Packaging references

The project follows [PyPA project metadata](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
and [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/),
uses [Ruff configuration](https://docs.astral.sh/ruff/configuration/), and follows
[GitHub's Python build/test workflow](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).
This is a maintainable research package, not a claim of certification against a
formal software standard.
