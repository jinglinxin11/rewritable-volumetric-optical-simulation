"""Packaged calibration split and example models, independent of the working directory."""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path


@contextmanager
def bundled_split() -> Iterator[Path]:
    """Expose the fixed, packaged MNIST split for the duration of a read."""
    with as_file(files("residual_material_cnn").joinpath("assets", "split_indices.npz")) as path:
        yield path


@contextmanager
def bundled_model(seed: int) -> Iterator[Path]:
    """Expose one of the three frozen compiled models without downloading anything."""
    if seed not in (0, 1, 2):
        raise ValueError("The packaged example seeds are 0, 1, and 2")
    resource = files("residual_material_cnn").joinpath(
        "assets", "models", f"seed{seed}_compiled.npz"
    )
    with as_file(resource) as path:
        yield path
