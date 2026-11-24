import json

import numpy as np
import pytest

from residual_material_cnn import artifacts
from residual_material_cnn.resources import bundled_model


def test_json_rejects_nan_without_damaging_previous_record(tmp_path):
    path = tmp_path / "record.json"
    artifacts.save_json(path, {"complete": True})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        artifacts.save_json(path, {"invalid": float("nan")})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_failed_atomic_replace_cleans_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    artifacts.save_json(path, {"old": True})

    def fail_replace(*args):
        raise PermissionError("injected replacement failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="injected"):
        artifacts.save_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "change",
    ["missing_metadata", "metadata_type", "schema", "bias", "missing_array", "extra_array"],
)
def test_malformed_compiled_model_is_rejected(tmp_path, change):
    with bundled_model(0) as path, np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    if change == "missing_metadata":
        del arrays["metadata"]
    elif change == "metadata_type":
        arrays["metadata"] = np.asarray("[]")
    elif change in ("schema", "bias"):
        metadata["schema" if change == "schema" else "dense_bias"] = "unsupported"
        arrays["metadata"] = np.asarray(json.dumps(metadata))
    elif change == "missing_array":
        del arrays["conv_counts"]
    else:
        arrays["unrecognized"] = np.zeros(1)
    target = tmp_path / "bad.npz"
    np.savez_compressed(target, **arrays)
    with pytest.raises(ValueError):
        artifacts.load_compiled(target)
