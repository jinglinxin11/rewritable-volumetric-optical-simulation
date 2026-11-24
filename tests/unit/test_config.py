import math

import pytest

from residual_material_cnn.config import (
    MaterialConfig,
    NetworkConfig,
    TrainingConfig,
    configuration,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(pitch_um=math.nan),
        dict(layer_spacing_um=math.inf),
        dict(sample_thickness_um=3000),
        dict(depth=10.0),
        dict(layer_order=(False, 1, 2, 3, 4, 5, 6, 7, 8, 9)),
    ],
)
def test_geometry_and_count_validation(kwargs):
    with pytest.raises(ValueError):
        MaterialConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(epochs=0),
        dict(epochs=1.5),
        dict(batch_size=0),
        dict(learning_rate=math.nan),
        dict(seeds=()),
        dict(seeds=(0, 0)),
        dict(seeds=(-1,)),
        dict(seeds=(True,)),
    ],
)
def test_training_validation(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_unsupported_network_rejected():
    with pytest.raises(ValueError):
        NetworkConfig(channels=12)
    assert configuration()["resources"]["binary_sites"] == 96660
