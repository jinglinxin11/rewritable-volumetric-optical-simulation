import numpy as np
import pytest
import torch

from residual_material_cnn.config import MaterialConfig, NetworkConfig
from residual_material_cnn.encoding import EncodedMatrix, build_codebook, encode_matrix, nearest_indices
from residual_material_cnn.metrics import error_metrics, legacy_mac_benchmark
from residual_material_cnn.readout import IdealResidualReadout


def test_codebook_has_111_signed_levels():
    book = build_codebook(MaterialConfig())
    assert len(book.values) == 111
    np.testing.assert_array_equal(book.values, -book.values[::-1])
    assert book.values[0] == -1 and book.values[-1] == 1
    assert book.alpha == pytest.approx(.033045674863435, abs=1e-14)


def test_tie_uses_lower_level():
    assert nearest_indices(np.array([0.]), np.array([-1., 1.])).item() == 0


def test_residual_example_and_endpoints():
    weight = np.array([[.25, -.25, 0., 1., -1.]])
    encoded = encode_matrix(weight)
    np.testing.assert_allclose(encoded.effective_weights()[0, :2], [.2500067044998698, -.2500067044998698], atol=1e-12)
    np.testing.assert_allclose(encoded.effective_weights()[0, 2:], [0., 1., -1.], atol=1e-12)
    assert np.all(encoded.counts[:, :, 0, 2] == 10)


def test_masks_preserve_counts_and_use_fixed_layer_order():
    packet = encode_matrix(np.array([[.25]]))
    mask = packet.write_masks()
    assert mask.shape == (3, 10, 2, 1, 1)
    np.testing.assert_array_equal(mask.sum(axis=1), packet.counts)
    assert set(np.flatnonzero(mask[0, :, 0, 0, 0])) == {2, 3, 4, 5, 6}


@pytest.mark.parametrize('weight', [np.array([[1.01]]), np.array([[np.nan]]), np.array([1.]), np.empty((0, 2))])
def test_invalid_weights_are_not_silently_compiled(weight):
    with pytest.raises(ValueError):
        encode_matrix(weight)


@pytest.mark.parametrize('counts', [np.full((3, 2, 1, 1), 11), np.zeros((3, 2, 1, 1), dtype=float),
                                  np.zeros((2, 2, 1, 1), dtype=int)])
def test_invalid_layer_counts_rejected(counts):
    with pytest.raises(ValueError):
        EncodedMatrix(counts, np.ones(3), MaterialConfig())


def test_physical_readout_matches_decoded_matrix_and_restores_input_scale():
    rng = np.random.default_rng(51)
    packet = encode_matrix(rng.uniform(-1, 1, (4, 7)))
    x = torch.from_numpy(rng.random((2, 3, 7)) * 8)
    actual = IdealResidualReadout(packet)(x).numpy()
    np.testing.assert_allclose(actual, x.numpy() @ packet.effective_weights().T, rtol=1e-10, atol=1e-12)


def test_negative_optical_input_rejected():
    with pytest.raises(ValueError):
        IdealResidualReadout(encode_matrix(np.zeros((1, 1))))(torch.tensor([[-1.]]))


def test_historical_clean_mac_benchmark_replays():
    result = legacy_mac_benchmark()
    assert result['equivalent_bits'] == pytest.approx(18.549528231629655, abs=1e-7)


def test_resource_count_includes_three_pairs_per_weight():
    result = NetworkConfig().resources(MaterialConfig())
    assert result['signed_weights'] == 1611
    assert result['binary_sites'] == 96660
    assert result['electronic_biases'] == 58
    assert result['residual_matrix_banks'] == 12


@pytest.mark.parametrize('kwargs', [dict(depth=20), dict(site_transmission=1.), dict(layer_order=(0,) * 10)])
def test_unsupported_material_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        MaterialConfig(**kwargs)


def test_perfect_metric_is_json_safe():
    result = error_metrics(np.zeros(3), np.zeros(3))
    assert result['exact'] and result['equivalent_bits'] is None
