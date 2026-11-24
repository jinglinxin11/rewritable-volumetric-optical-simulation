import csv
import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from residual_material_cnn.artifacts import export_masks, load_compiled, save_compiled
from residual_material_cnn.data import MNISTData, parse_idx, validate_split
from residual_material_cnn.network import ReferenceCNN, compile_network, image_patches
from residual_material_cnn.training import evaluate, train_reference


def test_only_convolution_bias_is_absent():
    model = ReferenceCNN()
    assert model.layers['conv'].bias is None
    assert [model.layers[n].bias.numel() for n in ('fc1', 'fc2', 'fc3')] == [32, 16, 10]


def test_forward_shapes_and_bias_restoration():
    torch.manual_seed(71)
    ref = ReferenceCNN().double()
    compiled = compile_network(ref)
    x = torch.rand(7, 100, 25, dtype=torch.float64)
    assert ref.forward_patches(x).shape == compiled.forward_patches(x).shape == (7, 10)
    value = torch.rand(3, 27, dtype=torch.float64) * 10
    matrix = compiled.layers['fc1'].encoded_matrix().effective_weights()
    expected = value.numpy() @ matrix.T + ref.layers['fc1'].bias.detach().numpy()
    np.testing.assert_allclose(compiled.affine('fc1', value).numpy(), expected, atol=1e-10, rtol=1e-10)
    assert not list(compiled.parameters())


def test_image_preprocessing_has_100_valid_windows():
    assert image_patches(torch.zeros(4, 1, 14, 14)).shape == (4, 100, 25)
    with pytest.raises(ValueError):
        image_patches(torch.zeros(4, 1, 28, 28))
    with pytest.raises(ValueError):
        image_patches(torch.full((4, 1, 14, 14), 2.))


def test_compiled_serialization_does_not_store_float_weight_matrices(tmp_path):
    torch.manual_seed(17)
    compiled = compile_network(ReferenceCNN())
    path = tmp_path / 'model.npz'
    save_compiled(path, compiled)
    with np.load(path, allow_pickle=False) as file:
        assert not any('weight' in name for name in file.files)
    restored = load_compiled(path)
    x = torch.rand(3, 100, 25)
    torch.testing.assert_close(compiled.forward_patches(x), restored.forward_patches(x), rtol=0., atol=0.)
    with pytest.raises(ValueError):
        save_compiled(path, compiled)


def test_write_export_is_complete_and_does_not_overwrite(tmp_path):
    model = compile_network(ReferenceCNN())
    output = tmp_path / 'masks'
    export_masks(model, output)
    with np.load(output / 'write_masks.npz', allow_pickle=False) as masks:
        assert sum(masks[name].size for name in masks.files) == 96660
        for name in masks.files:
            np.testing.assert_array_equal(masks[name].sum(axis=1), model.layers[name].counts.numpy())
    with (output / 'column_counts.csv').open(encoding='utf-8', newline='') as file:
        assert len(list(csv.DictReader(file))) == 9666
    layout = json.loads((output / 'layout.json').read_text())
    assert layout['layout_height_um'] == 18300
    assert layout['layer_centres_um'] == list(range(200, 4000, 400))
    with pytest.raises(FileExistsError):
        export_masks(model, output)


def test_local_idx_parser_rejects_truncation():
    payload = gzip.compress(struct.pack('>II', 2049, 2) + bytes([3, 7]))
    np.testing.assert_array_equal(parse_idx(payload), [3, 7])
    with pytest.raises(ValueError):
        parse_idx(gzip.compress(struct.pack('>II', 2049, 2) + bytes([3])))


def test_split_overlap_rejected():
    validate_split(np.arange(50000), np.arange(50000, 60000))
    with pytest.raises(ValueError):
        validate_split(np.arange(50000), np.arange(10000))


def test_bundled_split_is_complete_and_disjoint():
    path = Path(__file__).resolve().parents[1] / 'assets/split_indices.npz'
    with np.load(path, allow_pickle=False) as split:
        validate_split(split['train'], split['validation'])


def test_training_compile_and_reload_smoke(tmp_path):
    g = torch.Generator().manual_seed(9)
    def part(n):
        return torch.rand(n, 100, 25, generator=g), torch.randint(10, (n,), generator=g)
    data = MNISTData(part(24), part(12), part(12), {'source': 'synthetic unit test'})
    ref, record = train_reference(data, 2, tmp_path, epochs=1, batch_size=8)
    assert record['selected_epoch'] in (0, 1)
    assert (tmp_path / 'reference.pt').is_file()
    compiled = compile_network(ref)
    save_compiled(tmp_path / 'compiled.npz', compiled)
    result = evaluate(load_compiled(tmp_path / 'compiled.npz'), *data.test)
    assert 0 <= result['accuracy_pct'] <= 100
