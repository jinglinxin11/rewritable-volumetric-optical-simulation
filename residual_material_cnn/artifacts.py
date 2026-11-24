"""Portable compiled model serialization and illustrative layer-mask export."""

from dataclasses import asdict
import csv
import json
from pathlib import Path

import numpy as np

from .config import material_from_dict, network_from_dict
from .encoding import EncodedMatrix
from .network import CompiledCNN


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_compiled(path, model):
    path = Path(path)
    if path.suffix != '.npz' or path.exists():
        raise ValueError('Use a new .npz path; compiled artifacts are never overwritten')
    metadata = dict(schema=1, network=asdict(model.config), material=asdict(model.material),
                    convolution_bias=False, dense_bias='electronic', readout='ideal',
                    evidence='compiled layer counts, not a physical exposure recipe')
    arrays = dict(metadata=np.asarray(json.dumps(metadata)))
    for name, layer in model.layers.items():
        encoded = layer.encoded_matrix()
        arrays[name + '_counts'] = encoded.counts
        arrays[name + '_scales'] = encoded.scales
        if name != 'conv':
            arrays[name + '_bias'] = getattr(model, name + '_bias').cpu().numpy()
    np.savez_compressed(path, **arrays)


def load_compiled(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata'].item()))
        if metadata.get('schema') != 1 or metadata.get('convolution_bias') is not False:
            raise ValueError('Unsupported compiled-model schema or convolution bias')
        if metadata.get('dense_bias') != 'electronic' or metadata.get('readout') != 'ideal':
            raise ValueError('Unsupported bias or readout mode')
        network, material = network_from_dict(metadata['network']), material_from_dict(metadata['material'])
        expected = {'metadata'}
        encoded, biases = {}, {}
        for name in network.shapes:
            expected.update((name + '_counts', name + '_scales'))
            encoded[name] = EncodedMatrix(data[name + '_counts'].copy(), data[name + '_scales'].copy(), material)
            if name != 'conv':
                expected.add(name + '_bias')
                biases[name] = data[name + '_bias'].copy()
        if set(data.files) != expected:
            raise ValueError('Unexpected or missing compiled-model arrays')
    return CompiledCNN(encoded, biases, network)


def export_masks(model, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arrays, rows = {}, []
    row_offset = 0
    material = model.material
    margin = (material.sample_thickness_um - (material.depth - 1) * material.layer_spacing_um) / 2
    for name, layer in model.layers.items():
        packet = layer.encoded_matrix()
        masks = packet.write_masks()
        if not np.array_equal(masks.sum(axis=1), packet.counts):
            raise RuntimeError('Layer masks do not reproduce programmed counts')
        arrays[name] = masks.astype(np.uint8)
        for bank in range(material.residual_banks):
            for branch in range(2):
                for out in range(packet.counts.shape[2]):
                    row = row_offset + (2 * bank + branch) * packet.counts.shape[2] + out
                    for inp in range(packet.counts.shape[3]):
                        rows.append(dict(stage=name, residual_bank=bank + 1, branch='plus' if branch == 0 else 'minus',
                                         output_index=out, input_index=inp,
                                         written_layers=int(packet.counts[bank, branch, out, inp]),
                                         electronic_bank_scale=float(packet.scales[bank]),
                                         x_um=(inp + .5) * material.pitch_um,
                                         y_um=(row + .5) * material.pitch_um))
        row_offset += material.residual_banks * 2 * packet.counts.shape[2]
    np.savez_compressed(output / 'write_masks.npz', **arrays)
    with (output / 'column_counts.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    biases = {name: getattr(model, name + '_bias').cpu().tolist() for name in ('fc1', 'fc2', 'fc3')}
    save_json(output / 'electronic_biases.json', biases)
    save_json(output / 'layout.json', dict(axis_order=['residual_bank', 'depth', 'branch', 'output', 'input'],
        bit_1='written, lower transmission', layer_order_1based=[x + 1 for x in material.layer_order],
        layer_centres_um=[margin + i * material.layer_spacing_um for i in range(material.depth)],
        layout_width_um=max(x[1] for x in model.config.shapes.values()) * material.pitch_um,
        layout_height_um=row_offset * material.pitch_um, resources=model.config.resources(material),
        limitation='illustrative coordinates; no optical routing, crosstalk, or exposure calibration'))
