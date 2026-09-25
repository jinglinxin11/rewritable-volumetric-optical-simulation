"""Three material classifier layers; independent-channel, ideal-device simulation.

The convolution is unchanged. Every dense weight uses differential binary depth
columns. Electronic ReLU and intensity re-encoding separate the dense layers.
"""
import argparse
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import scheme_a_material_head as base

ROOT = Path(__file__).resolve().parent
WIDTHS = (27, 32, 16, 10)
ZONES = ('conv', 'fc1', 'fc2', 'fc3')


def shapes(config):
    result = {'conv': (config.layers, 2, 3, 9)}
    result.update({f'fc{i+1}': (config.layers, 2, b, a)
                   for i, (a, b) in enumerate(zip(WIDTHS, WIDTHS[1:]))})
    return result


class Model(nn.Module):
    def __init__(self, config, seed):
        super().__init__()
        self.config = config
        for zone, offset in zip(ZONES, (1000, 4000, 5000, 6000)):
            generator = torch.Generator().manual_seed(seed + offset)
            setattr(self, zone + '_theta', nn.Parameter(
                .5 * torch.randn(shapes(config)[zone], generator=generator)))
        self.conv_bias = nn.Parameter(torch.zeros(3))
        # Fan-in uniform biases only; no electronic dense weight matrices.
        for i, (a, b) in enumerate(zip(WIDTHS, WIDTHS[1:])):
            generator = torch.Generator().manual_seed(seed + 7000 + i)
            bias = (2 * torch.rand(b, generator=generator) - 1) / a**.5
            setattr(self, f'fc{i+1}_bias', nn.Parameter(bias))

    def bits(self):
        return {z: (getattr(self, z + '_theta').detach() >= 0).float() for z in ZONES}

    def weights(self):
        return {z: base.optical_weights(base.prior.binary_ste(getattr(self, z + '_theta')),
                                        self.config) for z in ZONES}

    def logits_from_weights(self, x, weights):
        h = base.feature_vector(x, weights['conv'], self.conv_bias)
        for i in range(1, 4):
            h = base.material_classifier(h, weights[f'fc{i}'], getattr(self, f'fc{i}_bias'))
            if i < 3:
                h = F.relu(h)
        return h

    def forward(self, x):
        return self.logits_from_weights(x, self.weights())


class Device(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        for zone, shape in shapes(config).items():
            self.register_buffer(zone + '_state', torch.zeros(shape))
        self.initialized = False
        self.requests = self.erase_events = self.skipped_identical = 0
        self.write_ones = dict.fromkeys(ZONES, 0)
        self.last_written = dict.fromkeys(ZONES, 0)

    @torch.no_grad()
    def program(self, target):
        if set(target) != set(ZONES):
            raise ValueError('All four zones required for atomic whole-volume rewrite')
        for z, shape in shapes(self.config).items():
            if tuple(target[z].shape) != shape or not bool(((target[z] == 0) | (target[z] == 1)).all()):
                raise ValueError('Exact binary masks with correct shapes required')
        self.requests += 1
        if self.initialized and all(torch.equal(target[z], getattr(self, z + '_state')) for z in ZONES):
            self.skipped_identical += 1
            self.last_written = dict.fromkeys(ZONES, 0)
            return False
        copied = {z: v.detach().clone() for z, v in target.items()}
        for z in ZONES:
            getattr(self, z + '_state').zero_()
        self.erase_events += 1
        for z, value in copied.items():
            getattr(self, z + '_state').copy_(value)
            self.last_written[z] = int(value.sum())
            self.write_ones[z] += self.last_written[z]
        self.initialized = True
        return True

    @torch.no_grad()
    def weights(self):
        return {z: base.optical_weights(getattr(self, z + '_state'), self.config) for z in ZONES}


def training_logits(model, device, x):
    proxy, physical = model.weights(), device.weights()
    weights = {z: base.prior.common.surrogate_bridge(physical[z], proxy[z]) for z in ZONES}
    return model.logits_from_weights(x, weights)


def train(seed, args, config, tx, ty, vx, vy):
    model, device = Model(config, seed), Device(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(seed + 3000)
    initial = model.bits()
    device.program(initial)
    history, events, snapshots = [], [], []
    best, checkpoint = -1., None
    name = f'three_layer_seed{seed}'
    for epoch in range(args.epochs + 1):
        total, count = 0., 0
        grads = dict.fromkeys(ZONES, 0.)
        if epoch:
            model.train()
            for ids in torch.randperm(len(ty), generator=generator).split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(training_logits(model, device, tx[ids]), ty[ids])
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                for z in ZONES:
                    grads[z] += float(getattr(model, z + '_theta').grad.norm())
                optimizer.step()
                changed = device.program(model.bits())
                count += 1
                total += float(loss.detach()) * len(ids)
                events.append(dict(epoch=epoch, batch=count, rewritten=changed,
                                   erase_events=device.erase_events, written_ones=dict(device.last_written)))
        accuracy, _ = base.evaluate(model, vx, vy, device)
        bits = model.bits()
        if accuracy > best:
            best = accuracy
            checkpoint = dict(seed=seed, config=asdict(config), widths=list(WIDTHS),
                              selected_epoch=epoch, validation_accuracy=accuracy,
                              model_state=copy.deepcopy(model.state_dict()),
                              bits={z: v.clone() for z, v in bits.items()})
        history.append(dict(epoch=epoch, training_loss=total/len(ty) if epoch else None,
                            validation_accuracy=accuracy, erase_events=device.erase_events,
                            write_ones=dict(device.write_ones),
                            gradient_norm={z: grads[z]/max(1, count) for z in ZONES},
                            changed_bits={z: int((bits[z] != initial[z]).sum()) for z in ZONES}))
        if epoch % 5 == 0 or epoch == args.epochs:
            # Full electronic and binary state, not masks alone, enables replay.
            snapshots.append(dict(epoch=epoch, model_state=copy.deepcopy(model.state_dict())))
            print(f'{name} epoch={epoch} val={accuracy:.4f} erases={device.erase_events}', flush=True)
    torch.save(checkpoint, args.output / f'{name}.pt')
    torch.save(snapshots, args.output / f'{name}_trajectory.pt')
    base.prior.common.save_json(args.output / f'{name}_history.json', history)
    base.prior.common.save_json(args.output / f'{name}_events.json', events)
    return dict(arm='three_layer', seed=seed, selected_epoch=checkpoint['selected_epoch'],
                validation_accuracy=best, erase_events=device.erase_events,
                program_requests=device.requests, skipped_identical=device.skipped_identical,
                write_ones=device.write_ones,
                trainable_parameters=sum(p.numel() for p in model.parameters()))


def load_selected(path):
    checkpoint = torch.load(path, weights_only=True)
    if tuple(checkpoint['widths']) != WIDTHS:
        raise ValueError('Checkpoint architecture mismatch')
    config = base.prior.Config(**checkpoint['config'])
    model = Model(config, checkpoint['seed'])
    model.load_state_dict(checkpoint['model_state'])
    for z, value in model.bits().items():
        assert torch.equal(value, checkpoint['bits'][z])
    device = Device(config)
    device.program(checkpoint['bits'])
    return checkpoint, model, device


def mask_rows(bits, config):
    # Nonoverlapping lateral row banks, all sharing the same ten depth planes.
    for depth in range(config.layers):
        row_offset = 0
        for z in ZONES:
            value = bits[z][depth]
            for branch in range(2):
                for out in range(value.shape[1]):
                    for inp in range(value.shape[2]):
                        row = row_offset + branch*value.shape[1] + out
                        if z == 'conv':
                            col, row = 9*out + inp, branch
                        else:
                            col = inp
                        yield dict(layer_index=depth, zone=z, branch=branch,
                                   output_index=out, input_index=inp,
                                   x_um=col*config.center_pitch_um,
                                   y_um=row*config.center_pitch_um,
                                   write_bit=int(value[branch, out, inp]))
            row_offset += 2 if z == 'conv' else 2*value.shape[1]


def export_mask(path, bits, config):
    rows = mask_rows(bits, config)
    first = next(rows)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/scheme_a_three_layer_20260925')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=.005)
    args = parser.parse_args()
    if args.output.exists() or min(args.epochs, args.batch_size) < 1 or args.lr <= 0:
        parser.error('A new output directory and positive training budgets are required')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Duplicate seeds')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    config = base.prior.Config()
    sources = ['scheme_a_multilayer_head.py', 'scheme_a_material_head.py', 'scheme_a_binary10.py',
               'scheme_a_closed_loop.py', 'scheme_a_round2.py', 'scheme_a_mnist.py', 'reproduce.py']
    snapshot_dir = args.output/'source_snapshot'
    snapshot_dir.mkdir()
    for name in sources:
        shutil.copy2(ROOT/name, snapshot_dir/name)
    manifest = dict(status='training', config=asdict(config), widths=list(WIDTHS),
                    seeds=args.seeds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    versions=dict(torch=str(torch.__version__), numpy=np.__version__),
                    source_hashes={s: base.prior.common.hash_file(ROOT/s) for s in sources},
                    physical_sites={z: int(np.prod(s)) for z, s in shapes(config).items()},
                    electronic_biases=3+sum(WIDTHS[1:]),
                    protocol='Fresh single-layer control and three-layer model; same fixed split and minibatch order; earliest best validation checkpoint; all selections before test.',
                    assumptions=[
                        'Three dense layers means 27->32->16->10, with two hidden ReLU layers; no residuals.',
                        'All four zones are trained jointly; every weight uses ten binary depth sites in each of two signed branches.',
                        'Biases, difference, summation, gain, ReLU and scale restoration are electronic.',
                        'Each dense input is separately normalized to [0,1], replicated to every output, then re-encoded at 418 nm.',
                        'Four sequential bank reads (convolution and three dense banks); not thirty depth planes.',
                        'Atomic whole-volume erase/rewrite updates all four banks; identical masks skip programming.',
                        'Zero noise/crosstalk, ideal re-encoding, no wave propagation, power, timing or hardware validation.',
                        'Same training budget but more weights and electronic biases; not a parameter-matched depth ablation.',
                        'Architecture fixed before results; no hyperparameter sweep or test-based selection.',
                        'Previously used MNIST test set, not a new blind holdout; three initialization seeds only.'])
    save = base.prior.common.save_json
    save(args.output/'manifest.json', manifest)
    raw, hashes = base.prior.common.r2.base.fetch_data(ROOT/'data/MNIST/raw')
    with np.load(ROOT/'outputs/scheme_a_mnist/split_indices.npz') as splits:
        tr, va = splits['train'], splits['validation']
        assert len(tr) == 50000 and len(va) == 10000 and not set(tr) & set(va)
        np.savez_compressed(args.output/'split_indices.npz', **{k: splits[k] for k in splits.files})
    manifest['dataset_hashes'] = hashes
    save(args.output/'manifest.json', manifest)
    x = base.prior.common.r2.patches(raw['train-images-idx3-ubyte.gz'], 14)
    y = torch.from_numpy(raw['train-labels-idx1-ubyte.gz'].astype(np.int64))
    tx, ty, vx, vy = x[tr], y[tr], x[va], y[va]
    del x
    rows = []
    for seed in args.seeds:
        rows.append(base.train('joint_closed_loop', seed, args, config, tx, ty, vx, vy))
        rows.append(train(seed, args, config, tx, ty, vx, vy))
        save(args.output/'validation_results.json', rows)
    save(args.output/'selection_before_test.json', rows)
    del tx, ty, vx, vy
    manifest['status'] = 'testing'
    save(args.output/'manifest.json', manifest)
    xt = base.prior.common.r2.patches(raw['t10k-images-idx3-ubyte.gz'], 14)
    yt = torch.from_numpy(raw['t10k-labels-idx1-ubyte.gz'].astype(np.int64))
    audit = []
    for row in rows:
        name = f"{row['arm']}_seed{row['seed']}"
        loader, exporter = ((load_selected, export_mask) if row['arm'] == 'three_layer'
                            else (base.load_selected, base.export_mask))
        ck, model, device = loader(args.output/f'{name}.pt')
        accuracy, pred = base.evaluate(model, xt, yt, device)
        proxy_accuracy, proxy_pred = base.evaluate(model, xt, yt)
        assert accuracy == proxy_accuracy and torch.equal(pred, proxy_pred)
        matrix = np.zeros((10, 10), dtype=np.int64)
        np.add.at(matrix, (yt.numpy(), pred.numpy()), 1)
        assert matrix.sum() == 10000
        row.update(test_accuracy=float((pred == yt).sum())/len(yt),
                   deployment_erases=device.erase_events)
        np.savez_compressed(args.output/f'{name}_test.npz', predictions=pred.numpy(),
                            labels=yt.numpy(), confusion_counts=matrix)
        exporter(args.output/f'{name}_write_mask.csv', ck['bits'], config)
        audit.append(dict(name=name, proxy_device_predictions_equal=True,
                          all_masks_binary=True, test_images=len(yt)))
        print('TEST', name, row['test_accuracy'], flush=True)
    save(args.output/'results.json', rows)
    summary = {}
    for arm in ('joint_closed_loop', 'three_layer'):
        values = [r['test_accuracy'] for r in rows if r['arm'] == arm]
        summary[arm] = dict(mean=float(np.mean(values)), population_sd=float(np.std(values)), seed_values=values)
    summary['paired_gain_pp'] = [100*(b-a) for a,b in zip(
        summary['joint_closed_loop']['seed_values'], summary['three_layer']['seed_values'])]
    save(args.output/'summary.json', summary)
    save(args.output/'audit.json', audit)
    assert all(base.prior.common.hash_file(ROOT/s) == h for s,h in manifest['source_hashes'].items())
    manifest.update(status='complete', elapsed_seconds=time.time()-started)
    save(args.output/'manifest.json', manifest)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
