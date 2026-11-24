"""Command line interface: python -m residual_material_cnn --help."""

import argparse
import json
from pathlib import Path

import torch

from .artifacts import export_masks, load_compiled, save_json
from .config import configuration
from .data import load_mnist
from .metrics import legacy_mac_benchmark
from .pipeline import run_training
from .training import evaluate


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent


def add_data_arguments(parser):
    parser.add_argument('--data', type=Path, default=ROOT / 'data/MNIST/raw')
    parser.add_argument('--split-indices', type=Path, default=PACKAGE / 'assets/split_indices.npz')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('describe', help='Print the selected network, material and resource counts')
    train = commands.add_parser('train', help='Train offline, compile, export masks, and compare held-out predictions')
    add_data_arguments(train)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    train.add_argument('--epochs', type=int, default=20)
    train.add_argument('--batch-size', type=int, default=512)
    train.add_argument('--learning-rate', type=float, default=.005)
    evaluate_parser = commands.add_parser('evaluate', help='Evaluate only a compiled model, without original float weights')
    add_data_arguments(evaluate_parser)
    evaluate_parser.add_argument('--model', type=Path, required=True)
    evaluate_parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    export = commands.add_parser('export', help='Export exact write masks and illustrative geometry')
    export.add_argument('--model', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    benchmark = commands.add_parser('benchmark', help='Replay the old clean 18.55-bit MAC fixture, not network ENOB')
    benchmark.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.command == 'describe':
        result = configuration()
    elif args.command == 'train':
        if args.output.exists():
            parser.error('Output directory exists; use a new directory to preserve previous runs')
        result = run_training(load_mnist(args.data, args.split_indices), args.output,
                              args.seeds, args.epochs, args.batch_size, args.learning_rate)
    elif args.command == 'evaluate':
        data = load_mnist(args.data, args.split_indices)
        evaluated = evaluate(load_compiled(args.model), *getattr(data, args.split))
        result = dict(model=str(args.model), split=args.split, accuracy_pct=evaluated['accuracy_pct'],
                      readout='ideal; evaluated from compiled masks, not the reference matrix')
    elif args.command == 'export':
        export_masks(load_compiled(args.model), args.output)
        result = dict(exported=str(args.output))
    else:
        result = legacy_mac_benchmark()
        if args.output:
            if args.output.exists():
                parser.error('Benchmark output exists; use a new filename')
            save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
