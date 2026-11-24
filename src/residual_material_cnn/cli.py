"""Installed command-line interface; stdout is JSON and logs go to stderr."""

import argparse
import json
import logging
from pathlib import Path

import torch

from .artifacts import export_masks, load_compiled, save_json
from .config import configuration
from .data import MNISTData, load_mnist
from .metrics import legacy_mac_benchmark
from .pipeline import run_training
from .resources import bundled_model, bundled_split
from .training import evaluate

LOGGER = logging.getLogger(__name__)


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/MNIST/raw"),
        help="Local MNIST gzip directory, relative to the current directory",
    )
    parser.add_argument("--split-indices", type=Path, help="Override the packaged fixed split")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="External schema-1 compiled NPZ model")
    source.add_argument(
        "--example-seed", type=int, choices=(0, 1, 2), help="Use a packaged, frozen compiled model"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe", help="Print network, material, and resource counts")
    train = commands.add_parser(
        "train", help="Train offline, compile, export, and compare predictions"
    )
    add_data_arguments(train)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=0.005)
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate compiled counts without float reference weights"
    )
    add_data_arguments(evaluation)
    add_model_arguments(evaluation)
    evaluation.add_argument("--split", choices=("validation", "test"), default="validation")
    export = commands.add_parser("export", help="Export exact masks and illustrative geometry")
    add_model_arguments(export)
    export.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser(
        "benchmark", help="Replay the clean random-MAC fixture, not hardware ENOB"
    )
    benchmark.add_argument("--output", type=Path)
    return parser


def load_data(args: argparse.Namespace) -> MNISTData:
    if args.split_indices is not None:
        return load_mnist(args.data, args.split_indices)
    with bundled_split() as path:
        return load_mnist(args.data, path)


def load_model(args: argparse.Namespace):
    if args.model is not None:
        return load_compiled(args.model)
    with bundled_model(args.example_seed) as path:
        return load_compiled(path)


def dispatch(args: argparse.Namespace) -> dict:
    if args.command == "describe":
        return configuration()
    if args.command == "train":
        if args.output.exists():
            raise FileExistsError(
                "Output directory exists; use a new directory to preserve previous runs"
            )
        return run_training(
            load_data(args),
            args.output,
            args.seeds,
            args.epochs,
            args.batch_size,
            args.learning_rate,
        )
    if args.command == "evaluate":
        data = load_data(args)
        model = load_model(args)
        evaluated = evaluate(model, *getattr(data, args.split))
        return dict(
            model=str(args.model) if args.model else f"packaged:seed{args.example_seed}",
            split=args.split,
            accuracy_pct=evaluated["accuracy_pct"],
            readout="ideal; evaluated from compiled masks, not the reference matrix",
        )
    if args.command == "export":
        export_masks(load_model(args), args.output)
        return dict(exported=str(args.output))
    result = legacy_mac_benchmark()
    if args.output:
        if args.output.exists():
            raise FileExistsError("Benchmark output exists; use a new filename")
        save_json(args.output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    try:
        result = dispatch(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    except RuntimeError:
        LOGGER.exception("The simulation failed; no success result was produced")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
