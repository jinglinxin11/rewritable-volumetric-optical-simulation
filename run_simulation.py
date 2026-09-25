"""Train and evaluate the fixed twelve-kernel, four-bank MNIST architecture."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import scheme_a_conv_search as simulation
from prepare_data import prepare

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/selected_model")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/MNIST/raw")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; existing results are not overwritten")
    if min(args.epochs, args.batch_size) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Positive finite training budgets required")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Use distinct nonnegative seeds")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    config = simulation.base.prior.Config()
    architecture = simulation.Architecture(12, 5, 1)
    raw, hashes, split = prepare(args.data_dir, args.output / "split_indices.npz")
    plan = {
        "architecture": asdict(architecture), "config": asdict(config),
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "seeds": args.seeds, "dataset_hashes": hashes,
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(ROOT.glob("*.py"))},
        "selection": "Fixed architecture; earliest best validation checkpoint per seed; no test-based selection.",
    }
    simulation.SAVE(args.output / "run_plan.json", plan)
    x = simulation.patches(raw["train-images-idx3-ubyte.gz"], architecture)
    y = torch.from_numpy(raw["train-labels-idx1-ubyte.gz"].astype(np.int64))
    tr, va = split["train"], split["validation"]
    tx, vx, ty, vy = x[tr], x[va], y[tr], y[va]
    del x
    records = [simulation.train(config, architecture, seed, args, args.output, tx, ty, vx, vy)
               for seed in args.seeds]
    simulation.SAVE(args.output / "selection_before_test.json", records)
    del tx, vx
    xt = simulation.patches(raw["t10k-images-idx3-ubyte.gz"], architecture)
    yt = torch.from_numpy(raw["t10k-labels-idx1-ubyte.gz"].astype(np.int64))
    for row in records:
        name = f"{architecture.name}_seed{row['seed']}"
        _, model, device = simulation.load_selected(args.output / f"{name}.pt")
        _, prediction = simulation.base.evaluate(model, xt, yt, device)
        _, proxy = simulation.base.evaluate(model, xt, yt)
        if not torch.equal(prediction, proxy):
            raise RuntimeError("Device and proxy predictions differ")
        confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion, (yt.numpy(), prediction.numpy()), 1)
        row["test_accuracy_percent"] = 100 * int((prediction == yt).sum()) / len(yt)
        row["macro_f1_percent"] = float(np.mean(
            200 * np.diag(confusion) / (confusion.sum(0) + confusion.sum(1))))
        np.savez_compressed(args.output / f"{name}_test.npz", predictions=prediction.numpy(),
                            labels=yt.numpy(), confusion_counts=confusion)
    summary = {key: {"mean": float(np.mean([r[key] for r in records])),
                     "population_sd_pp": float(np.std([r[key] for r in records]))}
               for key in ("test_accuracy_percent", "macro_f1_percent")}
    summary["resources"] = simulation.resources(config, architecture)
    simulation.SAVE(args.output / "test_results.json", records)
    simulation.SAVE(args.output / "test_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
