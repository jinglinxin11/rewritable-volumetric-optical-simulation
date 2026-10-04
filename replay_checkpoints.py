"""Replay locally supplied archived checkpoints, without training or selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import scheme_a_conv_search as simulation
from prepare_data import prepare
from supplementary_reporting import classification_metrics, load_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Archive with confirmation/ and provenance/")
    parser.add_argument("--data-dir", type=Path, default=Path("data/MNIST/raw"))
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    load_tables(args.archive)  # Validate all member hashes before loading checkpoints.
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    raw, hashes, split = prepare(args.data_dir, args.output / "split_indices.npz")
    with np.load(args.archive / "provenance/split_indices.npz", allow_pickle=False) as saved:
        for key, values in split.items():
            np.testing.assert_array_equal(values, saved[key])
    labels = raw["t10k-labels-idx1-ubyte.gz"].astype(np.int64)
    metadata = json.loads((args.archive / "provenance/test_history.json").read_text(encoding="utf-8"))
    if hashes != metadata["dataset_hashes"]:
        raise ValueError("Dataset identity differs from archived run")
    results = []
    for architecture in ("c3_k3_s1", "c12_k5_s1"):
        for seed in (3, 4, 5):
            stem = f"{architecture}_seed{seed}"
            path = args.archive / "confirmation" / f"{stem}.pt"
            checkpoint, model, device = simulation.load_selected(path)
            features = simulation.patches(raw["t10k-images-idx3-ubyte.gz"], model.arch)
            _, pred = simulation.base.evaluate(model, features, torch.from_numpy(labels), device)
            with np.load(args.archive / "confirmation" / f"{stem}_test.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(labels, saved["labels"])
                np.testing.assert_array_equal(pred.numpy(), saved["predictions"])
            result = classification_metrics(labels, pred.numpy())
            results.append({"architecture": architecture, "seed": seed,
                            "selected_epoch": checkpoint["selected_epoch"],
                            "accuracy_percent": result["accuracy_percent"],
                            "macro_f1_percent": result["macro_f1_percent"],
                            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "exact_prediction_match": True})
            del features, model, device
    (args.output / "checkpoint_replay.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
