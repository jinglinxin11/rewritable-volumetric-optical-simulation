"""Download verified MNIST and create the deterministic 50k/10k/10k split."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from scheme_a_mnist import fetch_data

ROOT = Path(__file__).resolve().parent


def make_split(labels):
    train, validation = train_test_split(
        np.arange(60000), test_size=10000, random_state=202609, stratify=labels
    )
    return {"train": train, "validation": validation, "test": np.arange(10000)}


def prepare(data_dir, split_path):
    raw, hashes = fetch_data(Path(data_dir))
    if len(raw["train-labels-idx1-ubyte.gz"]) != 60000:
        raise ValueError("Expected 60,000 official MNIST training labels")
    split = make_split(raw["train-labels-idx1-ubyte.gz"])
    split_path = Path(split_path)
    if split_path.exists():
        with np.load(split_path) as saved:
            if set(saved.files) != set(split) or any(
                not np.array_equal(saved[key], value) for key, value in split.items()
            ):
                raise ValueError("Existing split differs from the fixed protocol")
    else:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_path, **split)
    return raw, hashes, split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/MNIST/raw")
    parser.add_argument("--split", type=Path, default=ROOT / "outputs/scheme_a_mnist/split_indices.npz")
    args = parser.parse_args()
    _, _, split = prepare(args.data_dir, args.split)
    print({key: len(value) for key, value in split.items()})
    print(args.split)


if __name__ == "__main__":
    main()
