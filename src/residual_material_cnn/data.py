"""Read verified local MNIST files and the retained train/validation split."""

import gzip
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .preprocessing import image_patches

MNIST_MD5 = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def parse_idx(payload):
    raw = gzip.decompress(payload)
    if len(raw) < 8:
        raise ValueError("Truncated IDX header")
    magic, count = struct.unpack(">II", raw[:8])
    if magic == 2051 and len(raw) >= 16:
        rows, cols = struct.unpack(">II", raw[8:16])
        if (rows, cols) != (28, 28) or len(raw) != 16 + count * rows * cols:
            raise ValueError("Invalid MNIST image dimensions or payload length")
        return np.frombuffer(raw, np.uint8, offset=16).copy().reshape(count, rows, cols)
    if magic == 2049 and len(raw) == 8 + count:
        return np.frombuffer(raw, np.uint8, offset=8).copy()
    raise ValueError("Unsupported or malformed IDX file")


def validate_split(train, validation):
    for indices, count in ((train, 50000), (validation, 10000)):
        if indices.ndim != 1 or len(indices) != count or indices.dtype.kind not in "iu":
            raise ValueError("Expected integer MNIST indices of lengths 50000 and 10000")
        if (
            np.any(indices < 0)
            or np.any(indices >= 60000)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("Split indices are duplicated or outside the training cohort")
    if np.intersect1d(train, validation).size:
        raise ValueError("Training and validation indices overlap")


@dataclass
class MNISTData:
    train: tuple[torch.Tensor, torch.Tensor]
    validation: tuple[torch.Tensor, torch.Tensor]
    test: tuple[torch.Tensor, torch.Tensor]
    provenance: dict


def load_mnist(data_dir, split_path):
    data_dir, split_path = Path(data_dir), Path(split_path)
    raw, hashes = {}, {}
    for name, expected in MNIST_MD5.items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Local MNIST file missing: {path}; this loader does not download files"
            )
        payload = path.read_bytes()
        if hashlib.md5(payload).hexdigest() != expected:
            raise ValueError(f"MNIST checksum mismatch: {path}")
        raw[name] = parse_idx(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    with np.load(split_path, allow_pickle=False) as split:
        train, validation = split["train"], split["validation"]
    validate_split(train, validation)

    def prepare(image_name, label_name):
        images, labels = raw[image_name], raw[label_name]
        if len(images) != len(labels) or np.any(labels > 9):
            raise ValueError("Invalid image/label correspondence")
        tensor = F.interpolate(
            torch.from_numpy(images).float()[:, None] / 255, size=(14, 14), mode="area"
        )
        return image_patches(tensor), torch.from_numpy(labels.astype(np.int64))

    all_x, all_y = prepare("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
    test = prepare("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    if len(all_y) != 60000 or len(test[1]) != 10000:
        raise ValueError("Expected the complete MNIST 60000/10000 cohorts")
    return MNISTData(
        (all_x[train], all_y[train]),
        (all_x[validation], all_y[validation]),
        test,
        dict(
            mnist_sha256=hashes,
            split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
            counts=dict(train=50000, validation=10000, test=10000),
            test_cohort="existing MNIST test, reused by this project",
        ),
    )
