"""Checksum-verified official MNIST loader; no figures or document processing."""
import gzip
import hashlib
import struct
import urllib.request
import numpy as np


FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def fetch_data(folder):
    folder.mkdir(parents=True, exist_ok=True)
    data, hashes = {}, {}
    for name, md5 in FILES.items():
        path = folder / name
        if not path.exists():
            print("Downloading", name, flush=True)
            with urllib.request.urlopen(BASE + name, timeout=120) as response:
                payload = response.read()
            if hashlib.md5(payload).hexdigest() != md5:
                raise ValueError("MNIST checksum mismatch: " + name)
            path.write_bytes(payload)
        payload = path.read_bytes()
        if hashlib.md5(payload).hexdigest() != md5:
            raise ValueError("Existing MNIST checksum mismatch: " + name)
        hashes[name] = {"md5": md5, "sha256": hashlib.sha256(payload).hexdigest(), "url": BASE+name}
        raw = gzip.decompress(payload)
        magic, count = struct.unpack(">II", raw[:8])
        if magic == 2051:
            rows, cols = struct.unpack(">II", raw[8:16])
            data[name] = np.frombuffer(raw, np.uint8, offset=16).copy().reshape(count, rows, cols)
        elif magic == 2049:
            data[name] = np.frombuffer(raw, np.uint8, offset=8).copy()
        else:
            raise ValueError("Invalid IDX magic")
    return data, hashes
