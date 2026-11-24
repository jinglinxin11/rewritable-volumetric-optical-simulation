import gzip
import struct

import numpy as np
import pytest

from residual_material_cnn.data import load_mnist, parse_idx, validate_split


def test_idx_images_keep_row_column_order():
    pixels = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    payload = struct.pack(">IIII", 2051, 2, 28, 28) + pixels.tobytes()
    np.testing.assert_array_equal(parse_idx(gzip.compress(payload)), pixels)


@pytest.mark.parametrize("raw", [b"", struct.pack(">IIII", 2051, 2, 14, 14)])
def test_invalid_idx_header_or_shape_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_idx(gzip.compress(raw))


def test_missing_local_data_does_not_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not download"):
        load_mnist(tmp_path, tmp_path / "missing_split.npz")
    assert not list(tmp_path.iterdir())


def test_corrupt_local_data_is_rejected_before_parsing(tmp_path):
    (tmp_path / "train-images-idx3-ubyte.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_mnist(tmp_path, tmp_path / "missing_split.npz")


@pytest.mark.parametrize(
    "train", [np.arange(10), np.arange(50000, dtype=float), np.full(50000, -1)]
)
def test_split_invalid_length_type_or_indices_is_rejected(train):
    with pytest.raises(ValueError):
        validate_split(train, np.arange(50000, 60000))
