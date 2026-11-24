"""Greedy residual encoding to writable layer counts, independent of training."""

from dataclasses import dataclass

import numpy as np

from .config import MaterialConfig


@dataclass(frozen=True)
class Codebook:
    values: np.ndarray
    counts: np.ndarray
    alpha: float


def build_codebook(material: MaterialConfig):
    t = material.site_transmission**np.arange(material.depth + 1)
    pairs = {}
    for plus in range(len(t)):
        for minus in range(len(t)):
            value = float(np.round((t[plus] - t[minus]) / material.contrast, 14))
            light = t[plus] + t[minus]
            if value not in pairs or light < pairs[value][0]:
                pairs[value] = (light, plus, minus)
    values = np.array(sorted(pairs), dtype=np.float64)
    counts = np.array([[pairs[v][1], pairs[v][2]] for v in values], dtype=np.uint8)
    return Codebook(values, counts, float(np.diff(values).max() / 2))


def nearest_indices(target, values):
    right = np.searchsorted(values, target).clip(1, len(values) - 1)
    left = right - 1
    return np.where(np.abs(target - values[left]) <= np.abs(target - values[right]), left, right)


@dataclass(frozen=True)
class EncodedMatrix:
    counts: np.ndarray  # [residual_bank, branch, output, input]
    scales: np.ndarray
    material: MaterialConfig

    def __post_init__(self):
        c = np.asarray(self.counts)
        s = np.asarray(self.scales)
        if c.ndim != 4 or c.shape[:2] != (self.material.residual_banks, 2) or min(c.shape[2:]) < 1:
            raise ValueError('counts must have shape [3,2,output,input]')
        if c.dtype.kind not in 'iu' or np.any(c < 0) or np.any(c > self.material.depth):
            raise ValueError('Written layer counts must be integers from zero to ten')
        if s.shape != (self.material.residual_banks,) or not np.isfinite(s).all() or np.any(s <= 0):
            raise ValueError('Each residual bank requires a finite positive reconstruction scale')

    def effective_weights(self):
        t = self.material.site_transmission**self.counts.astype(np.float64)
        return np.sum(self.scales[:, None, None] * (t[:, 0] - t[:, 1]) / self.material.contrast, axis=0)

    def write_masks(self):
        rank = np.empty(self.material.depth, dtype=int)
        rank[np.array(self.material.layer_order)] = np.arange(self.material.depth)
        return rank[None, :, None, None, None] < self.counts[:, None]


def encode_matrix(weight, material=MaterialConfig()):
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2 or min(weight.shape) < 1 or not np.isfinite(weight).all():
        raise ValueError('A nonempty finite [output,input] weight matrix is required')
    if np.any(np.abs(weight) > 1 + 1e-12):
        raise ValueError('Weights must be within [-1,1]; do not silently clip during compilation')
    book = build_codebook(material)
    residual = weight.copy()
    scales = book.alpha**np.arange(material.residual_banks, dtype=np.float64)
    encoded = []
    for scale in scales:
        indices = nearest_indices(residual / scale, book.values)
        residual -= scale * book.values[indices]
        encoded.append(book.counts[indices].transpose(2, 0, 1))
    return EncodedMatrix(np.stack(encoded), scales, material)
