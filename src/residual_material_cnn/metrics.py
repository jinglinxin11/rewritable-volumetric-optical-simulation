"""Keep weight, MAC and classification metrics separate."""

import math

import numpy as np


def error_metrics(target, actual, span=2.0):
    target, actual = np.asarray(target, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if target.shape != actual.shape or target.size == 0 or not np.isfinite(span) or span <= 0:
        raise ValueError("Matching nonempty arrays and a positive fixed span are required")
    if not np.isfinite(target).all() or not np.isfinite(actual).all():
        raise ValueError("Metric inputs must be finite")
    error = actual - target
    rmse = float(np.sqrt(np.mean(error**2)))
    return dict(
        rmse=rmse,
        max_abs_error=float(np.max(np.abs(error))),
        fixed_span=float(span),
        equivalent_bits=math.log2(span / (math.sqrt(12) * rmse)) if rmse else None,
        exact=rmse == 0,
    )


def legacy_mac_benchmark():
    """The former 16-input/8-output clean fixture, not a network-precision claim."""
    import torch

    from .encoding import encode_matrix
    from .readout import IdealResidualReadout

    targets, actual = [], []
    for seed in (41, 42, 43):
        rng = np.random.default_rng(seed)
        weight = rng.uniform(-1, 1, (8, 16))
        x = rng.random((128, 16))
        with torch.no_grad():
            output = IdealResidualReadout(encode_matrix(weight))(torch.from_numpy(x)).numpy() / 16
        targets.append(x @ weight.T / 16)
        actual.append(output)
    return dict(
        inputs=16,
        outputs=8,
        seeds=[41, 42, 43],
        normalization="X W.T / 16",
        evidence="clean ideal MAC; not hardware ENOB or classification accuracy",
        **error_metrics(np.concatenate(targets), np.concatenate(actual), 2.0),
    )
