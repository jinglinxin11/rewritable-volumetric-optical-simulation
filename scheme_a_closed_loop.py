"""Numerical state/proxy bridge and shared simulation utilities."""
import hashlib
import json
import torch.nn.functional as F
import scheme_a_round2 as r2


def surrogate_bridge(observed, proxy):
    """Numerically physical forward; ideal-proxy backward (not the device Jacobian)."""
    return observed.detach()+(proxy-proxy.detach())


def electronic_logits(model, z):
    h = F.relu(z.transpose(1, 2).reshape(-1, 3, 12, 12)+model.bias[None, :, None, None])
    return model.head(F.avg_pool2d(h, 4).flatten(1))


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
