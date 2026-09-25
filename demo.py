"""Deterministic, no-download demonstration of the four-bank state model."""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import scheme_a_conv_search as simulation


def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(202609)
    raw = rng.integers(0, 256, (16, 28, 28), dtype=np.uint8)
    architecture = simulation.Architecture(12, 5, 1)
    config = simulation.base.prior.Config()
    model = simulation.Model(config, architecture, 3)
    device = simulation.Device(config, architecture)
    device.program(model.bits())
    x, labels = simulation.patches(raw, architecture), torch.arange(16) % 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss = F.cross_entropy(simulation.previous.training_logits(model, device, x), labels)
    loss.backward()
    gradients = {bank: getattr(model, bank + "_theta").grad for bank in simulation.ZONES}
    assert all(bool(torch.isfinite(g).all()) and float(g.norm()) > 0 for g in gradients.values())
    optimizer.step()
    device.program(model.bits())
    torch.testing.assert_close(model(x), model.logits_from_weights(x, device.weights()), rtol=0, atol=0)
    ones = {bank: torch.ones_like(bits) for bank, bits in model.bits().items()}
    zeros = {bank: torch.zeros_like(bits) for bank, bits in ones.items()}
    device.program(ones)
    device.program(zeros)
    assert all(int(getattr(device, bank + "_state").sum()) == 0 for bank in simulation.ZONES)
    assert not device.program(zeros)
    print(json.dumps({
        "synthetic_samples": len(labels), "synthetic_loss": float(loss.detach()),
        "all_bank_gradients_finite_nonzero": True, "proxy_device_exact": True,
        "reset_removes_written_states": True, "identical_masks_skip_programming": True,
        "resources": simulation.resources(config, architecture),
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
