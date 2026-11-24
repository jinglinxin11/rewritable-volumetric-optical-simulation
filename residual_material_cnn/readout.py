"""Ideal branch readout and electronic reconstruction; no hidden dense weights."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .encoding import EncodedMatrix


class IdealResidualReadout(nn.Module):
    def __init__(self, encoded: EncodedMatrix):
        super().__init__()
        self.material = encoded.material
        counts = np.ascontiguousarray(encoded.counts, dtype=np.uint8)
        self.register_buffer('counts', torch.from_numpy(counts.copy()))
        self.register_buffer('scales', torch.from_numpy(np.ascontiguousarray(encoded.scales, dtype=np.float64).copy()))
        transmission = np.ascontiguousarray(encoded.material.site_transmission**counts.astype(np.float64))
        self.register_buffer('transmission', torch.from_numpy(transmission).double())

    def forward(self, x):
        x = x.to(device=self.transmission.device, dtype=self.transmission.dtype)
        if x.shape[-1] != self.transmission.shape[-1]:
            raise ValueError('Input size disagrees with the encoded matrix')
        if not bool(torch.isfinite(x).all()) or bool((x < 0).any()):
            raise ValueError('Optical intensity inputs must be finite and nonnegative')
        scale = x.amax(-1, keepdim=True).clamp_min(1.)
        illumination = x / scale
        result = torch.zeros(*x.shape[:-1], self.transmission.shape[-2], dtype=x.dtype, device=x.device)
        for bank in range(len(self.scales)):
            plus = F.linear(illumination, self.transmission[bank, 0])
            minus = F.linear(illumination, self.transmission[bank, 1])
            result += self.scales[bank] * (plus - minus) / self.material.contrast
        return result * scale

    def encoded_matrix(self):
        return EncodedMatrix(self.counts.cpu().numpy().copy(), self.scales.cpu().numpy().copy(), self.material)
