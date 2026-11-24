"""Validated image-to-patch conversion shared by data loading and inference."""

import torch
import torch.nn.functional as F

from .config import NetworkConfig


def image_patches(images: torch.Tensor, config: NetworkConfig | None = None) -> torch.Tensor:
    """Convert [B,1,14,14] relative intensities into 100 valid 5x5 patches."""
    config = NetworkConfig() if config is None else config
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, config.input_side, config.input_side):
        raise ValueError("Expected images of shape [batch,1,14,14] in relative intensity units")
    if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
        raise ValueError("Image intensities must be finite values in [0,1]")
    return F.unfold(images, kernel_size=config.kernel_size, stride=1).transpose(1, 2).contiguous()
