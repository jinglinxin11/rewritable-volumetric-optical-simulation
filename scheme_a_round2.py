"""Shared 14-pixel input patch preparation for binary-material baselines."""
import torch
import torch.nn.functional as F
import scheme_a_mnist as base


def patches(images, size):
    x = torch.from_numpy(images).float()[:, None]/255
    x = F.interpolate(x, size=(size, size), mode="area")
    return F.unfold(x, 3).transpose(1, 2).contiguous()
