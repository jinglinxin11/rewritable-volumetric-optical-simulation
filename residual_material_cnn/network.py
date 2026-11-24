"""Reference training model and compiled, material-only weight inference."""

import torch
from torch import nn
import torch.nn.functional as F

from .config import MaterialConfig, NetworkConfig
from .encoding import encode_matrix
from .readout import IdealResidualReadout


def image_patches(images, config=NetworkConfig()):
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, config.input_side, config.input_side):
        raise ValueError('Expected images of shape [batch,1,14,14] in relative intensity units')
    if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
        raise ValueError('Image intensities must be finite values in [0,1]')
    return F.unfold(images, kernel_size=config.kernel_size, stride=1).transpose(1, 2).contiguous()


def classify_patches(patches, config, affine):
    if patches.ndim != 3 or tuple(patches.shape[1:]) != (100, 25):
        raise ValueError('Expected 100 valid 5x5 patches per image')
    value = affine('conv', patches)
    hidden = F.relu(value.transpose(1, 2).reshape(-1, config.channels, 10, 10))
    hidden = F.adaptive_avg_pool2d(hidden, config.pooled_side).flatten(1)
    for i, name in enumerate(('fc1', 'fc2', 'fc3')):
        hidden = affine(name, hidden)
        if i < 2:
            hidden = F.relu(hidden)
    return hidden


class ReferenceCNN(nn.Module):
    """Offline bounded-weight training; no convolution bias, electronic dense biases."""

    def __init__(self, config=NetworkConfig()):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleDict({name: nn.Linear(ins, outs, bias=name != 'conv')
                                     for name, (outs, ins) in config.shapes.items()})

    def forward_patches(self, patches):
        return classify_patches(patches, self.config, lambda name, value: self.layers[name](value))

    def forward(self, images):
        return self.forward_patches(image_patches(images, self.config))

    @torch.no_grad()
    def constrain_weights(self):
        for layer in self.layers.values():
            layer.weight.clamp_(-1., 1.)


class CompiledCNN(nn.Module):
    """Inference stores branch transmissions/counts, bank scales and electronic biases."""

    def __init__(self, encoded, dense_biases, config=NetworkConfig()):
        super().__init__()
        self.config = config
        if set(encoded) != set(config.shapes) or set(dense_biases) != {'fc1', 'fc2', 'fc3'}:
            raise ValueError('All four weight matrices and three dense electronic biases are required')
        materials = [packet.material for packet in encoded.values()]
        if any(m != materials[0] for m in materials):
            raise ValueError('All matrices must share the same material configuration')
        for name, shape in config.shapes.items():
            if tuple(encoded[name].counts.shape[2:]) != shape:
                raise ValueError(f'Invalid matrix dimensions for {name}')
        self.layers = nn.ModuleDict({name: IdealResidualReadout(encoded[name]) for name in config.shapes})
        for name in ('fc1', 'fc2', 'fc3'):
            bias = torch.as_tensor(dense_biases[name], dtype=torch.float64).detach().clone()
            if tuple(bias.shape) != (config.shapes[name][0],) or not bool(torch.isfinite(bias).all()):
                raise ValueError(f'Invalid electronic bias for {name}')
            self.register_buffer(name + '_bias', bias)

    @property
    def material(self):
        return self.layers['conv'].material

    def affine(self, name, value):
        result = self.layers[name](value)
        return result if name == 'conv' else result + getattr(self, name + '_bias')

    def forward_patches(self, patches):
        return classify_patches(patches, self.config, self.affine)

    def forward(self, images):
        return self.forward_patches(image_patches(images, self.config))


def compile_network(reference, material=MaterialConfig()):
    encoded = {name: encode_matrix(layer.weight.detach().cpu().double().numpy(), material)
               for name, layer in reference.layers.items()}
    biases = {name: layer.bias.detach().cpu().double().numpy() for name, layer in reference.layers.items()
              if layer.bias is not None}
    return CompiledCNN(encoded, biases, reference.config)
