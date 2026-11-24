"""Physical encoding and network configuration; no filesystem side effects."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MaterialConfig:
    site_transmission: float = .9
    depth: int = 10
    residual_banks: int = 3
    layer_order: tuple[int, ...] = (4, 5, 3, 6, 2, 7, 1, 8, 0, 9)
    pitch_um: float = 50.
    layer_spacing_um: float = 400.
    sample_thickness_um: float = 4000.

    def __post_init__(self):
        if not 0 < self.site_transmission < 1:
            raise ValueError('site_transmission must be between zero and one')
        if self.depth != 10 or self.residual_banks != 3:
            raise ValueError('This scheme uses exactly ten layers and three residual banks')
        if tuple(sorted(self.layer_order)) != tuple(range(self.depth)):
            raise ValueError('layer_order must be a permutation of zero-based depth indices')
        if self.pitch_um <= 0 or self.layer_spacing_um <= 0:
            raise ValueError('Geometrical distances must be positive')
        if self.sample_thickness_um < (self.depth - 1) * self.layer_spacing_um:
            raise ValueError('Depth-plane centres do not fit in the stated thickness')

    @property
    def contrast(self):
        return 1 - self.site_transmission**self.depth


@dataclass(frozen=True)
class NetworkConfig:
    input_side: int = 14
    channels: int = 3
    kernel_size: int = 5
    pooled_side: int = 3
    dense_widths: tuple[int, ...] = (27, 32, 16, 10)

    def __post_init__(self):
        expected = (14, 3, 5, 3, (27, 32, 16, 10))
        actual = (self.input_side, self.channels, self.kernel_size, self.pooled_side, tuple(self.dense_widths))
        if actual != expected:
            raise ValueError('The selected scheme is 14x14, three 5x5 kernels, pool 3x3, 27->32->16->10')

    @property
    def shapes(self):
        return dict(conv=(3, 25), fc1=(32, 27), fc2=(16, 32), fc3=(10, 16))

    def resources(self, material: MaterialConfig):
        weights = sum(out * ins for out, ins in self.shapes.values())
        return dict(signed_weights=weights, electronic_biases=sum(self.dense_widths[1:]),
                    convolution_biases=0, branch_columns=weights * 2 * material.residual_banks,
                    binary_sites=weights * 2 * material.residual_banks * material.depth,
                    neural_weight_layers=len(self.shapes), residual_matrix_banks=len(self.shapes) * material.residual_banks)


def material_from_dict(value):
    value = dict(value)
    value['layer_order'] = tuple(value['layer_order'])
    return MaterialConfig(**value)


def network_from_dict(value):
    value = dict(value)
    value['dense_widths'] = tuple(value['dense_widths'])
    return NetworkConfig(**value)


def configuration(network=NetworkConfig(), material=MaterialConfig()):
    return dict(network=asdict(network), material=asdict(material), resources=network.resources(material),
                convolution_bias=False, dense_bias='electronic', weight_range=[-1., 1.],
                readout='ideal independent intensity channels; no noise, ADC, or crosstalk')
