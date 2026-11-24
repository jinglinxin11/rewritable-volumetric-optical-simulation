# Recorded ideal residual3 models

Architecture: three 5x5 convolution kernels, no convolution bias, 3x3 adaptive
average pooling, and 27 -> 32 -> 16 -> 10 dense layers with electronic biases.
Every weight matrix is compiled into three differential ten-layer banks.

| Seed | Reference MNIST test | Compiled ideal test | Different predictions |
| --- | ---: | ---: | ---: |
| 0 | 95.00% | 95.00% | 0 |
| 1 | 94.96% | 94.96% | 0 |
| 2 | 94.88% | 94.88% | 0 |

Mean: 94.9467%, population standard deviation: 0.0499 percentage points.
Training: 20 epochs, Adam, learning rate 0.005, batch size 512, original fixed
50k/10k/10k split. Select the earliest maximum validation checkpoint. This is
offline bounded-float training followed by residual encoding, not binary-site
training or hardware verification. MNIST test data have been reused in this project.

The NPZ files are bundled in `src/residual_material_cnn/assets/models/` and installed
as package resources. This directory holds the unchanged recorded reports.
After `python -m pip install -e .`, use:

```bash
python -m residual_material_cnn evaluate --example-seed 0 --data /path/to/data/MNIST/raw --split test
python -m residual_material_cnn export --example-seed 0 --output outputs/new_masks
```

The NPZ files contain counts, residual scales, electronic biases and metadata;
there are no original floating-point weight matrices. NumPy loads them with
`allow_pickle=False`. See `summary.json` for coefficient and normalized clean MAC
error definitions, and `REPORT_CN.md` for the recorded experiment report.

The separate 18.55-bit benchmark is a fixed-span, noise-free random-matrix RMSE
comparison. It is not this network's hardware precision. Existing historical
v1.0.0/v1.1.0 manuscript releases are not modified by this experiment.
