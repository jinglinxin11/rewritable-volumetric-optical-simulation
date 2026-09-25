# Rewritable volumetric optical-weight simulation

Simulation-only Python release for the current Section 2.6 MNIST model. No plotting,
manuscript-generation, presentation, or document-processing code is included.
Datasets, trained checkpoints, manuscript files and generated results are excluded.

## Model

The selected architecture uses a 14 x 14 input, twelve 5 x 5 convolution kernels
(stride 1, no padding), adaptive average pooling to 3 x 3 per channel, and material
dense layers 108 -> 32 -> 16 -> 10. All four weight banks use differential branches
with ten binary depth sites. Site transmission is 1 or 0.9; retained states are
reset and rewritten during training. Activation, pooling, bias and intensity
re-encoding are electronic. The model uses ideal independent intensity channels,
not wave propagation or measured device noise.

There are 4,428 effective signed weights, 88,560 binary sites and 70 electronic
biases. Ten depth planes are shared by four lateral banks, not forty planes.

## Installation

The archived numerical environment used Python 3.13, PyTorch 2.10.0+cpu,
NumPy 2.1.3 and scikit-learn 1.6.1. A CPU installation matching that run is:

```bash
python -m venv .venv
# Activate the environment using the command for your shell.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Run the selected model

```bash
python run_simulation.py --output outputs/selected_model
```

The default protocol trains seeds 3, 4 and 5 for 20 epochs, with Adam learning
rate 0.005 and batch size 512. MNIST is downloaded on first use, checked against
known checksums, and split into 50,000 training, 10,000 validation and 10,000 test
images. Selection uses each run's earliest best-validation checkpoint; all
checkpoints are selected before test evaluation. The output contains numerical
histories, reset logs, checkpoints, predictions and metrics, with no figures.
Training is a substantial CPU workload; this command is not a quick smoke test.

The original archived three-seed mean accuracy was 97.84%, with population SD
0.28 percentage points. This release does not bundle those trained checkpoints
or substitute reported metrics for a new run. Numerical behavior depends on the
execution environment; use the pinned CPU environment for comparisons.

## Reproduce the architecture search

```bash
python prepare_data.py
python scheme_a_conv_search.py --output outputs/architecture_search
```

This runs the complete 12-candidate validation search with seeds 0, 1 and 2,
then fresh-seed confirmation of the winner and the smaller reference using
seeds 3, 4 and 5. It is more expensive than training only the selected model.
To resume completed candidates under the same source/configuration, add
`--resume`. Interrupted individual runs restart; there is no minibatch resume.

## Tests

```bash
python -m unittest discover -p "test_*.py" -v
```

Tests cover binary transmission levels, surrogate gradients, persistent states,
reset/rewrite behavior, differential readout, input encoding, all search-grid
shapes, resource counts and small synthetic train/reload cycles. Tests do not
download MNIST. They are not hardware validation.

## Source layout

- `scheme_a_conv_search.py`: current convolution search, model, device emulator,
  training and checkpoint evaluation.
- `scheme_a_multilayer_head.py`: three material dense layers and shared interfaces.
- `scheme_a_material_head.py`: material classifier and normalization primitives.
- `scheme_a_binary10.py`: binary depth states and differential weight primitives.
- `scheme_a_closed_loop.py`, `scheme_a_round2.py`, `scheme_a_mnist.py`,
  `reproduce.py`: numerical helpers and checksum-verified data loading only.
- `prepare_data.py`: deterministic split preparation for a fresh checkout.
- `run_simulation.py`: fixed-architecture training and numerical evaluation.
- `test_*.py`: simulation regression tests.

The four main simulation modules are byte-identical to the frozen 2026-09-25
MNIST source snapshot. The four older dependency modules were reduced to the
numerical/data routines used by these models; unused historical calibration,
plotting and document processing were removed. Their retained routine bodies
are unchanged. New data-preparation and fixed-model entry points make the
code-only checkout runnable without archived local output folders.
