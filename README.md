# Rewritable volumetric optical-weight simulation

Code accompanying Section 2.6 and the Supplementary Information **Rewritable
volumetric weights for multilayer inference**. The manuscript-linked release is
[v1.1.0](https://github.com/jinglinxin11/rewritable-volumetric-optical-simulation/releases/tag/v1.1.0).
The original frozen training release remains available as v1.0.0.
The repository is public and distributed under the [MIT licence](LICENSE).

The selected MNIST model uses twelve 5 x 5 convolution kernels, ten binary
material depth planes and three material dense layers, 108 -> 32 -> 16 -> 10.
It contains 4,428 signed weights, 88,560 binary sites and 70 electronic biases.
This is an independent-intensity-channel simulation, not optical propagation
or a hardware demonstration.

## Reproduction

Use the [reproduction guide](REPRODUCIBILITY.md) for the pinned environment,
demonstration, training commands, output definitions, expected results and
[Supplementary Note-to-code map](REPRODUCIBILITY.md#companion-supplement).

```bash
git clone --branch v1.1.0 --depth 1 https://github.com/jinglinxin11/rewritable-volumetric-optical-simulation.git
cd rewritable-volumetric-optical-simulation
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -p "test_*.py" -v
```

Full selected-model training starts with `python run_simulation.py`. The full
architecture search starts with `python prepare_data.py`, followed by
`python scheme_a_conv_search.py`. Both write numerical outputs only.

## Supplementary reporting update 4 October 2026

The v1.1.0 reporting update preserves the v1.0.0 training kernels and tag.
It adds source-CSV validation, Supplementary Figs. 12/13, Tables 6/7 and optional
exact replay of the six archived confirmation checkpoints. Source data,
checkpoints and supplementary figures are provided as separate
[release assets](https://github.com/jinglinxin11/rewritable-volumetric-optical-simulation/releases/tag/v1.1.0).
See [the update guide](SUPPLEMENTARY_REVISION_20261004.md).

```bash
python -m pip install -r requirements-reporting.txt
python supplementary_reporting.py --data Supplementary_Data_1.zip --output outputs/supplementary_figures
```

## Scope and citation

The repository contains simulation code, tests and reproduction documentation.
The published v1.0.0 release does not contain plotting code. The v1.1.0
update adds plotting code, but not manuscript-generation code, manuscript files,
MNIST archives, trained checkpoints or generated training outputs in the Git tree.
Checkpoints, numerical source data and figures are separate v1.1.0 release assets.
Numerical source data also accompany the paper as **Supplementary Data 1**;
the code regenerates training outputs from public MNIST inputs.

For the manuscript-associated implementation, cite **jinglinxin11. Rewritable
volumetric optical-weight simulation, version 1.1.0. GitHub (2026)** and the
[versioned release](https://github.com/jinglinxin11/rewritable-volumetric-optical-simulation/releases/tag/v1.1.0),
not an unspecified revision of `main`. This release has no DOI assigned.
