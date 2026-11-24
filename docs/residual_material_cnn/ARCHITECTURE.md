# Architecture and Contracts

## Layout

```text
pyproject.toml                 install metadata, CLI, Ruff, pytest, coverage
src/residual_material_cnn/
  config.py                    immutable scientific/training configuration
  encoding.py                  counts, codebook, residual scales and masks
  readout.py                   ideal differential intensity reconstruction
  preprocessing.py             validated image-to-patch transformation
  network.py                   float reference and compiled inference
  data.py                      local MNIST integrity and fixed splits
  training.py                  optimization and validation selection
  artifacts.py                 schema-1 models, JSON and mask exports
  metrics.py                   coefficient/MAC error and benchmark
  pipeline.py                  train, lock, compile and compare all seeds
  resources.py                 installed split/model resource access
  cli.py / __main__.py          command dispatch and process entry points
  assets/                      fixed split and three compiled examples
tests/unit/                    deterministic local contracts
tests/integration/             model/data/CLI/workflow contracts
examples/residual_material_cnn/ unchanged recorded results
tools/check_wheel.py           installed-wheel check outside source tree
```

## Dependency Direction

CLI coordinates resources, data, artifacts and pipeline; it does not implement
numerical operations. Pipeline uses training, metrics and artifact APIs.
Networks use preprocessing and readout; readout uses encoding; encoding depends
on configuration. Data preprocessing does not import a network. Numerical
modules do not import CLI or historical `scheme_a_*` modules.

## Scientific Invariants

Input is normalized 14x14 grayscale. Three valid 5x5 kernels have no bias;
electronic ReLU and 3x3 adaptive pooling produce 27 features. Dense layers are
27 -> 32 -> 16 -> 10, retaining 58 electronic bias values. Four weight matrices
use three differential residual banks per weight. Offline bounded-float training
is followed by compilation; this is not binary-site training or noisy hardware.

The compiled model stores integer counts, residual scales, configuration and
electronic biases, not the original float weight matrices. The NPZ schema stays
at 1 and loads with `allow_pickle=False`. Invalid shapes, ranges and unsupported
metadata are rejected. Masks use axes `[bank, depth, branch, output, input]`.

## Operations and Evidence

- Installation is required for the `src/` layout. Both module and console
  entry points call the same `cli.main(argv)` returning an exit code.
- CLI stdout is JSON; progress uses stderr logging. Expected input failures
  exit 2; runtime failures exit 1; successful commands exit 0.
- Resource helpers resolve installed package files, not repository-relative
  paths. User data/output paths are interpreted relative to the caller's cwd.
- New experiment and export directories must not exist. Compiled models use
  exclusive creation; internal JSON records use atomic replacement. A failed
  experiment may retain partial artifacts for diagnosis; do not treat them as
  a completed run or reuse its directory.
- All seed checkpoints are selected by validation and locked before test
  evaluation. Runs record data/package hashes, environment and configuration.
  Source hashes are checked before/after execution; `verification.json`
  indicates completion of that check.
- Recorded example results remain historical evidence. Noise-free MAC bits,
  classification accuracy and hardware ENOB are distinct metrics. No hardware
  noise, exposure calibration or wave propagation is implied by packaging.

The root manuscript modules are deliberately not part of the wheel. Their
commands remain run-from-checkout workflows under the historical documentation.
