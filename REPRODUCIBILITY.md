# Reproduction guide: Section 2.6 / Supplementary Information

This guide documents release **v1.0.0**. It contains operational details moved
out of the companion supplement; the supplement retains the model equations,
training and selection protocol, statistical definitions, tables and figures.

## Companion supplement

Companion document: **Supplementary Information: Rewritable volumetric weights
for multilayer inference**, GitHub-linked revision of 25 September 2026,
covering Section 2.6 and main-text Figs. 6 and 7. The companion Word/PDF file is
named `Supplementary_Information_Section_2_6`; numerical supporting data are
provided separately as `Supplementary_Data_1.zip`. Neither manuscript file is
hosted in this code-only repository. No publication DOI is assigned here.

| Companion section | Scientific content retained in the supplement | Implementation |
| --- | --- | --- |
| Note 1, equations S1-S3; Fig. S1 | Binary depth states and differential weights | `scheme_a_binary10.Config`, `binary_ste`; `scheme_a_material_head.optical_weights` |
| Note 2, equations S4-S7 | Convolution, overlapping adaptive pooling, intensity encoding and scale restoration | `scheme_a_conv_search.patches`, `Model.logits_from_weights`; `scheme_a_material_head.encode_features`, `material_classifier` |
| Note 3, equations S8-S9 | Persistent state, surrogate gradients and full-bank reset/rewrite | `scheme_a_conv_search.Device.program`, `train`; `scheme_a_multilayer_head.training_logits`; `scheme_a_closed_loop.surrogate_bridge` |
| Note 4, equations S10-S11; Fig. S2; Tables S1-S2 | Fixed split, validation-only selection, fresh-seed confirmation and statistics | `prepare_data.make_split`; `scheme_a_conv_search.GRID`, `summarize`, `main`; `run_simulation.main` |
| Note 5, equations S12-S14; Table S3 | Weight, site and branch-product counts; nominal capacity | `scheme_a_conv_search.resources`, `shapes`, `mask_rows`; geometry accounting below |
| Note 6 | Code and data access | This guide, the release record and `LICENSE` |

The equations, numbering and table references above correspond to the compressed,
GitHub-linked supplement, not its earlier nine-note draft.

## Environment and installation

The release was tested on Windows with Python 3.13.5, PyTorch 2.10.0+cpu,
NumPy 2.1.3 and scikit-learn 1.6.1. No GPU or specialist optical hardware is
required. Linux and macOS execution has not been tested in this release.
Training uses four CPU threads, deterministic PyTorch algorithms and float32.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` in a POSIX shell. Then install:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The CPU wheel index preserves the recorded PyTorch build rather than relying on
an automatically selected CUDA package. A clean installation time has not been
benchmarked; package download time depends on the connection and cache.

## Demonstration and tests

```bash
python demo.py
python -m unittest discover -p "test_*.py" -v
```

The demonstration generates sixteen deterministic synthetic 28 x 28 images,
executes one optimizer update, checks gradients in all four banks, confirms
device/proxy agreement, and verifies that reset removes previously written
states. It does not download MNIST or create output files. Expected invariants
are 88,560 material sites, 68,256 branch products per image, finite nonzero
gradients and successful state-removal/equality checks. Its printed loss is a
synthetic software check, not a classification-performance result. The script
reports elapsed time for the current machine.

The demonstration took approximately 1.6 seconds inside the script (about
5 seconds including interpreter/library startup) in the tested environment.

The 32 regression tests include synthetic train/reload cycles, all twelve search
architectures, differential readout, binary gradients, data split checks and
the fixed-model command. They ran in approximately 3 seconds within the test
runner (approximately 7 seconds including Python/PyTorch startup) on the local
Windows CPU environment; this is not a general hardware-performance benchmark.

## Dataset and exact split

The loader retrieves the official MNIST IDX archives from the public
[PyTorch-hosted mirror](https://ossci-datasets.s3.amazonaws.com/mnist/) and checks
their MD5 sums. It also records SHA-256 hashes in each run's metadata. Downloads
are reused after checksum verification. No images or labels are tracked in Git.

```bash
python prepare_data.py
```

This writes `outputs/scheme_a_mnist/split_indices.npz` with 50,000 training,
10,000 validation and 10,000 official test indices. Training/validation use
`train_test_split(..., test_size=10000, random_state=202609, stratify=labels)`.
An existing mismatched split raises an error rather than being overwritten.
Regenerated indices were compared with the archived study split and matched
element for element in the tested environment.

## Fixed-architecture reproduction

```bash
python run_simulation.py --output outputs/selected_model
```

Defaults are seeds 3, 4 and 5, twenty epochs, batch size 512 and Adam learning
rate 0.005. The selected model uses C=12, K=5, stride=1 and material dense widths
108->32->16->10. Each seed selects its earliest maximum validation checkpoint;
all seed checkpoints are fixed before test evaluation. The command refuses an
existing output directory. It automatically downloads/checks data and saves the
split, so running `prepare_data.py` separately is optional for this command.

A shorter budget is useful for software checks but does not reproduce the
paper protocol:

```bash
python run_simulation.py --output outputs/short_check --epochs 1 --seeds 3
```

Full training is a substantial CPU workload. Its end-to-end duration has not
been benchmarked for this release and is not inferred from the small demo.

## Full validation search and paired comparison

```bash
python prepare_data.py
python scheme_a_conv_search.py --output outputs/architecture_search
```

This runs 36 search fits (12 architectures x seeds 0/1/2), ranks them by mean
best validation accuracy, locks the winner, and trains it and the C=3, K=3,
stride=1 reference with seeds 3/4/5. It then evaluates their selected checkpoints
on the same official test set. The process never uses test accuracy to rank
architectures or pick the best seed. These test images had been used in earlier
project analyses, so this is not a newly collected blind test cohort.

Resume completed candidates with:

```bash
python scheme_a_conv_search.py --output outputs/architecture_search --resume
```

Resume requires identical source hashes and configuration. Incomplete individual
fits restart from their initial seed; there is no minibatch checkpoint resume.
Because dependencies were trimmed for publication, this release has different
source hashes from the original archived project and must use a new output
directory. Do not disable the source-identity check to reuse an old output tree.

## Initialization and reset accounting

For run seed r, the convolution, FC1, FC2 and FC3 latent generators use
r+1000, r+4000, r+5000 and r+6000. Each latent variable is drawn from N(0,0.5^2).
Convolution biases start at zero; dense biases use uniform +/-1/sqrt(fan-in),
with seeds r+7000, r+7001 and r+7002. Batch shuffling uses r+3000. Matched seeds
share image order, not identically shaped parameter tensors.

Each epoch has ceil(50000/512)=98 minibatches, ending with 336 samples. All three
archived selected-model runs rewrote after every optimizer step. Their 1,961
logical reset events equal one initial programming event plus 20 x 98 updates.
The count covers all twenty epochs, not only the selected epoch. If any target
mask changes, all four banks are cleared and all target one-bits rewritten;
identical masks skip programming. This is a simulator event count, not measured
thermal cycling or endurance. Reset-disabled accuracy was not tested.

## Output definitions and expected results

The fixed-model command writes `run_plan.json`, `split_indices.npz`, per-seed
`*_history.json`, `*_events.json`, `*.pt`, `*_result.json` and `*_test.npz`, plus
`selection_before_test.json`, `test_results.json` and `test_summary.json`.
Accuracy in the original per-seed training result is a fraction; fields ending
in `_percent` in the fixed-model test summary are percentages. Population SD
uses denominator R=3 and is reported in percentage points.

The search additionally writes `search_plan.json`, a `source_snapshot/`,
`validation_ranking.json`, `locked_selection_before_confirmation.json`,
`confirmation_selection_before_test.json`, write-mask CSV files and
`confirmation_test_results.json`. `validation_mean` and `test_accuracy` in
these original search outputs are fractions, not percentages.

The archived selected-model seed accuracies were 97.45%, 98.05% and 98.03%
(mean 97.84%, population SD 0.28 percentage points). Mean macro-F1 was 97.83%.
The paired reference accuracies were 94.35%, 94.63% and 94.53% (mean 94.50%).
The expected winner of the recorded search is `c12_k5_s1`. These values identify
the study record; the code does not force a new run to attain them. Floating-point
and platform differences can affect training trajectories.

For main-text Fig. 7, class confusion percentages are averaged across three
row-normalized confusion matrices. A display exponent of 0.3 and off-diagonal
annotation threshold of 0.5% were used in the figure, without changing its
underlying data. Plotting code is outside this repository. Training-loss means
weight each minibatch by sample count; epoch zero has validation but no loss.

## Geometry accounting

The placement scenario assumes a 20 x 20 x 4 mm host, 50 micrometre lateral
pitch and ten depth planes with 400 micrometre spacing. This gives 400 x 400 x 10
=1,600,000 nominal addresses and 88,560/1,600,000=5.535% allocated addresses.
The ten plane centres span 3.6 mm with 0.2 mm end margins. A simple row-bank
layout spans 300 x 118 lateral slots, or 15.00 x 5.90 mm, including unused slots.
Pitch changes mask coordinates but not the trained forward coefficients.

The four banks share ten physical depth planes; sequential neural-network layers
are not additional depth planes. The 400 micrometre spacing is a computational
geometry scenario, separate from the approximately 200 micrometre spacing of
the experimental multilayer records described in the manuscript.

## Provenance and availability

`scheme_a_conv_search.py`, `scheme_a_multilayer_head.py`,
`scheme_a_material_head.py` and `scheme_a_binary10.py` are byte-identical to the
frozen 25 September 2026 MNIST implementation. Four dependency modules were
reduced to their required numerical/data functions without changing those
function bodies. The retired calibration model, figure generation and document
processing were removed. The released code reproduced all six archived
10,000-image test prediction arrays exactly; checkpoints were not retrained
for this check. Unit tests do not establish measured device performance.

Raw MNIST is publicly downloadable. The accompanying **Supplementary Data 1**
contains source CSV tables for reported metrics, epoch histories, mean confusion
percentages, per-image predictions, individual search runs, complete architecture
ranking, resource counts, class counts and depth encoding. Those CSV files and
archived model weights are not hosted in
this source-only repository. Training commands regenerate numerical outputs;
exact replay of a saved model additionally requires its original checkpoint.

The release is MIT-licensed. Version v1.0.0 fixes the manuscript-associated
source; future edits should use new versions rather than moving this tag.
No DOI has been minted. A DOI-bearing archival copy and final journal-specific
code-sharing checks remain publication-stage tasks for the authors.
