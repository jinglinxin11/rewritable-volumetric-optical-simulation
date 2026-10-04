# CNN supplementary reporting update

Version v1.1.0 adds reporting tools, not a different trained model.
The published v1.0.0 release and frozen training kernels are unchanged.
[Release assets](https://github.com/jinglinxin11/rewritable-volumetric-optical-simulation/releases/tag/v1.1.0)
provide Supplementary Data 1, the code snapshot and supplementary figure exports.
No manuscript DOCX or raw MNIST is uploaded. No DOI is assigned.

## Three different reproduction procedures

1. Validate source CSVs and regenerate Supplementary Figs. 12/13 and Tables 6/7:

```bash
python -m pip install -r requirements-reporting.txt
python supplementary_reporting.py --data Supplementary_Data_1.zip --output outputs/si_figures
```

The reporting layer validates input SHA-256 hashes, all six prediction-derived
accuracies and macro-F1 values, class counts, mean row-normalised confusion,
36 search fits, selected epochs, complete training reset counts, resource totals
and depth encoding. It never trains or selects a different result. Optional
`--red-text` marks all regenerated figure text red for manuscript review.
Tables 6/7 are exported as full-precision CSVs; Word display rounding is separate.

2. Extract Supplementary Data 1 and replay its six original confirmation weights:

```bash
python replay_checkpoints.py --archive path/to/extracted_data --data-dir data/MNIST/raw --output outputs/archived_replay
```

This requires the recorded training environment, verified MNIST archives and
trusted study checkpoints. It checks data/split identity and exact equality to
all six archived prediction arrays. It does not retrain or choose epochs.

3. Retrain: use `run_simulation.py` or `scheme_a_conv_search.py` as documented in
REPRODUCIBILITY.md, always in a new output directory. Retraining is distinct
from checkpoint replay; platform differences can change training trajectories.

## Scientific and historical scope

The current official MNIST test images were used in earlier project analyses.
`provenance/test_history.json` in the revised data archive points to the earlier
multilayer manifest/results and current search selection records. Current
architecture ranking and checkpoint selection use validation accuracy; all six
confirmation selections precede testing. These records do not reconstruct every
historical human decision. No new blind cohort, independent three-cohort test,
ensemble, hardware classification, measured endurance, latency or energy benefit
is claimed.

The ten-site product model assumes site transmission 1/0.9, ideal independent
intensity channels and zero explicit noise/crosstalk. Ten sites give eleven
count-based transmission levels, not 1,024 distinct transmission levels. The
400-micrometre assumed depth spacing is a layout scenario separate from the
approximately 200-micrometre experimental spacing. The 5.54% allocation is an
address fraction, not measured volumetric efficiency.

## Figure mapping

| Output | Source tables | Contents |
| --- | --- | --- |
| Supplementary Fig. 12 | depth_encoding.csv | Assumed branch transmission and signed differential weights |
| Supplementary Fig. 13a | validation_ranking.csv, search_run_metrics.csv | Twelve-architecture validation ranking, mean +/- population SD |
| Supplementary Fig. 13b | confirmation_metrics.csv, test_predictions.csv | Paired seed 3/4/5 test accuracies and summary |
| Supplementary Fig. 13c | selected_training_history.csv | Per-seed sample-weighted cross-entropy, epochs 1-20 |
| Supplementary Fig. 13d | test_predictions.csv, mean_confusion_percent.csv | Mean of separately row-normalised confusion matrices |
| Supplementary Table 6 | confirmation_metrics.csv | Six runs and their validation-selected epochs |
| Supplementary Table 7 | resources.csv | Selected model bank-level weights, sites, biases and branch products |

Figure 13d uses exponent 0.3 for colour display only. Diagonal entries and
off-diagonal entries >=0.5% are annotated; source percentages are unchanged.
Population SD uses ddof=0 and is expressed in percentage points. Macro-F1 is
the equally weighted mean across all ten classes, computed before rounding.

The reporting scripts were exercised with bundled Python 3.12.14, NumPy 2.3.5,
Matplotlib 3.11.2 and Pillow 12.3.0. This reporting environment is not a replacement
for the original Python 3.13.5 / PyTorch 2.10.0+cpu training environment.
