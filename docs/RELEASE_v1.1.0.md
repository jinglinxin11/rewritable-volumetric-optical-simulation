# v1.1.0 Supplementary reporting and reproducibility update

Adds source-data validation, Supplementary Figs. 12/13 and Tables 6/7 generation,
and exact replay of archived confirmation checkpoints. Updates the current
supplement numbering, test-set reuse documentation and model/geometry/resource
boundaries. The original v1.0.0 tag and four frozen training kernels are unchanged.

Verification: 38 software tests passed; all six archived checkpoints reproduced
their original 10,000-image prediction arrays exactly. Nine numerical source
CSVs are byte-identical to the previous study archive. Reported selected-model
accuracy remains 97.84 +/- 0.28% and macro-F1 remains 97.83 +/- 0.28%, with
population SD across three training seeds. No full model retraining was performed.

Assets:
- Supplementary_Data_1.zip: numerical source tables, data dictionary, SHA-256
  manifest, exact split, documented test history and six archived confirmation
  checkpoints with predictions, histories and logical rewrite logs.
- CNN_Code_Reporting_20261004.zip: versioned code/documentation snapshot.
- CNN_Supplementary_Figures_20261004.zip: standard and red-review figures in
  PDF, SVG, 600-dpi PNG and LZW TIFF formats.

No manuscript DOCX or raw MNIST archive is uploaded. These results remain an
ideal independent-intensity-channel numerical simulation, not measured hardware
classification, endurance, timing or energy performance. The official test set
was reused from earlier analyses and is not a new blind cohort. No DOI is assigned.
