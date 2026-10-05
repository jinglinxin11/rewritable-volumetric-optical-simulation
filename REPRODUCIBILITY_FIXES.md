# Reproduction integrity fixes, 5 October 2026

This additive maintenance revision does not change the frozen training kernels,
architecture, dataset split, selection procedure or checkpoint weights. Existing
v1.0.0 and v1.1.0 tags remain historical study records, not movable aliases for
this maintenance revision.

The source loader requires one manifest entry per actual input file, rejects
duplicate ZIP members and manifest paths, and verifies optional byte sizes as
well as SHA-256 values. The reporting and replay entry points declare the files
they actually require. Missing files cannot evade validation by removing their
manifest entries. Confirmation statistics require exactly six unique records.

The published Supplementary Data 1 archive is accepted without changing it.
Reproduction remains the same source-table reporting and archived-checkpoint
replay documented in SUPPLEMENTARY_REVISION_20261004.md. Replay is not retraining.

Run `python -m unittest discover -p "test_*.py" -v`, then the documented reporting
and checkpoint-replay commands with a new output directory.
