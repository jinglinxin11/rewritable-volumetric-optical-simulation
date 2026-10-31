"""Fresh-checkout entry points, using synthetic inputs and no network access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import prepare_data
import run_simulation


class EntryPointTests(unittest.TestCase):
    def test_fixed_split_is_stratified_and_disjoint(self):
        split = prepare_data.make_split(np.tile(np.arange(10), 6000))
        self.assertEqual([len(split[k]) for k in ("train", "validation", "test")],
                         [50000, 10000, 10000])
        self.assertEqual(len(np.intersect1d(split["train"], split["validation"])), 0)
        np.testing.assert_array_equal(np.sort(np.r_[split["train"], split["validation"]]),
                                      np.arange(60000))
        np.testing.assert_array_equal(np.bincount(split["validation"] % 10), np.full(10, 1000))

    def test_existing_different_split_is_rejected(self):
        raw = {"train-labels-idx1-ubyte.gz": np.tile(np.arange(10), 6000)}
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "split.npz"
            np.savez_compressed(dest, train=np.arange(10))
            with patch.object(prepare_data, "fetch_data", return_value=(raw, {})):
                with self.assertRaisesRegex(ValueError, "Existing split differs"):
                    prepare_data.prepare(Path(folder), dest)

    def test_selected_runner_synthetic_train_and_evaluate(self):
        rng = np.random.default_rng(12)
        raw = {"train-images-idx3-ubyte.gz": rng.integers(0, 256, (30, 28, 28), dtype=np.uint8),
               "train-labels-idx1-ubyte.gz": np.arange(30) % 10,
               "t10k-images-idx3-ubyte.gz": rng.integers(0, 256, (10, 28, 28), dtype=np.uint8),
               "t10k-labels-idx1-ubyte.gz": np.arange(10)}
        split = {"train": np.arange(20), "validation": np.arange(20, 30), "test": np.arange(10)}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            argv = ["run_simulation.py", "--output", str(output), "--epochs", "1",
                    "--batch-size", "10", "--seeds", "3"]
            with patch("sys.argv", argv), patch.object(run_simulation, "prepare", return_value=(raw, {}, split)):
                run_simulation.main()
            selected = json.loads((output / "selection_before_test.json").read_text())
            self.assertNotIn("test_accuracy_percent", selected[0])
            summary = json.loads((output / "test_summary.json").read_text())
            self.assertEqual(summary["resources"]["material_sites"], 88560)
            with np.load(output / "c12_k5_s1_seed3_test.npz") as saved:
                accuracy = float(100 * np.mean(saved["predictions"] == saved["labels"]))
            self.assertEqual(summary["test_accuracy_percent"]["mean"], accuracy)
            self.assertTrue((output / "c12_k5_s1_seed3.pt").exists())


if __name__ == "__main__":
    unittest.main()
