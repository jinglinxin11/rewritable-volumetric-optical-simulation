"""Statistical reporting tests; no download or training required."""
import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile
import numpy as np
from supplementary_reporting import classification_metrics, load_tables


class ReportingTests(unittest.TestCase):
    def test_perfect_ten_class_prediction(self):
        result = classification_metrics(np.arange(10), np.arange(10))
        self.assertEqual(result["accuracy_percent"], 100)
        self.assertEqual(result["macro_f1_percent"], 100)
        np.testing.assert_array_equal(result["confusion_counts"], np.eye(10, dtype=int))

    def test_missing_classes_are_finite(self):
        result = classification_metrics(np.array([0, 0]), np.array([0, 0]))
        self.assertEqual(result["macro_f1_percent"], 10)

    def test_input_validation(self):
        for labels, predictions in (([0], [10]), ([0], [0.5]), ([0], [0, 1]), ([], [])):
            with self.assertRaises(ValueError):
                classification_metrics(np.array(labels), np.array(predictions))

    def test_manifest_hash_directory_and_zip(self):
        data = b"value\n1\n"
        manifest = json.dumps([{"file": "example.csv", "sha256": hashlib.sha256(data).hexdigest()}]).encode()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "example.csv").write_bytes(data)
            (path / "manifest.json").write_bytes(manifest)
            tables, _ = load_tables(path)
            self.assertEqual(tables["example.csv"], [{"value": "1"}])
            with ZipFile(path / "data.zip", "w") as archive:
                archive.writestr("example.csv", data)
                archive.writestr("manifest.json", manifest)
            zipped, _ = load_tables(path / "data.zip")
            self.assertEqual(tables, zipped)

    def test_corrupted_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "example.csv").write_text("value\n2\n")
            (path / "manifest.json").write_text(json.dumps([{"file": "example.csv", "sha256": "0" * 64}]))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_tables(path)

    def test_nested_archive_members_are_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "provenance").mkdir()
            data = b"archived evidence"
            (path / "provenance/evidence.json").write_bytes(data)
            (path / "manifest.json").write_text(json.dumps([
                {"file": "provenance/evidence.json", "sha256": hashlib.sha256(data).hexdigest()}]))
            _, members = load_tables(path)
            self.assertEqual(members["provenance/evidence.json"], data)


if __name__ == "__main__":
    unittest.main()
