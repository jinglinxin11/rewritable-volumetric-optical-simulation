"""Statistical reporting tests; no download or training required."""
import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile
import numpy as np
from supplementary_reporting import classification_metrics, confirmation_records, load_tables


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

    def test_unlisted_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'checkpoint.pt').write_bytes(b'corrupt checkpoint')
            (path / 'manifest.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'coverage'):
                load_tables(path)

    def test_required_file_cannot_be_removed_with_its_manifest_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'manifest.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Missing required'):
                load_tables(path, required_files={'checkpoint.pt'})

    def test_duplicate_manifest_path_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            data = b'evidence'
            entry = {'file': 'evidence.bin', 'sha256': hashlib.sha256(data).hexdigest()}
            (path / 'evidence.bin').write_bytes(data)
            (path / 'manifest.json').write_text(json.dumps([entry, entry]))
            with self.assertRaisesRegex(ValueError, 'Duplicate manifest'):
                load_tables(path)

    def test_size_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            data = b'evidence'
            (path / 'evidence.bin').write_bytes(data)
            (path / 'manifest.json').write_text(json.dumps([{
                'file': 'evidence.bin', 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': 1}]))
            with self.assertRaisesRegex(ValueError, 'Size mismatch'):
                load_tables(path)

    def test_duplicate_confirmation_record_rejected(self):
        rows = [{'architecture': arch, 'seed': seed}
                for arch in ('c3_k3_s1', 'c12_k5_s1') for seed in (3, 4, 5)]
        self.assertEqual(len(confirmation_records(rows)), 6)
        with self.assertRaisesRegex(ValueError, 'six unique'):
            confirmation_records(rows + [rows[0]])
        rows[-1] = rows[0]
        with self.assertRaisesRegex(ValueError, 'six unique'):
            confirmation_records(rows)


if __name__ == "__main__":
    unittest.main()
