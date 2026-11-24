import json
from pathlib import Path

import pytest
import torch

from residual_material_cnn import cli
from residual_material_cnn.data import MNISTData
from residual_material_cnn.resources import bundled_model


def synthetic_data():
    generator = torch.Generator().manual_seed(34)

    def part(count):
        return torch.rand(count, 100, 25, generator=generator), torch.randint(
            10, (count,), generator=generator
        )

    return MNISTData(part(12), part(8), part(8), {"source": "synthetic integration test"})


def test_describe_stdout_is_json(capsys):
    assert cli.main(["describe"]) == 0
    assert json.loads(capsys.readouterr().out)["resources"]["binary_sites"] == 96660


def test_benchmark_writes_json_and_refuses_overwrite(tmp_path, capsys):
    path = tmp_path / "benchmark.json"
    assert cli.main(["benchmark", "--output", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["equivalent_bits"] == pytest.approx(
        18.549528231629655, abs=1e-7
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["benchmark", "--output", str(path)])
    assert exc.value.code == 2


def test_export_packaged_example_from_unrelated_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["export", "--example-seed", "0", "--output", "masks"]) == 0
    assert json.loads(capsys.readouterr().out)["exported"] == "masks"
    assert (tmp_path / "masks/write_masks.npz").is_file()


def test_invalid_example_seed_and_missing_file_are_clear(tmp_path):
    with pytest.raises(ValueError), bundled_model(9):
        pass
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["export", "--model", str(tmp_path / "absent.npz"), "--output", str(tmp_path / "out")]
        )
    assert exc.value.code == 2


def test_evaluate_uses_packaged_or_external_model(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_data", lambda args: synthetic_data())
    for source in (["--example-seed", "0"],):
        assert cli.main(["evaluate", *source]) == 0
        assert 0 <= json.loads(capsys.readouterr().out)["accuracy_pct"] <= 100
    with bundled_model(0) as path:
        assert cli.main(["evaluate", "--model", str(path), "--split", "test"]) == 0
        assert json.loads(capsys.readouterr().out)["split"] == "test"


def test_training_cli_locks_checkpoints_before_test(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_data", lambda args: synthetic_data())
    output = tmp_path / "run"
    assert (
        cli.main(
            ["train", "--output", str(output), "--seeds", "5", "--epochs", "1", "--batch-size", "4"]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["seeds"] == [5]
    locked = json.loads((output / "locked_before_test.json").read_text())
    assert "compiled_test_accuracy_pct" not in locked[0]
    assert (output / "verification.json").is_file()
    with pytest.raises(SystemExit):
        cli.main(["train", "--output", str(output)])


def test_bad_training_options_do_not_create_output(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_data", lambda args: synthetic_data())
    output = tmp_path / "run"
    with pytest.raises(SystemExit):
        cli.main(["train", "--output", str(output), "--epochs", "0"])
    assert not output.exists()


def test_dataset_paths_and_packaged_split(monkeypatch):
    args = cli.build_parser().parse_args(["train", "--output", "out"])
    seen = []
    monkeypatch.setattr(cli, "load_mnist", lambda data, split: seen.append((data, split.is_file())))
    cli.load_data(args)
    assert seen == [(Path("data/MNIST/raw"), True)]


def test_unexpected_runtime_failure_has_no_success_json(monkeypatch, capsys):
    def fail(args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(cli, "dispatch", fail)
    assert cli.main(["describe"]) == 1
    assert not capsys.readouterr().out
