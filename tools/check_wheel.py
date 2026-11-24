"""Install a wheel into a temporary target and exercise it outside the checkout."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="residual-wheel-") as directory:
        root = Path(directory)
        target = root / "installed"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target),
                str(wheel),
            ],
            check=True,
        )
        probe = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import residual_material_cnn
from residual_material_cnn.cli import main
from residual_material_cnn.artifacts import load_compiled
from residual_material_cnn.resources import bundled_model, bundled_split
import numpy as np
assert Path(residual_material_cnn.__file__).is_relative_to(Path(sys.argv[1]))
with bundled_split() as path, np.load(path, allow_pickle=False) as split:
    assert len(split['train']) == 50000 and len(split['validation']) == 10000
with bundled_model(0) as path:
    model = load_compiled(path)
assert model.config.resources(model.material)['binary_sites'] == 96660
assert main(['benchmark']) == 0
"""
        process = subprocess.run(
            [sys.executable, "-I", "-c", probe, str(target)],
            cwd=root,
            check=True,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        result = json.loads(process.stdout)
        if abs(result["equivalent_bits"] - 18.549528231629655) > 1e-7:
            raise RuntimeError("Installed-wheel benchmark changed")
    print("Installed wheel, bundled assets, and isolated benchmark verified.")


if __name__ == "__main__":
    main()
