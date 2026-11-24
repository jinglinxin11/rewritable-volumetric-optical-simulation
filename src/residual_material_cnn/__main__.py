"""Keep python -m residual_material_cnn equivalent to the installed entry point."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
