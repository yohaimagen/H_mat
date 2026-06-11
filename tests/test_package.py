"""Smoke test: the package is importable and exposes a version string."""

import gfcompress


def test_package_importable() -> None:
    assert hasattr(gfcompress, "__version__")
    assert isinstance(gfcompress.__version__, str)
    assert gfcompress.__version__
