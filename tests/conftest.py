"""Shared fixtures. The oracle tests need the public game files.

Set ARC_ENV_DIR to the competition's ``environment_files`` folder; tests that
need it skip when it is absent or the toolkit is not installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DEFAULT_ENV_DIR = Path(r"C:\arc-study\data\environment_files")


@pytest.fixture(scope="session")
def env_dir() -> Path:
    d = Path(os.environ.get("ARC_ENV_DIR", DEFAULT_ENV_DIR))
    if not d.is_dir():
        pytest.skip(f"public game files not found at {d}")
    pytest.importorskip("arc_agi")
    pytest.importorskip("arcengine")
    return d
