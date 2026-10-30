from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))


@pytest.fixture
def run_path(tmp_path: Path) -> Path:
    return tmp_path / "run.jsonl"
