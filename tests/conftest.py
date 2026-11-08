from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from caliper.config import DEFAULTS, Config  # noqa: E402
from caliper.storage import Store  # noqa: E402


@pytest.fixture()
def config(tmp_path: Path) -> Config:
    import copy

    raw = copy.deepcopy(DEFAULTS)
    raw["paths"]["data_dir"] = str(tmp_path / "data")
    return Config(raw=raw)


@pytest.fixture()
def store(config: Config) -> Store:
    with Store(config.db_path) as s:
        yield s
