from __future__ import annotations

from pathlib import Path

import pytest

from caliper.config import DEFAULTS, load_config


def test_defaults_load_without_a_file():
    config = load_config(None)
    assert config.run["repetitions"] == DEFAULTS["run"]["repetitions"]
    assert config.engine("ollama")["base_url"].startswith("http://")


def test_user_file_overrides_only_given_keys(tmp_path: Path):
    path = tmp_path / "caliper.toml"
    path.write_text(
        "[run]\nrepetitions = 7\n\n[safety]\nmin_free_vram_mib = 1000\n\n"
        '[engines.ollama]\nbase_url = "http://127.0.0.1:99999"\n'
    )
    config = load_config(path)
    assert config.run["repetitions"] == 7
    assert config.run["temperature"] == DEFAULTS["run"]["temperature"]
    assert config.safety["min_free_vram_mib"] == 1000
    assert config.safety["max_gpu_utilization_pct"] == DEFAULTS["safety"]["max_gpu_utilization_pct"]
    assert (
        config.engine("ollama")["request_timeout_s"]
        == DEFAULTS["engines"]["ollama"]["request_timeout_s"]
    )


def test_caliper_home_env_overrides_data_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CALIPER_HOME", str(tmp_path / "elsewhere"))
    config = load_config(None)
    assert config.data_dir == (tmp_path / "elsewhere").resolve()
    assert config.db_path.name == "caliper.sqlite"


def test_no_absolute_host_paths_in_defaults():
    assert not Path(DEFAULTS["paths"]["data_dir"]).is_absolute()


def test_the_installed_distribution_version_matches_the_package():
    """These drifted silently: the package said 0.4.0, the distribution 0.1.0.

    Anyone installing from a published repository would have seen the wrong
    number from ``pip show`` and from any tool reading the metadata, while
    ``caliper --version`` said something else. ``pyproject.toml`` now derives
    the version from ``caliper.__version__``; this fails if a literal comes
    back.
    """
    import importlib.metadata as metadata

    import caliper

    try:
        installed = metadata.version("caliper")
    except metadata.PackageNotFoundError:  # pragma: no cover - not installed
        pytest.skip("caliper is not installed in this environment")
    assert installed == caliper.__version__


def test_the_changelog_documents_the_current_version():
    """A published repo whose changelog lags its version is a trust problem."""
    import caliper

    changelog = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    assert f"## {caliper.__version__}" in changelog
