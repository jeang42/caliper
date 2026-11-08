"""Benchmark task specifications and suite loading.

Suites are TOML files under ``evals/suites/<version>/``. Each suite declares a
``suite_version``; results record it so runs stay comparable only when they
were graded by the same task definitions.

A suite may also *extend* another through a ``suite.toml`` manifest, replacing
whole categories while inheriting the rest. That keeps a successor suite from
copying tasks it does not change, which would drift. The manifest must pin the
base suite's version: if the base is edited afterwards, loading fails loudly
rather than silently changing what the successor measures while its own
``suite_version`` stays put.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CATEGORIES = (
    "coding",
    "reasoning",
    "instruction_following",
    "tool_use",
    "long_context",
    "agent",
)

SUITES_DIR = Path(__file__).parent / "suites"

#: Name of the optional manifest that makes a suite extend another.
MANIFEST = "suite.toml"
MANIFEST_KEYS = frozenset(
    {"suite_version", "extends", "extends_version", "replaces", "description"}
)


@dataclass(slots=True)
class Task:
    id: str
    category: str
    title: str
    prompt: str
    validator: dict[str, Any]
    system: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    think: bool | None = None
    max_output_tokens: int | None = None
    context_length: int | None = None
    temperature: float | None = None
    weight: float = 1.0
    # Long-context tasks can be rendered at several lengths.
    filler_tokens: int | None = None
    tags: list[str] = field(default_factory=list)
    requires_code_execution: bool = False
    source: str = "original"
    license: str = "original content, project license"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Suite:
    name: str
    suite_version: str
    tasks: list[Task]
    description: str = ""

    def by_category(self, categories: list[str] | None = None) -> list[Task]:
        if not categories:
            return list(self.tasks)
        wanted = set(categories)
        return [t for t in self.tasks if t.category in wanted]

    def get(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(task_id)

    @property
    def categories(self) -> list[str]:
        return sorted({t.category for t in self.tasks})


def _load_file(path: Path) -> tuple[str, list[Task]]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    version = str(data.get("suite_version", "0"))
    tasks: list[Task] = []
    for raw in data.get("task", []):
        unknown = set(raw) - {f for f in Task.__slots__}
        if unknown:
            raise ValueError(
                f"{path.name}: task {raw.get('id')} has unknown keys {sorted(unknown)}"
            )
        task = Task(**raw)
        if task.category not in CATEGORIES:
            raise ValueError(f"{path.name}: task {task.id} has unknown category {task.category!r}")
        if "kind" not in task.validator:
            raise ValueError(f"{path.name}: task {task.id} validator is missing 'kind'")
        tasks.append(task)
    return version, tasks


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    unknown = set(data) - MANIFEST_KEYS - {"task"}
    if unknown:
        raise ValueError(f"{path.name}: unknown manifest keys {sorted(unknown)}")
    if data.get("task"):
        raise ValueError(f"{path.name}: the manifest declares the suite, not tasks")
    return data


def _inherited(
    name: str, manifest: dict[str, Any], suites_dir: Path | None, own: list[Task]
) -> list[Task]:
    """Resolve a manifest's ``extends`` into the base tasks this suite keeps."""
    parent = str(manifest["extends"])
    pinned = manifest.get("extends_version")
    if not pinned:
        raise ValueError(f"suite {name} extends {parent} without pinning extends_version")
    base = load_suite(parent, suites_dir=suites_dir)
    if base.suite_version != str(pinned):
        raise ValueError(
            f"suite {name} was composed against {parent} {pinned} but {parent} is now "
            f"{base.suite_version}; review the inherited tasks and update extends_version "
            f"and {name}'s own suite_version together"
        )
    replaces = {str(c) for c in manifest.get("replaces", [])}
    unknown = replaces - set(CATEGORIES)
    if unknown:
        raise ValueError(f"suite {name} replaces unknown categories {sorted(unknown)}")
    provided = {t.category for t in own}
    missing = replaces - provided
    if missing:
        raise ValueError(
            f"suite {name} replaces {sorted(missing)} but provides no task in those "
            f"categories, which would drop them from every run"
        )
    unreplaced = provided - replaces
    if unreplaced:
        raise ValueError(
            f"suite {name} adds tasks in {sorted(unreplaced)} without listing those "
            f"categories in replaces; inherited and new tasks in one category would mix"
        )
    return [t for t in base.tasks if t.category not in replaces]


def load_suite(name: str = "v1", suites_dir: Path | None = None) -> Suite:
    """Load every ``*.toml`` in a suite directory into one :class:`Suite`."""
    base = (suites_dir or SUITES_DIR) / name
    if not base.is_dir():
        raise FileNotFoundError(f"no suite directory {base}")
    manifest: dict[str, Any] = {}
    tasks: list[Task] = []
    versions: set[str] = set()
    for path in sorted(base.glob("*.toml")):
        if path.name == MANIFEST:
            manifest = _load_manifest(path)
            versions.add(str(manifest.get("suite_version", "0")))
            continue
        version, file_tasks = _load_file(path)
        versions.add(version)
        tasks.extend(file_tasks)
    if len(versions) > 1:
        raise ValueError(f"suite {name} declares conflicting suite_version values: {versions}")
    if manifest.get("extends"):
        tasks = _inherited(name, manifest, suites_dir, tasks) + tasks
    ids = [t.id for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"suite {name} has duplicate task ids: {sorted(dupes)}")
    return Suite(
        name=name,
        suite_version=versions.pop() if versions else "0",
        tasks=tasks,
        description=str(manifest.get("description", "")),
    )
