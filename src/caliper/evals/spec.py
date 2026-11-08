"""Benchmark task specifications and suite loading.

Suites are TOML files under ``evals/suites/<version>/``. Each suite declares a
``suite_version``; results record it so runs stay comparable only when they
were graded by the same task definitions.
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


def load_suite(name: str = "v1", suites_dir: Path | None = None) -> Suite:
    """Load every ``*.toml`` in a suite directory into one :class:`Suite`."""
    base = (suites_dir or SUITES_DIR) / name
    if not base.is_dir():
        raise FileNotFoundError(f"no suite directory {base}")
    tasks: list[Task] = []
    versions: set[str] = set()
    for path in sorted(base.glob("*.toml")):
        version, file_tasks = _load_file(path)
        versions.add(version)
        tasks.extend(file_tasks)
    if len(versions) > 1:
        raise ValueError(f"suite {name} declares conflicting suite_version values: {versions}")
    ids = [t.id for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"suite {name} has duplicate task ids: {sorted(dupes)}")
    return Suite(name=name, suite_version=versions.pop() if versions else "0", tasks=tasks)
