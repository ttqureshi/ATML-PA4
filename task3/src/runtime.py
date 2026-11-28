"""Where a Task 3 run writes, whether it is a smoke run, and crash-safe JSON output.

A smoke run (PA4_SMOKE=1 or `run.py --smoke <stage>`) writes to
results/task3_smoke/ and checkpoints/task3_smoke/, never to the real
directories, and every JSON object it writes carries a SMOKE_RUN marker.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from shared.paths import REPO_ROOT, course_data_root

SMOKE_ENV = "PA4_SMOKE"
SMOKE_MARKER = "SMOKE_RUN"
SMOKE_NOTE = "smoke run: reduced prompts, tokens and quotas; NOT a result"
REVISION_LOCK = Path("data") / "local_source_revisions.json"


def is_smoke() -> bool:
    return os.environ.get(SMOKE_ENV, "").strip().lower() not in ("", "0", "false", "no")


def results_dir() -> Path:
    return REPO_ROOT / "results" / ("task3_smoke" if is_smoke() else "task3")


def checkpoint_dir() -> Path:
    return REPO_ROOT / "checkpoints" / ("task3_smoke" if is_smoke() else "task3")


def mark(payload):
    """Tag a JSON object written during a smoke run so it cannot pass for a result."""
    if is_smoke() and isinstance(payload, dict) and SMOKE_MARKER not in payload:
        return {SMOKE_MARKER: SMOKE_NOTE, **payload}
    return payload


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value)!r}")


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file and rename, so a killed process never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def dump_json(path: Path, payload, *, indent: int | None = 2) -> None:
    atomic_write_text(path, json.dumps(mark(payload), indent=indent, sort_keys=True, default=json_default) + "\n")


def revision_lock_path(data_root: Path | None = None) -> Path:
    return (course_data_root() if data_root is None else Path(data_root)) / REVISION_LOCK


def pinned_revision(model_name: str, data_root: Path | None = None) -> str | None:
    """Commit hash for a hub repo from the course lock file; None for local paths or without a lock."""
    path = revision_lock_path(data_root)
    if Path(model_name).exists() or not path.is_file():
        return None
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key, repo in lock.get("repos", {}).items():
        if repo == model_name:
            return lock["revisions"][key]
    return None


def with_retries(fn, what: str, attempts: int = 3, wait_s: float = 20.0):
    """Retry hub downloads on network errors (OSError); anything else is raised at once."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except OSError as error:
            if attempt == attempts:
                raise
            print(f"{what}: attempt {attempt} failed ({error!r}); retrying in {wait_s * attempt:.0f}s", flush=True)
            time.sleep(wait_s * attempt)
    raise AssertionError("unreachable")
