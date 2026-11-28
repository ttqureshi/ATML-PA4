"""Locked-split loading. Test files stay closed until an allocation is frozen."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

from shared.paths import DATA_REPO_COMMIT, course_data_root
from task1.src.constants import TEST_FILENAMES


class TestFileGate:
    """Blocks WikiText-2, C4 validation, and HellaSwag until selection is finished."""

    def __init__(self) -> None:
        self.frozen = False
        self.note = ""

    def freeze(self, note: str) -> None:
        self.frozen = True
        self.note = note

    def check(self, path: Path) -> None:
        if path.name in TEST_FILENAMES and not self.frozen:
            raise RuntimeError(
                f"Refusing to open test file {path.name} before the allocation is frozen."
            )


_FILES = {
    "calibration": "c4_calibration_locked.jsonl.gz",
    "development": "c4_development_locked.jsonl.gz",
    "wikitext": "wikitext2_test_windows_locked.jsonl.gz",
    "c4_validation": "c4_validation_windows_locked.jsonl.gz",
    "hellaswag": "hellaswag_val_locked.jsonl.gz",
}


def locked_path(kind: str) -> Path:
    if kind not in _FILES:
        raise KeyError(kind)
    return course_data_root() / "data" / "public_locked" / _FILES[kind]


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def _missing(path: Path) -> SystemExit:
    return SystemExit(
        f"Missing {path}. Clone the course data repo at commit {DATA_REPO_COMMIT} "
        "and materialize public locks with the course setup_data.py. "
        "Task 1 does not download data and does not run setup_data.py."
    )


def require_locked(kinds: list[str], gate: TestFileGate) -> None:
    """Fail before any weight load if a locked split is absent."""
    for kind in kinds:
        path = locked_path(kind)
        gate.check(path)
        if not path.is_file():
            raise _missing(path)


def require_present(kinds: list[str]) -> None:
    """Fail fast when a locked split is absent. Only checks existence; it never opens a test file."""
    for kind in kinds:
        path = locked_path(kind)
        if not path.is_file():
            raise _missing(path)


def load_locked(kind: str, gate: TestFileGate) -> list[dict]:
    path = locked_path(kind)
    gate.check(path)
    if not path.is_file():
        raise _missing(path)
    return read_jsonl(path)


def token_rows(records: list[dict]) -> list[list[int]]:
    rows = []
    for record in records:
        ids = [int(token) for token in record["token_ids"]]
        if len(ids) < 2:
            raise ValueError("a calibration or development sequence needs at least two tokens")
        rows.append(ids)
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError("expected every sequence in this split to have the same length")
    return rows


def _json_default(value: object):
    import numpy as np

    if isinstance(value, np.ndarray):
        raise TypeError("refusing to serialize a raw array; store a summary instead")
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write then rename, so a session killed mid-write never leaves a truncated file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, payload: object) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    _atomic_write_text(path, text)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, sort_keys=True, default=_json_default) + "\n" for row in rows)
    _atomic_write_text(path, text)
