"""MQAR streams. Training examples come from the course generator.

Frozen development and test files are read, never rewritten. ``validate`` from
``scripts/mqar_generator.py`` is the invariant check used before training.
Distance is ``query position - stored source position``.
"""

from __future__ import annotations

import importlib.util
import json
import gzip
from pathlib import Path
from typing import Iterator

from shared.paths import course_data_root

DISTANCE_BINS = ((0, 32), (32, 64), (64, 128), (128, 256))
REGIMES = ("near", "far", "mixed")
LOADS = (4, 8, 16)


def generator_path() -> Path:
    return course_data_root() / "scripts" / "mqar_generator.py"


_GENERATOR = None


def load_generator():
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR
    path = generator_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Course MQAR generator is not at {path}. Clone the data repo under external/."
        )
    spec = importlib.util.spec_from_file_location("pa4_mqar_generator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _GENERATOR = module
    return module


def frozen_path(split: str) -> Path:
    if split not in ("development", "test"):
        raise ValueError("only development and test are frozen files")
    return course_data_root() / "data" / "mqar" / f"mqar_{split}.jsonl.gz"


def read_jsonl_gz(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def distance_from_source(example: dict) -> list[int]:
    distances = []
    for query, source, stored in zip(
        example["query_positions"], example["source_positions"], example["distances"]
    ):
        computed = int(query) - int(source)
        if computed != int(stored):
            raise ValueError(f"{example.get('id')} distance {stored} != query-source {computed}")
        distances.append(computed)
    return distances


def bin_name(distance: int) -> str:
    for low, high in DISTANCE_BINS:
        if low <= distance < high:
            return f"[{low},{high})"
    return "other"


def iter_train(k: int, regime: str, start: int, count: int) -> Iterator[dict]:
    generator = load_generator()
    for index in range(start, start + count):
        example = generator.make_example("train", k, regime, index)
        generator.validate(example)
        distance_from_source(example)
        yield example


def check_generator_invariants(sample: int = 4) -> None:
    """Run the course ``validate`` and the far-distance constraint on a train prefix.

    Far queries must cover more than the single offset 128, including at least
    one distance not divisible by four. This does not write the frozen test file.
    """
    generator = load_generator()
    far_distances: list[int] = []
    for regime in REGIMES:
        for k in LOADS:
            for index in range(sample):
                example = generator.make_example("train", k, regime, index)
                generator.validate(example)
                distances = distance_from_source(example)
                if regime == "near" and any(distance >= 64 for distance in distances):
                    raise AssertionError("near query is outside distance < 64")
                if regime == "far":
                    far_distances.extend(distances)
    if len(set(far_distances)) < 2 or all(distance == 128 for distance in far_distances):
        raise AssertionError("far regime collapsed to a single offset of 128")
    if all(distance % 4 == 0 for distance in far_distances):
        raise AssertionError("far sample has no offset that is not divisible by four")


def load_frozen(split: str, k: int | None = None, regime: str | None = None) -> list[dict]:
    path = frozen_path(split)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen MQAR {split} file is missing: {path}")
    generator = load_generator()
    rows = []
    for example in read_jsonl_gz(path):
        if k is not None and int(example["K"]) != k:
            continue
        if regime is not None and example["regime"] != regime:
            continue
        generator.validate(example)
        distance_from_source(example)
        rows.append(example)
    return rows

