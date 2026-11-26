"""Locations for the pinned course data repository. Raw data is never committed."""

from __future__ import annotations

import os
from pathlib import Path

DATA_REPO_URL = "https://github.com/AbDu11aHHH/atml-pa4-datasets.git"
DATA_REPO_COMMIT = "6a0a9ec9da7ef3d313ee3036cfec493886723644"

REPO_ROOT = Path(__file__).resolve().parents[1]


def course_data_root() -> Path:
    override = os.environ.get("PA4_DATA_REPO")
    if override:
        return Path(override)
    return REPO_ROOT / "external" / "atml-pa4-datasets"
