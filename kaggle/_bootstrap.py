"""Shared Kaggle bootstrap. Copied into notebooks as a cell."""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_and_prepare_data() -> Path:
    root = Path("/kaggle/input")
    locked = next(root.rglob("c4_calibration_locked.jsonl*"))
    source = locked.parents[2]
    data = Path("/kaggle/working/pa4-data")
    if data.exists():
        shutil.rmtree(data)
    gzip_dirs = {"public_locked", "kv_retrieval", "mqar"}
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        if path.suffix == ".jsonl" and path.parent.name in gzip_dirs:
            target = data / rel.with_suffix(".jsonl.gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as raw, gzip.open(target, "wb") as packed:
                shutil.copyfileobj(raw, packed)
        else:
            target = data / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return data


def clone_branch(branch: str) -> Path:
    repo = Path("/kaggle/working/ATML-PA4")
    if repo.exists():
        shutil.rmtree(repo)
    subprocess.check_call(["git", "clone", "https://github.com/ttqureshi/ATML-PA4.git", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "checkout", "--detach", f"origin/{branch}"])
    return repo


def install_pins() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers==4.57.1",
            "huggingface_hub==0.36.2",
            "accelerate",
            "peft",
            "sentencepiece",
            "protobuf",
            "matplotlib",
        ]
    )


def run_stage(repo: Path, args: list[str], env: dict) -> None:
    print("====", " ".join(args), "====", flush=True)
    completed = subprocess.run([sys.executable, *args], cwd=repo, env=env)
    print("exit", completed.returncode, flush=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
