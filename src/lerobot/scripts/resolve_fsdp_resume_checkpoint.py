#!/usr/bin/env python
"""
Resolve a resumable FSDP checkpoint directory from AMLT output storage.

This script handles both:
1. A complete checkpoint directory that already contains `.metadata` and all shards.
2. A legacy split checkpoint where shards were written into sibling timestamp dirs.

For split checkpoints, it assembles a merged directory by symlinking files into
the requested output location, then prints the resolved checkpoint path.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


def shard_indices(step_dir: Path) -> list[int]:
    indices: list[int] = []
    for path in step_dir.iterdir():
        match = re.fullmatch(r"__(\d+)_0\.distcp", path.name)
        if match is None:
            continue
        indices.append(int(match.group(1)))
    return sorted(indices)


def is_complete_checkpoint(step_dir: Path) -> bool:
    if not step_dir.is_dir():
        return False
    if not (step_dir / ".metadata").exists():
        return False

    indices = shard_indices(step_dir)
    if not indices:
        return False
    return indices == list(range(indices[-1] + 1))


def collect_step_dirs(source_root: Path, step_dir_name: str) -> list[Path]:
    if source_root.name == step_dir_name and source_root.is_dir():
        return [source_root]
    return sorted(path for path in source_root.rglob(step_dir_name) if path.is_dir())


def reset_output_dir(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def merge_step_dirs(step_dirs: list[Path], output_dir: Path):
    reset_output_dir(output_dir)

    for step_dir in step_dirs:
        for item in sorted(step_dir.iterdir()):
            target = output_dir / item.name
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(item.resolve())


def main():
    parser = argparse.ArgumentParser(description="Resolve or merge an FSDP checkpoint for resume.")
    parser.add_argument("--source-root", required=True, help="Mounted run root or checkpoint root on AMLT storage.")
    parser.add_argument(
        "--step",
        required=True,
        help="Step directory name or numeric step, e.g. 133968 or step_133968",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to place the merged checkpoint directory if legacy split shards are detected.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    step_dir_name = args.step if str(args.step).startswith("step_") else f"step_{args.step}"
    output_dir = Path(args.output_dir).resolve()

    step_dirs = collect_step_dirs(source_root, step_dir_name)
    if not step_dirs:
        raise FileNotFoundError(f"No checkpoint directories named '{step_dir_name}' were found under {source_root}")

    for step_dir in step_dirs:
        if is_complete_checkpoint(step_dir):
            print(step_dir)
            return

    merge_step_dirs(step_dirs, output_dir)
    if not is_complete_checkpoint(output_dir):
        found = ", ".join(str(path) for path in step_dirs)
        raise RuntimeError(
            f"Failed to assemble a complete checkpoint at {output_dir}. Candidate dirs: {found}"
        )

    print(output_dir)


if __name__ == "__main__":
    main()
