#!/usr/bin/env python
"""Reproduce AMLT preflight checks locally before submitting a cluster job."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(name: str, cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n=== {name} ===")
    print("cmd:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--video-backend", choices=["torchcodec", "pyav", "video_reader"], default="torchcodec")
    parser.add_argument("--python", default=sys.executable, help="Python executable from the target conda env.")
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    vlm_path = args.vlm_path.resolve()
    python_exe = args.python

    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {dataset_root / 'meta' / 'info.json'}")
    if not vlm_path.is_dir():
        raise FileNotFoundError(f"Missing VLM directory: {vlm_path}")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    run_step(
        "Import Stack",
        [
            python_exe,
            "-c",
            (
                "import diffusers.models.transformers.transformer_flux2 as flux2; "
                "from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration; "
                "import lerobot.scripts.train_lola_azure as train_mod; "
                "print('flux2_ok', hasattr(flux2, 'Flux2TransformerBlock')); "
                "print('qwen_ok', Qwen3_5ForConditionalGeneration.__name__); "
                "print('train_mod_ok', hasattr(train_mod, 'LoLATrainer'))"
            ),
        ],
        env=env,
    )

    run_step(
        "AMLT Env Check",
        ["bash", str(SCRIPT_DIR / "amlt_env_check.sh"), args.video_backend],
        env=env,
    )

    if not args.skip_decode:
        run_step(
            "AMLT Decode Check",
            ["bash", str(SCRIPT_DIR / "amlt_decode_check.sh"), str(dataset_root), args.video_backend],
            env=env,
        )

    run_step(
        "Bash Syntax",
        ["bash", "-n", str(SCRIPT_DIR / "test_azure.sh")],
        env=env,
    )

    run_step(
        "Train Script Help",
        [python_exe, str(SCRIPT_DIR / "train_lola_azure.py"), "--help"],
        env=env,
    )

    print("\npreflight: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
