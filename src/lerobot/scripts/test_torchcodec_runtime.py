#!/usr/bin/env python
"""Local runtime check for the TorchCodec / FFmpeg / PyTorch stack."""

from __future__ import annotations

import argparse
import ctypes
import glob
import importlib.util
import os
import sys
from pathlib import Path


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def try_import(name: str):
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "<unknown>")
        origin = getattr(module, "__file__", "<builtin>")
        print(f"{name}: OK version={version} file={origin}")
        return module
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"{name}: FAIL {type(exc).__name__}: {exc}")
        return None


def locate_torchcodec_libs() -> list[Path]:
    spec = importlib.util.find_spec("torchcodec")
    if spec is None or spec.origin is None:
        return []
    package_dir = Path(spec.origin).resolve().parent
    return sorted(Path(p) for p in glob.glob(str(package_dir / "libtorchcodec_core*.so")))


def try_load_shared_object(path: Path) -> None:
    try:
        ctypes.CDLL(str(path))
        print(f"shared-lib: OK {path}")
    except OSError as exc:
        print(f"shared-lib: FAIL {path} -> {exc}")


def decode_sample(video: Path) -> None:
    from lerobot.datasets.video_utils import decode_video_frames

    frames = decode_video_frames(video, [0.0, 1.0 / 30.0, 2.0 / 30.0], tolerance_frames=1, backend="torchcodec")
    print(f"decode: OK video={video} shape={tuple(frames.shape)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=None, help="Optional local mp4 path for a real decode test.")
    args = parser.parse_args()

    print_header("Environment")
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"cwd: {os.getcwd()}")
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    print(f"LIBRARY_PATH: {os.environ.get('LIBRARY_PATH')}")

    print_header("Python Packages")
    torch = try_import("torch")
    try_import("torchvision")
    try_import("diffusers")
    try_import("transformers")

    spec = importlib.util.find_spec("torchcodec")
    print(f"torchcodec_spec: {getattr(spec, 'origin', None)}")

    if torch is not None:
        torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
        print(f"torch_lib_dir: {torch_lib_dir}")
        print(f"torch_lib_dir_exists: {torch_lib_dir.exists()}")
        print(f"libavutil_matches: {glob.glob(str(Path(sys.prefix) / 'lib' / 'libavutil.so*'))}")

    print_header("TorchCodec Shared Libraries")
    libs = locate_torchcodec_libs()
    if not libs:
        print("No libtorchcodec_core*.so found.")
    for lib in libs:
        try_load_shared_object(lib)

    print_header("TorchCodec Import")
    torchcodec = try_import("torchcodec")
    if torchcodec is None:
        return 1

    if args.video is not None:
        print_header("Decode Sample")
        if not args.video.exists():
            print(f"decode: FAIL missing video {args.video}")
            return 1
        try:
            decode_sample(args.video)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"decode: FAIL {type(exc).__name__}: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
