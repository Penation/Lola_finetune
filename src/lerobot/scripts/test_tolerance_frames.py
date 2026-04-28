#!/usr/bin/env python
"""Inspect how query timestamps map to decoded frames under a frame tolerance."""

from __future__ import annotations

import argparse
from pathlib import Path


def resolve_video(dataset_root: Path | None, video: Path | None) -> Path:
    if video is not None:
        return video
    if dataset_root is None:
        raise ValueError("Provide either --dataset-root or --video.")
    matches = sorted(dataset_root.glob("videos/chunk-*/observation.images.primary/*.mp4"))
    if not matches:
        raise FileNotFoundError(f"No CALVIN/LeRobot videos found under {dataset_root}")
    return matches[0]


def load_timestamps(video: Path, backend: str) -> torch.Tensor:
    import torch
    if backend == "torchcodec":
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(video), seek_mode="approximate")
        metadata = decoder.metadata
        if metadata.average_fps is None or metadata.num_frames is None:
            raise RuntimeError(f"torchcodec metadata is incomplete for video={video}")
        return torch.arange(int(metadata.num_frames), dtype=torch.float32) / float(metadata.average_fps)

    import torchvision
    torchvision.set_video_backend(backend)
    timestamps, _ = torchvision.io.read_video_timestamps(str(video), pts_unit="sec")
    if not timestamps:
        raise RuntimeError(f"No timestamps returned from backend={backend} for video={video}")
    return torch.tensor([float(ts) for ts in timestamps], dtype=torch.float32)


def build_queries(loaded_ts: torch.Tensor, query_count: int, start_frame: int, frame_jitter: float) -> torch.Tensor:
    import torch

    if len(loaded_ts) < start_frame + query_count:
        raise ValueError("Video is too short for the requested query window.")
    base = loaded_ts[start_frame : start_frame + query_count].clone()
    if frame_jitter != 0.0 and len(loaded_ts) > 1:
        frame_interval = torch.median(torch.diff(loaded_ts)).item()
        base = base + float(frame_jitter) * frame_interval
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset root containing videos/chunk-*.")
    parser.add_argument("--video", type=Path, default=None, help="Explicit video path.")
    parser.add_argument("--backend", choices=["pyav", "video_reader", "torchcodec"], default="pyav")
    parser.add_argument("--query-count", type=int, default=8)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--frame-jitter",
        type=float,
        default=0.0,
        help="Shift each query by this many frame intervals. Example: 0.49 stays within one frame, 1.2 should fail.",
    )
    parser.add_argument("--tolerance-frames", type=int, default=1)
    parser.add_argument("--run-decode", action="store_true", help="Also call decode_video_frames() with the queries.")
    args = parser.parse_args()

    video = resolve_video(args.dataset_root, args.video)
    import torch

    loaded_ts = load_timestamps(video, args.backend)
    queries = build_queries(loaded_ts, args.query_count, args.start_frame, args.frame_jitter)

    frame_interval = torch.median(torch.diff(loaded_ts)).item() if len(loaded_ts) > 1 else 0.0
    dist = torch.cdist(queries[:, None], loaded_ts[:, None], p=1)
    min_dist_s, closest_idx = dist.min(1)
    frame_distance = min_dist_s / frame_interval if frame_interval > 0 else torch.zeros_like(min_dist_s)
    within_tol = frame_distance <= args.tolerance_frames

    print(f"video: {video}")
    print(f"backend: {args.backend}")
    print(f"num_loaded_frames: {len(loaded_ts)}")
    print(f"frame_interval_s: {frame_interval:.8f}")
    print(f"tolerance_frames: {args.tolerance_frames}")
    print(f"frame_jitter: {args.frame_jitter}")

    for i, (query_ts, chosen_idx, chosen_ts, dist_s, dist_frames, ok) in enumerate(
        zip(
            queries.tolist(),
            closest_idx.tolist(),
            loaded_ts[closest_idx].tolist(),
            min_dist_s.tolist(),
            frame_distance.tolist(),
            within_tol.tolist(),
        )
    ):
        print(
            f"[{i}] query={query_ts:.8f}s closest_idx={chosen_idx} closest_ts={chosen_ts:.8f}s "
            f"dist_s={dist_s:.8f} dist_frames={dist_frames:.4f} within_tol={ok}"
        )

    if args.run_decode:
        from lerobot.datasets.video_utils import decode_video_frames

        try:
            frames = decode_video_frames(
                video,
                queries.tolist(),
                tolerance_frames=args.tolerance_frames,
                backend=args.backend,
            )
            print(f"decode: OK shape={tuple(frames.shape)}")
        except Exception as exc:
            print(f"decode: FAIL {type(exc).__name__}: {exc}")
            return 1

    max_frame_distance = frame_distance.max().item() if len(frame_distance) > 0 else 0.0
    print(f"max_frame_distance: {max_frame_distance:.4f}")
    return 0 if bool(within_tol.all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
