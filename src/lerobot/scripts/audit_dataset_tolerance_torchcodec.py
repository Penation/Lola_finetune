#!/usr/bin/env python
"""Audit all dataset samples against torchcodec-decoded frame timestamps."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def open_decoder(video_path: Path):
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path), seek_mode="approximate")
    metadata = decoder.metadata
    if metadata.average_fps is None or metadata.num_frames is None:
        raise RuntimeError(f"Incomplete torchcodec metadata for {video_path}")
    return decoder, float(metadata.average_fps), int(metadata.num_frames)


def nearest_frame_distance_torchcodec(
    decoder,
    query_ts: np.ndarray,
    average_fps: float,
    num_frames: int,
    batch_size: int,
) -> np.ndarray:
    frame_distance_batches = []
    for start in range(0, len(query_ts), batch_size):
        chunk = query_ts[start : start + batch_size]
        frame_indices = np.rint(chunk * average_fps).astype(np.int64)
        clamped = np.clip(frame_indices, 0, num_frames - 1)
        frame_batch = decoder.get_frames_at(indices=clamped.tolist())
        loaded_ts = np.asarray(frame_batch.pts_seconds, dtype=np.float64)
        frame_distance = np.abs(chunk - loaded_ts) * average_fps
        frame_distance_batches.append(frame_distance)
    if not frame_distance_batches:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(frame_distance_batches)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tolerance-frames", type=float, default=1.0)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--decode-batch-size", type=int, default=2048)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    meta = LeRobotDatasetMetadata(repo_id=f"local/{dataset_root.name}", root=str(dataset_root))

    episode_file_map: dict[int, Path] = {}
    for ep_file in (dataset_root / "data").glob("chunk-*/episode_*.parquet"):
        ep_idx = int(ep_file.stem.split("_")[1])
        episode_file_map[ep_idx] = ep_file

    decoder_cache: dict[str, tuple[object, float, int]] = {}
    total_episodes = meta.total_episodes if args.limit_episodes is None else min(meta.total_episodes, args.limit_episodes)
    episodes = range(total_episodes)

    summary = {
        "dataset_root": str(dataset_root),
        "fps": float(meta.fps),
        "tolerance_frames": args.tolerance_frames,
        "episodes_checked": 0,
        "samples_checked": 0,
        "violations": 0,
        "max_frame_distance": 0.0,
        "per_camera": {},
        "top_violations": [],
    }
    per_camera = defaultdict(lambda: {"samples_checked": 0, "violations": 0, "max_frame_distance": 0.0})

    for ep_idx in tqdm(episodes, desc="audit episodes"):
        ep = meta.episodes[ep_idx]
        ep_file = episode_file_map.get(ep_idx)
        if ep_file is None:
            raise FileNotFoundError(f"Missing parquet for episode {ep_idx}")

        table = pq.read_table(ep_file, columns=["timestamp"])
        rel_ts = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
        summary["episodes_checked"] += 1

        for cam_key in meta.video_keys:
            rel_video_path = meta.get_video_file_path(ep_idx, cam_key)
            video_path = dataset_root / rel_video_path
            cache_key = str(video_path)
            if cache_key not in decoder_cache:
                decoder_cache[cache_key] = open_decoder(video_path)
            decoder, average_fps, num_frames = decoder_cache[cache_key]

            abs_ts = rel_ts + float(ep[f"videos/{cam_key}/from_timestamp"])
            frame_distance = nearest_frame_distance_torchcodec(
                decoder,
                abs_ts,
                average_fps=average_fps,
                num_frames=num_frames,
                batch_size=args.decode_batch_size,
            )
            violations = frame_distance > args.tolerance_frames

            sample_count = int(frame_distance.size)
            violation_count = int(violations.sum())
            max_frame_distance = float(frame_distance.max()) if sample_count else 0.0

            summary["samples_checked"] += sample_count
            summary["violations"] += violation_count
            summary["max_frame_distance"] = max(summary["max_frame_distance"], max_frame_distance)

            cam_stats = per_camera[cam_key]
            cam_stats["samples_checked"] += sample_count
            cam_stats["violations"] += violation_count
            cam_stats["max_frame_distance"] = max(cam_stats["max_frame_distance"], max_frame_distance)

            if violation_count > 0 and len(summary["top_violations"]) < 20:
                bad_idx = np.where(violations)[0][:20]
                for local_idx in bad_idx:
                    summary["top_violations"].append(
                        {
                            "episode_index": ep_idx,
                            "camera": cam_key,
                            "sample_offset": int(local_idx),
                            "relative_timestamp": float(rel_ts[local_idx]),
                            "absolute_timestamp": float(abs_ts[local_idx]),
                            "frame_distance": float(frame_distance[local_idx]),
                            "video_path": str(video_path),
                        }
                    )
                    if len(summary["top_violations"]) >= 20:
                        break

    summary["per_camera"] = dict(per_camera)
    print(json.dumps(summary, indent=2))

    if args.report_json is not None:
        args.report_json.write_text(json.dumps(summary, indent=2))
        print(f"report_json: {args.report_json}")

    return 0 if summary["violations"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
