#!/usr/bin/env python
"""Validate the full LoLA train dataloader path on a LeRobot-format dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lola_dataset import LoLADataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola.configuration_lola import LoLAConfig


def build_config_and_dataset(args):
    dataset_metadata = LeRobotDatasetMetadata(repo_id=f"local/{Path(args.dataset_root).name}", root=args.dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    action_dim = features["action"].shape[0]
    fps = dataset_metadata.fps

    config = LoLAConfig(
        vlm_model_name="Qwen/Qwen3.5-4B",
        vlm_path="/tmp/dummy",
        action_dim=action_dim,
        action_chunk_size=args.action_chunk_size,
        pred_chunk_size=args.pred_chunk_size,
        n_obs_steps=args.n_obs_steps,
        input_features={k: v for k, v in features.items() if v.type != FeatureType.ACTION},
        output_features={k: v for k, v in features.items() if v.type == FeatureType.ACTION},
        train_vlm=False,
        load_full_history=True,
        max_history_length=args.max_history_length,
        history_padding_side=args.history_padding_side,
    )

    delta_timestamps = {
        "observation.state": [i / fps for i in config.observation_delta_indices],
        "action": [i / fps for i in config.action_delta_indices],
    }
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    dataset = LoLADataset(
        repo_id=f"local/{Path(args.dataset_root).name}",
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        history_padding_side=args.history_padding_side,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )

    return dataset_metadata, dataset


def collate_lola(batch):
    result = {}
    variable_length_keys = {"hist_actions_full", "hist_actions_mask"}

    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key == "task":
            result[key] = values
        elif key in variable_length_keys and isinstance(values[0], torch.Tensor):
            max_len = max(v.shape[0] for v in values)
            padded_values = []
            for v in values:
                if v.shape[0] < max_len:
                    pad_len = max_len - v.shape[0]
                    if key == "hist_actions_full":
                        padding = torch.zeros(pad_len, v.shape[1], dtype=v.dtype)
                    else:
                        padding = torch.zeros(pad_len, dtype=v.dtype)
                    v = torch.cat([padding, v], dim=0)
                padded_values.append(v)
            result[key] = torch.stack(padded_values)
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def save_batch_images(batch, camera_keys, output_dir: Path, max_items: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for cam_key in camera_keys:
        if cam_key not in batch or not isinstance(batch[cam_key], torch.Tensor):
            continue
        batch_tensor = batch[cam_key]
        limit = min(max_items, batch_tensor.shape[0])
        for i in range(limit):
            frame = batch_tensor[i, 0] if batch_tensor.ndim == 5 else batch_tensor[i]
            image = (frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            path = output_dir / f"{cam_key.replace('.', '_')}_sample_{i:02d}.png"
            Image.fromarray(image).save(path)
            saved.append(str(path))
    return saved


def check_batch_for_nan_inf(batch) -> list[str]:
    issues = []
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if torch.isnan(value).any():
                issues.append(f"{key}: NaN detected")
            if torch.isinf(value).any():
                issues.append(f"{key}: Inf detected")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--video_backend", choices=["torchcodec", "pyav", "video_reader"], default="torchcodec")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_history_length", type=int, default=128)
    parser.add_argument("--history_padding_side", choices=["left", "right"], default="left")
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--log_every_batches", type=int, default=100)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--save_images_dir", type=str, default=None)
    parser.add_argument("--save_image_count", type=int, default=3)
    parser.add_argument("--report_json", type=str, default=None)
    args = parser.parse_args()

    metadata, dataset = build_config_and_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_lola,
        pin_memory=False,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    total_batches = len(loader) if args.limit_batches is None else min(len(loader), args.limit_batches)
    total_samples = min(len(dataset), total_batches * args.batch_size)

    print("=" * 60)
    print("LoLA Train DataLoader Full Validation")
    print("=" * 60)
    print(f"dataset_root: {args.dataset_root}")
    print(f"dataset_len: {len(dataset)}")
    print(f"camera_keys: {metadata.camera_keys}")
    print(f"video_backend: {args.video_backend}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"total_batches: {total_batches}")
    print(f"total_samples(approx): {total_samples}")

    saved_images = []
    nan_inf_issues = []
    start = time.monotonic()
    processed_batches = 0
    processed_samples = 0

    for batch_idx, batch in enumerate(loader):
        if args.limit_batches is not None and batch_idx >= args.limit_batches:
            break

        processed_batches += 1
        batch_size = None
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                batch_size = int(value.shape[0])
                break
        if batch_size is None:
            batch_size = args.batch_size
        processed_samples += batch_size

        if processed_batches == 1 and args.save_images_dir:
            saved_images = save_batch_images(
                batch,
                metadata.camera_keys,
                Path(args.save_images_dir),
                max_items=args.save_image_count,
            )

        issues = check_batch_for_nan_inf(batch)
        if issues:
            nan_inf_issues.extend([f"batch {batch_idx}: {msg}" for msg in issues])
            break

        if processed_batches == 1 or processed_batches % max(1, args.log_every_batches) == 0:
            elapsed = time.monotonic() - start
            speed = processed_batches / max(elapsed, 1e-6)
            remaining = max(0, total_batches - processed_batches)
            eta_seconds = remaining / max(speed, 1e-6)
            print(
                f"[{processed_batches / max(total_batches, 1) * 100:5.1f}%] "
                f"batch {processed_batches}/{total_batches}, "
                f"samples {processed_samples}, "
                f"{speed:.2f} batch/s, ETA {eta_seconds/60:.1f} min"
            )

    elapsed = time.monotonic() - start
    summary = {
        "dataset_root": args.dataset_root,
        "dataset_len": len(dataset),
        "camera_keys": list(metadata.camera_keys),
        "video_backend": args.video_backend,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "total_batches": total_batches,
        "processed_batches": processed_batches,
        "processed_samples": processed_samples,
        "elapsed_seconds": elapsed,
        "batches_per_second": processed_batches / max(elapsed, 1e-6),
        "saved_images": saved_images,
        "nan_inf_issues": nan_inf_issues,
        "status": "passed" if not nan_inf_issues and processed_batches == total_batches else "failed",
    }

    print(json.dumps(summary, indent=2))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(summary, indent=2))
        print(f"report_json: {args.report_json}")

    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
