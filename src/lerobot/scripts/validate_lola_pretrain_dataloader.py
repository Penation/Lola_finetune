#!/usr/bin/env python
"""
Validate LoLAPretrainDataset (map-style) + DataLoader correctness.

Tests:
1. Dataset creation + metadata sanity
2. Multi-worker DataLoader traversal (shuffle, persistent_workers)
3. Index uniqueness & correctness (no duplicate data, same idx returns same data)
4. collate_fn correctness with real data (variable-length padding, camera lists)
5. Video decode correctness (PIL quality, invalid cameras, resolution check)
   + save sample images to disk for visual inspection

Usage:
    python src/lerobot/scripts/validate_lola_pretrain_dataloader.py \
        --dataset_root /path/to/dataset \
        --dataset_to_episodes_path /path/to/dataset_to_episodes.json \
        --batch_size 4 --num_workers 2 --max_batches 10

Note:
    - Multi-worker DataLoader runs FIRST (Step 2) to avoid torchcodec fork deadlock.
    - Use --no_mapping to skip per-dataset normalization if no dataset_to_episodes.json.
    - Use --save_images_dir "" to disable image saving.
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from lerobot.configs.types import FeatureType
from lerobot.datasets.lola_pretrain_dataset import LoLAPretrainDataset, make_collate_fn
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig


# ── Validation helpers ──────────────────────────────────────────────────────


class ValidationResult:
    """Collect validation results."""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def ok(self, msg):
        self.passed.append(msg)

    def fail(self, msg):
        self.failed.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def summary(self):
        total = len(self.passed) + len(self.failed)
        lines = [
            "=" * 60,
            "Validation Summary",
            "=" * 60,
            f"Passed: {len(self.passed)}/{total}",
            f"Failed: {len(self.failed)}/{total}",
            f"Warnings: {len(self.warnings)}",
        ]
        if self.failed:
            lines.append("--- Failures ---")
            for f in self.failed:
                lines.append(f"  [FAIL] {f}")
        if self.warnings:
            lines.append("--- Warnings ---")
            for w in self.warnings:
                lines.append(f"  [WARN] {w}")
        lines.append("")
        if not self.failed:
            lines.append("All validations passed!")
        else:
            lines.append("Some validations failed -- please investigate!")
        return "\n".join(lines)


def _build_config(dataset_root, dataset_to_episodes_path=None, sub_root=None,
                   temp_process=False, max_history_length=100,
                   action_chunk_size=10, n_obs_steps=1, pred_chunk_size=50):
    """Build LoLAConfig and delta_timestamps from dataset metadata."""
    dataset_metadata = LoLAPretrainDataset._build_metadata_polars(
        repo_id="test", root=dataset_root, revision=None
    )
    fps = dataset_metadata.fps
    features = dataset_to_policy_features(dataset_metadata.features)
    action_dim = features["action"].shape[0] if "action" in features else 20

    config = LoLAConfig(
        vlm_model_name="Qwen/Qwen3.5-4B",
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        pred_chunk_size=pred_chunk_size,
        n_obs_steps=n_obs_steps,
        input_features={key: ft for key, ft in features.items() if ft.type != FeatureType.ACTION},
        output_features={key: ft for key, ft in features.items() if ft.type == FeatureType.ACTION},
        load_full_history=True,
        max_history_length=max_history_length,
    )

    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in config.action_delta_indices]
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    return config, delta_timestamps


def check_no_nan_inf(batch, result, label=""):
    """Check all float tensors for NaN / Inf."""
    prefix = f"[{label}] " if label else ""
    has_issue = False
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            if torch.isnan(val).any():
                result.fail(f"{prefix}[{key}] contains NaN")
                has_issue = True
            if torch.isinf(val).any():
                result.fail(f"{prefix}[{key}] contains Inf")
                has_issue = True
    if not has_issue:
        result.ok(f"{prefix}No NaN/Inf in float tensors")


def _make_invalid_placeholder(width, height):
    """Create a black PIL Image with 'INVALID' text for None camera entries."""
    if not HAS_PIL:
        return None
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((width // 4, height // 4), "INVALID", fill=(255, 255, 255))
    except Exception:
        pass
    return img


def save_pil_images(items, camera_keys, save_dir, stage_name, max_samples=3, dataset=None):
    """Save decoded camera PIL Images to disk for visual inspection.

    Args:
        items: list of dataset __getitem__ dicts
        camera_keys: camera key list
        save_dir: root save directory
        stage_name: sub-directory name (e.g. 'step5_video_decode')
        max_samples: max items to save
        dataset: optional, for getting expected resolution
    """
    if not HAS_PIL:
        print("    [Skip image save] PIL not available")
        return

    stage_dir = os.path.join(save_dir, stage_name)
    os.makedirs(stage_dir, exist_ok=True)

    saved_count = 0
    n_samples = min(len(items), max_samples)

    # Get default placeholder size from dataset metadata if available
    placeholder_w, placeholder_h = 64, 64
    if dataset is not None and camera_keys:
        first_cam = camera_keys[0]
        shape = dataset.meta.info["features"][first_cam]["shape"]
        if len(shape) == 3:
            placeholder_h, placeholder_w = shape[1], shape[2]

    for i in range(n_samples):
        item = items[i]
        ep_idx = item.get("episode_index")
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = ep_idx.item()
        idx_val = item.get("index")
        if isinstance(idx_val, torch.Tensor):
            idx_val = idx_val.item()

        valid_frames = []
        for cam_key in camera_keys:
            val = item.get(cam_key)
            cam_name = cam_key.replace("/", "_").replace(".", "_")

            if val is not None and isinstance(val, Image.Image):
                fname = os.path.join(stage_dir, f"{cam_name}_ep{ep_idx}_idx{idx_val}.png")
                val.save(fname)
                saved_count += 1
                valid_frames.append(val)
            else:
                # Save placeholder for invalid camera
                placeholder = _make_invalid_placeholder(placeholder_w, placeholder_h)
                if placeholder is not None:
                    fname = os.path.join(stage_dir, f"{cam_name}_INVALID_ep{ep_idx}_idx{idx_val}.png")
                    placeholder.save(fname)
                    saved_count += 1
                    valid_frames.append(placeholder)

        # Concatenate all cameras horizontally
        if len(valid_frames) > 1:
            total_w = sum(f.width for f in valid_frames)
            max_h = max(f.height for f in valid_frames)
            composite = Image.new("RGB", (total_w, max_h), color=(0, 0, 0))
            x_offset = 0
            for f in valid_frames:
                composite.paste(f, (x_offset, 0))
                x_offset += f.width
            composite_fname = os.path.join(stage_dir, f"all_cameras_ep{ep_idx}_idx{idx_val}.png")
            composite.save(composite_fname)
            saved_count += 1

    if saved_count > 0:
        print(f"    Saved {saved_count} images to {stage_dir}/")


def save_batch_images(batch, camera_keys, save_dir, stage_name, max_samples=3, dataset=None):
    """Save camera images from a DataLoader batch dict to disk.

    In LoLAPretrainDataset collate, camera keys are lists of PIL Image / None.
    """
    if not HAS_PIL:
        print("    [Skip image save] PIL not available")
        return

    stage_dir = os.path.join(save_dir, stage_name)
    os.makedirs(stage_dir, exist_ok=True)

    saved_count = 0

    # Get placeholder size
    placeholder_w, placeholder_h = 64, 64
    if dataset is not None and camera_keys:
        first_cam = camera_keys[0]
        shape = dataset.meta.info["features"][first_cam]["shape"]
        if len(shape) == 3:
            placeholder_h, placeholder_w = shape[1], shape[2]

    # Determine batch size from any list-valued key
    bs = 0
    for cam_key in camera_keys:
        if cam_key in batch and isinstance(batch[cam_key], list):
            bs = len(batch[cam_key])
            break
    if bs == 0:
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                bs = val.shape[0]
                break
    if bs == 0:
        return

    n_samples = min(bs, max_samples)

    for i in range(n_samples):
        ep_idx_batch = batch.get("episode_index")
        if isinstance(ep_idx_batch, torch.Tensor):
            ep_idx = ep_idx_batch[i].item()
        elif isinstance(ep_idx_batch, list):
            ep_idx = ep_idx_batch[i]
        else:
            ep_idx = i

        idx_val_batch = batch.get("index")
        if isinstance(idx_val_batch, torch.Tensor):
            idx_val = idx_val_batch[i].item()
        elif isinstance(idx_val_batch, list):
            idx_val = idx_val_batch[i]
        else:
            idx_val = i

        valid_frames = []
        for cam_key in camera_keys:
            cam_list = batch.get(cam_key)
            cam_name = cam_key.replace("/", "_").replace(".", "_")

            if cam_list is not None and i < len(cam_list):
                val = cam_list[i]
                if val is not None and isinstance(val, Image.Image):
                    fname = os.path.join(stage_dir, f"{cam_name}_ep{ep_idx}_idx{idx_val}.png")
                    val.save(fname)
                    saved_count += 1
                    valid_frames.append(val)
                else:
                    placeholder = _make_invalid_placeholder(placeholder_w, placeholder_h)
                    if placeholder is not None:
                        fname = os.path.join(stage_dir, f"{cam_name}_INVALID_ep{ep_idx}_idx{idx_val}.png")
                        placeholder.save(fname)
                        saved_count += 1
                        valid_frames.append(placeholder)

        # Concatenate horizontally
        if len(valid_frames) > 1:
            total_w = sum(f.width for f in valid_frames)
            max_h = max(f.height for f in valid_frames)
            composite = Image.new("RGB", (total_w, max_h), color=(0, 0, 0))
            x_offset = 0
            for f in valid_frames:
                composite.paste(f, (x_offset, 0))
                x_offset += f.width
            composite_fname = os.path.join(stage_dir, f"all_cameras_ep{ep_idx}_idx{idx_val}.png")
            composite.save(composite_fname)
            saved_count += 1

    if saved_count > 0:
        print(f"    Saved {saved_count} images to {stage_dir}/")


# ── Step 1: Dataset creation + metadata ─────────────────────────────────────


def step1_dataset_creation(args, result):
    """Step 1: Create dataset and verify metadata."""
    print("\n" + "=" * 60)
    print("Step 1: Dataset Creation + Metadata Sanity")
    print("=" * 60)

    dataset_to_episodes_path = args.dataset_to_episodes_path
    if args.no_mapping:
        dataset_to_episodes_path = None

    config, delta_timestamps = _build_config(
        args.dataset_root,
        dataset_to_episodes_path,
        args.sub_root,
        args.temp_process,
        args.max_history_length,
        args.action_chunk_size,
        args.n_obs_steps,
        args.pred_chunk_size,
    )

    dataset = LoLAPretrainDataset(
        repo_id="test",
        max_history_length=args.max_history_length,
        action_chunk_size=config.action_chunk_size,
        history_padding_side=args.history_padding_side,
        root=args.dataset_root,
        sub_root=args.sub_root,
        delta_timestamps=delta_timestamps,
        dataset_to_episodes_path=dataset_to_episodes_path,
        temp_process=args.temp_process,
        tolerance_frames=args.tolerance_frames,
        decode_device=args.decode_device,
    )

    # Validate basic properties
    if len(dataset) > 0:
        result.ok(f"Dataset has {len(dataset)} frames")
    else:
        result.fail("Dataset has 0 frames")

    if dataset.num_episodes > 0:
        result.ok(f"Dataset has {dataset.num_episodes} episodes")
    else:
        result.fail("Dataset has 0 episodes")

    if dataset.action_dim > 0:
        result.ok(f"action_dim={dataset.action_dim}")
    else:
        result.fail("action_dim=0")

    camera_keys = dataset.meta.camera_keys
    video_keys = dataset.meta.video_keys
    if len(camera_keys) > 0:
        result.ok(f"camera_keys={camera_keys}")
    else:
        result.warn("No camera keys found")

    if set(video_keys).issubset(set(camera_keys)):
        result.ok("video_keys subset of camera_keys")
    else:
        result.fail(f"video_keys not subset of camera_keys: {video_keys} vs {camera_keys}")

    sub_ds_count = len(dataset._sub_dataset_names)
    if dataset_to_episodes_path is not None:
        if sub_ds_count > 0:
            result.ok(f"Sub-datasets loaded: {sub_ds_count}")
        else:
            result.fail("No sub-datasets loaded despite dataset_to_episodes_path provided")
    else:
        result.ok("No dataset_to_episodes_path provided, skipping sub-dataset check")

    print(f"  fps: {dataset.fps}")
    print(f"  total_rows: {len(dataset)}")
    print(f"  total_episodes: {dataset.num_episodes}")
    print(f"  camera_keys: {camera_keys}")
    print(f"  video_keys: {video_keys}")
    print(f"  sub_datasets: {sub_ds_count}")

    return dataset


# ── Step 2: Multi-worker DataLoader traversal ───────────────────────────────


def step2_multiworker_traversal(dataset, args, result):
    """Step 2: Multi-worker DataLoader traversal (runs FIRST for torchcodec fork safety)."""
    print("\n" + "=" * 60)
    print(f"Step 2: Multi-Worker DataLoader Traversal "
          f"(num_workers={args.num_workers}, batch_size={args.batch_size}, "
          f"max_batches={args.max_batches})")
    print("=" * 60)

    # Create a fresh dataset instance for multi-worker test
    dataset_to_episodes_path = args.dataset_to_episodes_path
    if args.no_mapping:
        dataset_to_episodes_path = None

    config, delta_timestamps = _build_config(
        args.dataset_root,
        dataset_to_episodes_path,
        args.sub_root,
        args.temp_process,
        args.max_history_length,
        args.action_chunk_size,
        args.n_obs_steps,
        args.pred_chunk_size,
    )

    fresh_dataset = LoLAPretrainDataset(
        repo_id="test",
        max_history_length=args.max_history_length,
        action_chunk_size=config.action_chunk_size,
        history_padding_side=args.history_padding_side,
        root=args.dataset_root,
        sub_root=args.sub_root,
        delta_timestamps=delta_timestamps,
        dataset_to_episodes_path=dataset_to_episodes_path,
        temp_process=args.temp_process,
        tolerance_frames=args.tolerance_frames,
        decode_device=args.decode_device,
    )

    loader = DataLoader(
        fresh_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        collate_fn=make_collate_fn(),
        pin_memory=False,
    )

    batch_count = 0
    sample_count = 0
    start_time = time.time()
    camera_keys = fresh_dataset.meta.camera_keys
    first_batch_keys = None

    try:
        for batch_idx, batch in enumerate(loader):
            batch_count += 1

            # Determine batch size
            bs = None
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    bs = val.shape[0]
                    break
                elif isinstance(val, list):
                    bs = len(val)
                    break
            sample_count += bs if bs is not None else 0

            # Record first batch keys
            if first_batch_keys is None:
                first_batch_keys = set(batch.keys())

            # Consistent batch_size across all keys
            batch_sizes = set()
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch_sizes.add(val.shape[0])
                elif isinstance(val, list):
                    batch_sizes.add(len(val))
            if len(batch_sizes) > 1:
                result.fail(f"[batch {batch_idx}] batch dimensions inconsistent: {batch_sizes}")
            elif len(batch_sizes) == 1:
                actual_bs = batch_sizes.pop()
                if actual_bs != args.batch_size and batch_idx == 0:
                    result.warn(f"[batch {batch_idx}] batch_size={actual_bs} < {args.batch_size}")

            # hist_actions checks
            if "hist_actions_full" in batch and "hist_actions_mask" in batch:
                haf = batch["hist_actions_full"]
                ham = batch["hist_actions_mask"]
                if haf.shape[0] != ham.shape[0]:
                    result.fail(f"[batch {batch_idx}] hist_actions batch dim mismatch: {haf.shape[0]} vs {ham.shape[0]}")
                if haf.shape[1] != ham.shape[1]:
                    result.fail(f"[batch {batch_idx}] hist_actions seq len mismatch: {haf.shape[1]} vs {ham.shape[1]}")
                if haf.shape[-1] != fresh_dataset.action_dim:
                    result.fail(f"[batch {batch_idx}] hist_actions last dim {haf.shape[-1]} != action_dim {fresh_dataset.action_dim}")
                if ham.dtype != torch.bool:
                    result.fail(f"[batch {batch_idx}] hist_actions_mask dtype {ham.dtype}, expected bool")

            if "hist_actions_length" in batch:
                hal = batch["hist_actions_length"]
                max_hal = hal.max().item()
                if max_hal > args.max_history_length:
                    result.fail(f"[batch {batch_idx}] hist_actions_length max {max_hal} > max_history_length {args.max_history_length}")

            # action_dim / state_dim
            if "action_dim" in batch:
                ad = batch["action_dim"]
                if not isinstance(ad, torch.Tensor):
                    result.fail(f"[batch {batch_idx}] action_dim type {type(ad).__name__}, expected tensor")
                elif ad.shape[0] != (bs or args.batch_size):
                    result.fail(f"[batch {batch_idx}] action_dim shape {ad.shape}, expected [{bs}]")
            if "state_dim" in batch:
                sd = batch["state_dim"]
                if not isinstance(sd, torch.Tensor):
                    result.fail(f"[batch {batch_idx}] state_dim type {type(sd).__name__}, expected tensor")

            # NaN/Inf check
            check_no_nan_inf(batch, result, label=f"batch{batch_idx}")

            # episode_index / index range check
            if "episode_index" in batch and isinstance(batch["episode_index"], torch.Tensor):
                ep_max = batch["episode_index"].max().item()
                ep_min = batch["episode_index"].min().item()
                if ep_max >= fresh_dataset.num_episodes:
                    result.fail(f"[batch {batch_idx}] episode_index max {ep_max} >= num_episodes {fresh_dataset.num_episodes}")
                if ep_min < 0:
                    result.fail(f"[batch {batch_idx}] episode_index min {ep_min} < 0")

            if "index" in batch and isinstance(batch["index"], torch.Tensor):
                idx_max = batch["index"].max().item()
                idx_min = batch["index"].min().item()
                if idx_max >= len(fresh_dataset):
                    result.fail(f"[batch {batch_idx}] index max {idx_max} >= len(dataset {len(fresh_dataset)}")
                if idx_min < 0:
                    result.fail(f"[batch {batch_idx}] index min {idx_min} < 0")

            # Camera keys are lists of PIL Image / None
            for cam_key in camera_keys:
                if cam_key in batch:
                    val = batch[cam_key]
                    if not isinstance(val, list):
                        result.fail(f"[batch {batch_idx}] camera key '{cam_key}' is not a list, got {type(val).__name__}")
                    elif len(val) != (bs or args.batch_size):
                        result.fail(f"[batch {batch_idx}] camera key '{cam_key}' list length {len(val)} != batch_size {bs}")
                    else:
                        for v in val:
                            if v is not None and not isinstance(v, Image.Image):
                                result.fail(f"[batch {batch_idx}] camera '{cam_key}' entry is neither None nor PIL Image: {type(v).__name__}")

            # camera_valid_mask is list of dicts
            if "camera_valid_mask" in batch:
                cvm = batch["camera_valid_mask"]
                if not isinstance(cvm, list):
                    result.fail(f"[batch {batch_idx}] camera_valid_mask is not a list, got {type(cvm).__name__}")
                elif len(cvm) != (bs or args.batch_size):
                    result.fail(f"[batch {batch_idx}] camera_valid_mask length {len(cvm)} != batch_size {bs}")
                else:
                    for d in cvm:
                        if not isinstance(d, dict):
                            result.fail(f"[batch {batch_idx}] camera_valid_mask entry is not dict: {type(d).__name__}")

            # Print first 3 batches detail
            if batch_idx < 3:
                print(f"  batch {batch_idx}:")
                for key, val in batch.items():
                    if isinstance(val, torch.Tensor):
                        print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
                    elif isinstance(val, list):
                        types = set(type(v).__name__ for v in val[:4])
                        print(f"    {key}: list len={len(val)}, types={types}")
                    elif isinstance(val, dict):
                        print(f"    {key}: dict")
                    else:
                        print(f"    {key}: {type(val).__name__}")

            # Save images from first batch
            if args.save_images_dir and batch_idx == 0:
                save_batch_images(batch, camera_keys, args.save_images_dir, "step2_multibatch",
                                  max_samples=args.num_images_per_stage, dataset=fresh_dataset)

            # Progress
            elapsed = time.time() - start_time
            if (batch_idx + 1) % max(1, args.max_batches // 200) == 0 or batch_idx == 0:
                speed = (batch_idx + 1) / max(elapsed, 1e-6)
                eta = (args.max_batches - batch_idx - 1) / max(speed, 1e-6)
                print(f"  [{(batch_idx+1)/args.max_batches*100:.0f}%] batch {batch_idx+1}/{args.max_batches}, "
                      f"{sample_count} samples, {speed:.1f} batch/s, ETA {eta:.0f}s")

            if batch_count >= args.max_batches:
                break

        elapsed = time.time() - start_time
        result.ok(f"Multi-worker traversal: {batch_count} batches, {sample_count} samples, "
                  f"{elapsed:.1f}s, {elapsed/max(batch_count,1):.2f}s/batch")

    except Exception as e:
        result.fail(f"Multi-worker traversal failed: {e}")
        traceback.print_exc()


# ── Step 3: Index uniqueness & correctness ──────────────────────────────────


def step3_index_uniqueness(dataset, args, result):
    """Step 3: Index uniqueness & correctness (num_workers=0)."""
    print("\n" + "=" * 60)
    print("Step 3: Index Uniqueness & Correctness")
    print("=" * 60)

    n = len(dataset)
    rng = np.random.default_rng(42)

    # (a) Index field matches dataset index
    print("  (a) Checking index field matches dataset index...")
    test_indices = [0, n // 2, n - 1]
    for _ in range(min(args.num_index_samples - 3, 17)):
        test_indices.append(int(rng.integers(0, n)))
    test_indices = test_indices[:args.num_index_samples]

    mismatch_count = 0
    for idx in test_indices:
        item = dataset[idx]
        item_index = item.get("index")
        if isinstance(item_index, torch.Tensor):
            item_index = item_index.item()
        if item_index != idx:
            mismatch_count += 1
            if mismatch_count <= 5:
                result.fail(f"dataset[{idx}]['index'] = {item_index}, expected {idx}")

    if mismatch_count == 0:
        result.ok(f"Index field matches dataset index for all {len(test_indices)} samples")
    else:
        result.fail(f"Index field mismatch for {mismatch_count}/{len(test_indices)} samples")

    # (b) Different indices yield different data
    print("  (b) Checking different indices yield different data...")
    # Pick indices across different episodes
    num_episodes = dataset.num_episodes
    sample_indices = []
    if num_episodes >= 10:
        # Pick one index from each of 10 different episodes
        ep_step = max(1, num_episodes // 10)
        for ep in range(0, num_episodes, ep_step):
            ep_start = int(dataset._episode_starts[ep])
            ep_end = int(dataset._episode_ends[ep])
            mid = (ep_start + ep_end) // 2
            if mid < n:
                sample_indices.append(mid)
    else:
        # Fewer episodes, pick multiple indices within each
        for ep in range(num_episodes):
            ep_start = int(dataset._episode_starts[ep])
            ep_end = int(dataset._episode_ends[ep])
            for offset in [0, (ep_end - ep_start) // 2, ep_end - ep_start - 1]:
                idx = ep_start + offset
                if idx < n and idx >= 0:
                    sample_indices.append(idx)

    sample_indices = sample_indices[:args.num_index_samples]

    # Check (episode_index, frame_index) uniqueness
    identity_set = set()
    duplicate_count = 0
    for idx in sample_indices:
        item = dataset[idx]
        ep_idx = item.get("episode_index")
        frame_idx = item.get("frame_index")
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = ep_idx.item()
        if isinstance(frame_idx, torch.Tensor):
            frame_idx = frame_idx.item()
        identity = (ep_idx, frame_idx)
        if identity in identity_set:
            duplicate_count += 1
            result.fail(f"dataset[{idx}] has same (episode_index, frame_index)={identity} as another item")
        identity_set.add(identity)

    if duplicate_count == 0:
        result.ok(f"All {len(sample_indices)} items have unique (episode_index, frame_index) pairs")
    else:
        result.fail(f"{duplicate_count} items share (episode_index, frame_index) with another item")

    # (c) Same index returns identical data
    print("  (c) Checking same index returns identical data...")
    repeat_indices = [0, n // 4, n // 2, n - 10, n - 1]
    repeat_indices = [i for i in repeat_indices if 0 <= i < n]

    mismatch_count = 0
    for idx in repeat_indices:
        item1 = dataset[idx]
        item2 = dataset[idx]

        # Compare all tensor fields
        for key in item1.keys():
            v1, v2 = item1[key], item2[key]
            if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                if v1.is_floating_point():
                    if not torch.allclose(v1, v2):
                        mismatch_count += 1
                        result.fail(f"dataset[{idx}] called twice: '{key}' float tensors differ")
                else:
                    if not torch.equal(v1, v2):
                        mismatch_count += 1
                        result.fail(f"dataset[{idx}] called twice: '{key}' tensors differ")
            elif isinstance(v1, str) and isinstance(v2, str):
                if v1 != v2:
                    mismatch_count += 1
                    result.fail(f"dataset[{idx}] called twice: '{key}' strings differ: '{v1}' vs '{v2}'")
            elif isinstance(v1, dict) and isinstance(v2, dict):
                if v1 != v2:
                    mismatch_count += 1
                    result.fail(f"dataset[{idx}] called twice: '{key}' dicts differ")
            elif isinstance(v1, Image.Image) and isinstance(v2, Image.Image):
                # PIL images: compare size and mode
                if v1.size != v2.size or v1.mode != v2.mode:
                    mismatch_count += 1
                    result.fail(f"dataset[{idx}] called twice: '{key}' PIL Image size/mode differ")
            elif v1 is None and v2 is None:
                pass  # Both None, OK
            elif v1 is None or v2 is None:
                mismatch_count += 1
                result.fail(f"dataset[{idx}] called twice: '{key}' one is None, other is {type(v2 if v1 is None else v1).__name__}")

    if mismatch_count == 0:
        result.ok(f"Same index returns identical data for all {len(repeat_indices)} tested indices")
    else:
        result.fail(f"Same index returns different data for {mismatch_count} comparisons")


# ── Step 4: collate_fn correctness ──────────────────────────────────────────


def step4_collate_fn(dataset, args, result):
    """Step 4: collate_fn correctness with real data."""
    print("\n" + "=" * 60)
    print("Step 4: collate_fn Correctness with Real Data")
    print("=" * 60)

    n = len(dataset)
    camera_keys = dataset.meta.camera_keys

    # Fetch 3 items from different episodes (likely different hist_actions_length)
    num_episodes = dataset.num_episodes
    if num_episodes < 3:
        result.warn(f"Only {num_episodes} episodes, fetching items from same episode for collate test")
        test_indices = [0, n // 2, n - 1]
    else:
        # Pick indices from 3 different episodes
        ep_indices = [0, num_episodes // 2, num_episodes - 1]
        test_indices = []
        for ep in ep_indices:
            ep_start = int(dataset._episode_starts[ep])
            ep_end = int(dataset._episode_ends[ep])
            mid = (ep_start + ep_end) // 2
            test_indices.append(mid)

    items = [dataset[idx] for idx in test_indices]
    print(f"  Fetched items at indices: {test_indices}")
    for i, item in enumerate(items):
        hal = item.get("hist_actions_length")
        if isinstance(hal, torch.Tensor):
            hal = hal.item()
        print(f"    item {i}: hist_actions_length={hal}, episode_index={item.get('episode_index')}")

    # Call collate_fn
    collate_fn = make_collate_fn()
    batch = collate_fn(items)

    # (a) Variable-length padding
    print("  (a) Checking variable-length hist_actions padding...")
    if "hist_actions_full" in batch and "hist_actions_mask" in batch:
        haf = batch["hist_actions_full"]
        ham = batch["hist_actions_mask"]

        # Shape checks
        if haf.shape[0] != len(items):
            result.fail(f"hist_actions_full batch dim {haf.shape[0]} != {len(items)} items")
        else:
            result.ok(f"hist_actions_full batch dim = {len(items)}")

        if haf.shape[-1] != dataset.action_dim:
            result.fail(f"hist_actions_full last dim {haf.shape[-1]} != action_dim {dataset.action_dim}")
        else:
            result.ok(f"hist_actions_full last dim = action_dim ({dataset.action_dim})")

        if haf.shape[1] != ham.shape[1]:
            result.fail(f"hist_actions seq len mismatch: {haf.shape[1]} vs {ham.shape[1]}")
        else:
            result.ok(f"hist_actions seq len consistent: {haf.shape[1]}")

        padded_len = haf.shape[1]
        # Check padding regions
        for i, item in enumerate(items):
            orig_len = item["hist_actions_length"].item()
            pad_count = padded_len - orig_len

            if args.history_padding_side == "left":
                # Left padding: first pad_count entries should be mask=False, actions=0
                if pad_count > 0:
                    pad_mask = ham[i, :pad_count]
                    pad_actions = haf[i, :pad_count]
                    if not (pad_mask == False).all():
                        result.fail(f"item {i}: left padding mask region has True values")
                    else:
                        result.ok(f"item {i}: left padding mask all False ({pad_count} entries)")
                    if not (pad_actions == 0).all():
                        result.warn(f"item {i}: left padding actions not all zero (may be truncation)")
                    else:
                        result.ok(f"item {i}: left padding actions all zero")

                    # Valid region: compare only the truly valid portion using hist_actions_length
                    orig_len = item["hist_actions_length"].item()
                    collated_valid = ham[i, pad_count:pad_count + orig_len]
                    orig_mask = item["hist_actions_mask"]
                    # orig_mask may have its own padding; find the valid portion
                    if orig_mask.shape[0] >= orig_len:
                        # If left-padded, valid portion is at the end
                        orig_valid = orig_mask[-orig_len:] if orig_mask.shape[0] > orig_len else orig_mask
                    else:
                        orig_valid = orig_mask
                    if not torch.equal(collated_valid, orig_valid):
                        result.fail(f"item {i}: valid mask region doesn't match original")
                    else:
                        result.ok(f"item {i}: valid mask matches original")

    # (b) Camera key handling
    print("  (b) Checking camera key handling (PIL Image / None lists)...")
    for cam_key in camera_keys:
        if cam_key in batch:
            val = batch[cam_key]
            if not isinstance(val, list):
                result.fail(f"camera '{cam_key}' collated to {type(val).__name__}, expected list")
            elif len(val) != len(items):
                result.fail(f"camera '{cam_key}' list length {len(val)} != {len(items)}")
            else:
                result.ok(f"camera '{cam_key}' collated as list of {len(val)} entries")
                # Check each entry
                pil_count = sum(1 for v in val if v is not None and isinstance(v, Image.Image))
                none_count = sum(1 for v in val if v is None)
                other_count = len(val) - pil_count - none_count
                if other_count > 0:
                    result.fail(f"camera '{cam_key}' has {other_count} non-PIL/non-None entries")
                else:
                    result.ok(f"camera '{cam_key}': {pil_count} PIL, {none_count} None")

    # (c) camera_valid_mask consistency
    print("  (c) Checking camera_valid_mask consistency...")
    if "camera_valid_mask" in batch:
        cvm = batch["camera_valid_mask"]
        if not isinstance(cvm, list):
            result.fail(f"camera_valid_mask type {type(cvm).__name__}, expected list")
        elif len(cvm) != len(items):
            result.fail(f"camera_valid_mask length {len(cvm)} != {len(items)}")
        else:
            result.ok(f"camera_valid_mask is list of {len(cvm)} dicts")
            # Verify consistency with camera entries
            for i, mask_dict in enumerate(cvm):
                for cam_key, is_valid in mask_dict.items():
                    if cam_key in batch:
                        cam_val = batch[cam_key][i]
                        if is_valid and (cam_val is None or not isinstance(cam_val, Image.Image)):
                            result.fail(f"item {i}: camera '{cam_key}' valid but value is None or not PIL")
                        elif not is_valid and cam_val is not None:
                            result.fail(f"item {i}: camera '{cam_key}' invalid but value is {type(cam_val).__name__}")

    # (d) action_dim / state_dim
    print("  (d) Checking action_dim / state_dim tensors...")
    if "action_dim" in batch:
        ad = batch["action_dim"]
        if isinstance(ad, torch.Tensor):
            if ad.shape == (len(items),):
                result.ok(f"action_dim shape {ad.shape}, dtype {ad.dtype}")
            else:
                result.fail(f"action_dim shape {ad.shape}, expected ({len(items)},)")
        else:
            result.fail(f"action_dim type {type(ad).__name__}, expected tensor")

    if "state_dim" in batch:
        sd = batch["state_dim"]
        if isinstance(sd, torch.Tensor):
            if sd.shape == (len(items),):
                result.ok(f"state_dim shape {sd.shape}, dtype {sd.dtype}")
            else:
                result.fail(f"state_dim shape {sd.shape}, expected ({len(items)},)")
        else:
            result.fail(f"state_dim type {type(sd).__name__}, expected tensor")

    # (e) task
    print("  (e) Checking task (list of strings)...")
    if "task" in batch:
        task = batch["task"]
        if isinstance(task, list) and len(task) == len(items):
            non_empty = sum(1 for t in task if isinstance(t, str) and len(t) > 0)
            if non_empty == len(task):
                result.ok(f"task is list of {len(task)} non-empty strings")
            else:
                result.fail(f"task has {len(task) - non_empty} empty/non-string entries")
        else:
            result.fail(f"task type {type(task).__name__}, expected list of {len(items)} strings")

    # (f) Standard tensor stacking
    print("  (f) Checking standard tensor stacking...")
    for key in ["observation.state", "action", "episode_index", "index", "hist_actions_length"]:
        if key in batch:
            val = batch[key]
            if isinstance(val, torch.Tensor):
                if val.shape[0] == len(items):
                    result.ok(f"'{key}' stacked: shape {val.shape}, dtype {val.dtype}")
                else:
                    result.fail(f"'{key}' batch dim {val.shape[0]} != {len(items)}")
            elif isinstance(val, list):
                # episode_index/index come as Python ints from _row_to_item,
                # make_collate_fn has no special case for them → list of ints
                if key in ("episode_index", "index"):
                    if len(val) == len(items):
                        result.ok(f"'{key}' collated as list of {len(val)} ints")
                    else:
                        result.fail(f"'{key}' list length {len(val)} != {len(items)}")
                else:
                    result.fail(f"'{key}' type list, expected tensor")
            else:
                result.fail(f"'{key}' type {type(val).__name__}, expected tensor or list")


# ── Step 5: Video decode correctness ────────────────────────────────────────


def step5_video_decode(dataset, args, result):
    """Step 5: Video decode correctness + save sample images."""
    print("\n" + "=" * 60)
    print("Step 5: Video Decode Correctness")
    print("=" * 60)

    camera_keys = dataset.meta.camera_keys
    num_episodes = dataset.num_episodes
    n = len(dataset)

    # (a) Valid cameras
    print("  (a) Checking valid camera frames...")
    valid_items = []
    rng = np.random.default_rng(42)

    # Find episodes with all valid cameras
    all_valid_episodes = []
    for ep_idx in range(num_episodes):
        ep_meta = dataset.meta.episodes[ep_idx]
        all_valid = True
        for cam_key in camera_keys:
            is_valid_key = f"videos/{cam_key}/is_valid"
            if ep_meta.get(is_valid_key, 1) == 0:
                all_valid = False
                break
        if all_valid:
            all_valid_episodes.append(ep_idx)

    if not all_valid_episodes:
        result.warn("No episodes with all-valid cameras found")
    else:
        # Sample indices from valid episodes
        sample_ep_indices = all_valid_episodes[:min(3, len(all_valid_episodes))]
        valid_indices = []
        for ep in sample_ep_indices:
            ep_start = int(dataset._episode_starts[ep])
            ep_end = int(dataset._episode_ends[ep])
            mid = (ep_start + ep_end) // 2
            valid_indices.append(mid)

        for idx in valid_indices:
            item = dataset[idx]
            valid_items.append(item)
            ep_idx = item["episode_index"]
            if isinstance(ep_idx, torch.Tensor):
                ep_idx = ep_idx.item()

            for cam_key in camera_keys:
                val = item.get(cam_key)

                # Type check: should be PIL Image for valid cameras
                if val is None:
                    result.fail(f"dataset[{idx}] camera '{cam_key}' is None (should be PIL Image for valid camera)")
                elif not isinstance(val, Image.Image):
                    result.fail(f"dataset[{idx}] camera '{cam_key}' is {type(val).__name__} (expected PIL Image)")
                else:
                    result.ok(f"dataset[{idx}] camera '{cam_key}': PIL Image {val.size} {val.mode}")

                    # Not all-black
                    arr = np.array(val)
                    mean_val = arr.mean()
                    if mean_val < 0.01 * 255:
                        result.fail(f"camera '{cam_key}' all-black: mean={mean_val:.1f}")
                    else:
                        result.ok(f"camera '{cam_key}' not all-black: mean={mean_val:.1f}")

                    # Not all-white
                    min_val = arr.min()
                    if min_val > 0.99 * 255:
                        result.fail(f"camera '{cam_key}' all-white: min={min_val}")
                    else:
                        result.ok(f"camera '{cam_key}' not all-white: min={min_val}")

                    # Channel check
                    if val.mode not in ("RGB", "L"):
                        result.warn(f"camera '{cam_key}' mode={val.mode}, expected RGB or L")
                    else:
                        result.ok(f"camera '{cam_key}' mode={val.mode}")

                    # Resolution check
                    expected_shape = dataset.meta.info["features"][cam_key]["shape"]
                    if len(expected_shape) == 3:
                        expected_w, expected_h = expected_shape[2], expected_shape[1]
                        actual_w, actual_h = val.size
                        if (actual_w, actual_h) != (expected_w, expected_h):
                            result.warn(f"camera '{cam_key}' resolution ({actual_w},{actual_h}) "
                                        f"!= expected ({expected_w},{expected_h}) -- dynamic resolution?")
                        else:
                            result.ok(f"camera '{cam_key}' resolution matches ({expected_w},{expected_h})")

    # (b) Invalid cameras
    print("  (b) Checking invalid camera handling...")
    invalid_camera_episodes = {}
    for ep_idx in range(num_episodes):
        ep_meta = dataset.meta.episodes[ep_idx]
        invalid_cams = []
        for cam_key in camera_keys:
            is_valid_key = f"videos/{cam_key}/is_valid"
            if ep_meta.get(is_valid_key, 1) == 0:
                invalid_cams.append(cam_key)
        if invalid_cams:
            invalid_camera_episodes[ep_idx] = invalid_cams

    if not invalid_camera_episodes:
        result.ok("No invalid camera episodes found (all cameras valid)")
    else:
        result.ok(f"Found {len(invalid_camera_episodes)} episodes with invalid cameras")
        # Test a few invalid episodes
        test_invalid_eps = list(invalid_camera_episodes.keys())[:min(2, len(invalid_camera_episodes))]
        invalid_items = []

        for ep_idx in test_invalid_eps:
            invalid_cams = invalid_camera_episodes[ep_idx]
            ep_start = int(dataset._episode_starts[ep_idx])
            ep_end = int(dataset._episode_ends[ep_idx])
            mid = (ep_start + ep_end) // 2
            if mid >= n:
                continue

            item = dataset[mid]
            invalid_items.append(item)
            cvm = item.get("camera_valid_mask", {})

            for cam_key in invalid_cams:
                val = item.get(cam_key)
                mask_val = cvm.get(cam_key)
                if val is not None:
                    result.fail(f"dataset[{mid}] invalid camera '{cam_key}' should be None, got {type(val).__name__}")
                else:
                    result.ok(f"dataset[{mid}] invalid camera '{cam_key}' is None")
                if mask_val != False:
                    result.fail(f"dataset[{mid}] invalid camera '{cam_key}' mask should be False, got {mask_val}")
                else:
                    result.ok(f"dataset[{mid}] invalid camera '{cam_key}' mask is False")

            # Valid cameras in same episode should still work
            valid_cams = [k for k in camera_keys if k not in invalid_cams]
            for cam_key in valid_cams:
                val = item.get(cam_key)
                mask_val = cvm.get(cam_key)
                if mask_val != True:
                    result.fail(f"dataset[{mid}] valid camera '{cam_key}' mask should be True, got {mask_val}")
                elif val is None:
                    result.fail(f"dataset[{mid}] valid camera '{cam_key}' should be PIL Image, got None")
                elif not isinstance(val, Image.Image):
                    result.fail(f"dataset[{mid}] valid camera '{cam_key}' is {type(val).__name__}")
                else:
                    result.ok(f"dataset[{mid}] valid camera '{cam_key}' is PIL Image")

    # (c) Cross-episode resolution check
    print("  (c) Checking cross-episode resolution consistency...")
    resolutions = {}
    sample_ep_indices = list(range(0, num_episodes, max(1, num_episodes // 5)))[:5]
    for ep_idx in sample_ep_indices:
        ep_start = int(dataset._episode_starts[ep_idx])
        ep_end = int(dataset._episode_ends[ep_idx])
        mid = (ep_start + ep_end) // 2
        if mid >= n:
            continue
        item = dataset[mid]
        for cam_key in camera_keys:
            val = item.get(cam_key)
            if isinstance(val, Image.Image):
                resolutions.setdefault(cam_key, set()).add(val.size)

    for cam_key, res_set in resolutions.items():
        if len(res_set) > 1:
            result.warn(f"camera '{cam_key}' has varying resolutions: {res_set} (dynamic resolution expected for some AV1 datasets)")
        else:
            result.ok(f"camera '{cam_key}' consistent resolution: {res_set}")

    # (d) Save sample images to disk
    if args.save_images_dir:
        print("  (d) Saving sample images to disk...")
        all_items_to_save = valid_items + (invalid_items if invalid_camera_episodes else [])
        save_pil_images(all_items_to_save, camera_keys, args.save_images_dir,
                        "step5_video_decode", max_samples=args.num_images_per_stage, dataset=dataset)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Validate LoLAPretrainDataset DataLoader")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Dataset root directory")
    parser.add_argument("--dataset_to_episodes_path", type=str, default=None,
                        help="dataset_to_episodes.json path")
    parser.add_argument("--no_mapping", action="store_true",
                        help="Skip per-dataset mapping")
    parser.add_argument("--sub_root", type=str, default=None,
                        help="Sub-dataset root directory")
    parser.add_argument("--temp_process", action="store_true",
                        help="Zero-pad mismatched sub-dataset stats")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=20,
                        help="Max batches for traversal")
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--history_padding_side", type=str, default="left",
                        choices=["left", "right"])
    parser.add_argument("--decode_device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--num_index_samples", type=int, default=20,
                        help="Number of indices for uniqueness checks")
    parser.add_argument("--save_images_dir", type=str, default="./validate_images",
                        help="Save decoded images for visual inspection (empty string to disable)")
    parser.add_argument("--num_images_per_stage", type=int, default=3,
                        help="Max images per camera per stage")
    parser.add_argument("--tolerance_frames", type=int, default=1,
                        help="Max allowed frame offset for video decode (tolerance_s computed per-video)")

    args = parser.parse_args()
    if args.save_images_dir == "":
        args.save_images_dir = None

    result = ValidationResult()

    print("=" * 60)
    print("LoLAPretrainDataset DataLoader Validation")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Dataset-to-episodes: {args.dataset_to_episodes_path or '(none)'}")
    print(f"Batch size: {args.batch_size}, num_workers: {args.num_workers}")

    time_start = time.time()

    # Step 1: Dataset creation
    dataset = step1_dataset_creation(args, result)
    time_s1 = time.time()

    # Step 2: Multi-worker traversal (MUST run before num_workers=0 steps)
    step2_multiworker_traversal(dataset, args, result)
    time_s2 = time.time()

    # Step 3: Index uniqueness (num_workers=0, after fork-safe step)
    step3_index_uniqueness(dataset, args, result)
    time_s3 = time.time()

    # Step 4: collate_fn correctness
    step4_collate_fn(dataset, args, result)
    time_s4 = time.time()

    # Step 5: Video decode correctness + image saving
    step5_video_decode(dataset, args, result)
    time_s5 = time.time()

    # Timing summary
    print("\n" + "=" * 60)
    print("Timing Summary")
    print("=" * 60)
    print(f"  Step 1 (dataset creation):  {time_s1 - time_start:.2f}s")
    print(f"  Step 2 (multi-worker):      {time_s2 - time_s1:.2f}s")
    print(f"  Step 3 (index uniqueness):  {time_s3 - time_s2:.2f}s")
    print(f"  Step 4 (collate_fn):        {time_s4 - time_s3:.2f}s")
    print(f"  Step 5 (video decode):      {time_s5 - time_s4:.2f}s")
    print(f"  Total:                      {time_s5 - time_start:.2f}s")

    # Final summary
    print("\n" + result.summary())
    sys.exit(1 if result.failed else 0)


if __name__ == "__main__":
    main()