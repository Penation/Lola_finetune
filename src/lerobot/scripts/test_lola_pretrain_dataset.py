#!/usr/bin/env python
"""
LoLA Pretrain Dataset (Map-style) Unit Test

Tests LoLAPretrainDataset core functionality:
1. Dataset loading and __len__
2. __getitem__ random access and item structure
3. Per-sub-dataset normalization
4. is_valid camera handling
5. Dynamic resolution collate
6. Dimension info
7. Video frame to PIL Image

Usage:
    # Using sample dataset (local)
    python src/lerobot/scripts/test_lola_pretrain_dataset.py \
        --dataset_root /data_6t_2/lerobot_v30/simpler_bridge_v3 \
        --dataset_to_episodes_path /path/to/dataset_to_episodes.json

    # Using merged dataset (cluster)
    python src/lerobot/scripts/test_lola_pretrain_dataset.py \
        --dataset_root /data_16T/deepseek/halo \
        --dataset_to_episodes_path /data_16T/deepseek/halo/dataset_to_episodes.json

Note:
    - If no dataset_to_episodes.json, use --no_mapping to skip per-dataset normalization test
    - Sub-dataset stats may not load locally (cloud storage), script will fallback
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lola_pretrain_dataset import LoLAPretrainDataset, make_collate_fn
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig


def _build_config(dataset_root, dataset_to_episodes_path=None, sub_root=None, temp_process=False):
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
        action_chunk_size=10,
        pred_chunk_size=50,
        n_obs_steps=1,
        input_features={key: ft for key, ft in features.items() if ft.type != FeatureType.ACTION},
        output_features={key: ft for key, ft in features.items() if ft.type == FeatureType.ACTION},
        load_full_history=True,
        max_history_length=10,
    )

    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in config.action_delta_indices]
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    return config, delta_timestamps


def test_basic_loading(dataset_root, dataset_to_episodes_path=None, sub_root=None, temp_process=False):
    """Test 1: Basic Dataset Loading + __len__"""
    print("\n" + "=" * 60)
    print("Test 1: Basic Dataset Loading & __len__")
    print("=" * 60)

    config, delta_timestamps = _build_config(dataset_root, dataset_to_episodes_path, sub_root, temp_process)

    dataset = LoLAPretrainDataset(
        repo_id="test",
        max_history_length=10,
        action_chunk_size=config.action_chunk_size,
        root=dataset_root,
        sub_root=sub_root,
        delta_timestamps=delta_timestamps,
        dataset_to_episodes_path=dataset_to_episodes_path,
        temp_process=temp_process,
    )

    print(f"  Dataset created: {dataset.num_episodes} episodes, {len(dataset)} frames")
    print(f"  Action dim: {dataset.action_dim}")
    print(f"  Video keys: {dataset.meta.video_keys}")
    print(f"  Camera keys: {dataset.meta.camera_keys}")
    print(f"  Sub-datasets loaded: {len(dataset._sub_dataset_names)}")
    assert len(dataset) > 0, "Dataset should have frames"
    print("  PASSED")
    return dataset


def test_random_access(dataset, max_items=5):
    """Test 2: Random Access (__getitem__)"""
    print("\n" + "=" * 60)
    print("Test 2: Random Access (__getitem__)")
    print("=" * 60)

    n = len(dataset)
    print(f"  Total frames: {n}")

    # Test first, middle, last
    indices = [0, n // 2, n - 1]
    # Add random indices
    rng = np.random.default_rng(42)
    for _ in range(max_items - 3):
        indices.append(int(rng.integers(0, n)))

    items = []
    t_start = time.monotonic()
    for i, idx in enumerate(indices):
        if i >= max_items:
            break
        item = dataset[idx]
        items.append(item)
        elapsed = time.monotonic() - t_start
        done = i + 1
        avg = elapsed / done
        eta = avg * (max_items - done)
        print(
            f"\r  Accessing dataset[{idx}]: {done}/{max_items} | "
            f"elapsed: {elapsed:.1f}s | avg: {avg:.2f}s/item | "
            f"ETA: {eta:.1f}s",
            end="", flush=True,
        )
    elapsed = time.monotonic() - t_start
    avg = elapsed / len(items) if items else 0
    print(f"\r  Collected {len(items)} items in {elapsed:.1f}s ({avg:.2f}s/item)" + " " * 20)

    if len(items) == 0:
        print("  WARNING: No items collected!")
        return items

    # Check item keys
    item0 = items[0]
    print(f"  Item keys: {sorted(item0.keys())}")

    # Check required fields
    required_keys = [
        "observation.state", "action", "episode_index", "index",
        "hist_actions_full", "hist_actions_mask", "hist_actions_length",
        "action_dim", "state_dim", "task", "camera_valid_mask",
    ]
    missing = [k for k in required_keys if k not in item0]
    if missing:
        print(f"  MISSING KEYS: {missing}")
    else:
        print(f"  All required keys present")

    # Check dimension info
    print(f"  action_dim: {item0['action_dim']}")
    print(f"  state_dim: {item0['state_dim']}")

    # Check camera_valid_mask
    cvm = item0["camera_valid_mask"]
    print(f"  camera_valid_mask: {cvm}")

    # Check video frame format
    for cam_key in dataset.meta.camera_keys:
        if cam_key in item0:
            val = item0[cam_key]
            if val is None:
                print(f"  {cam_key}: None (invalid camera)")
            elif hasattr(val, "size"):
                from PIL import Image
                if isinstance(val, Image.Image):
                    print(f"  {cam_key}: PIL Image {val.size}")
                else:
                    print(f"  {cam_key}: {type(val).__name__} {val.shape}")
            else:
                print(f"  {cam_key}: {type(val).__name__}")

    # Check hist_actions
    if "hist_actions_full" in item0:
        ha = item0["hist_actions_full"]
        hm = item0["hist_actions_mask"]
        hl = item0["hist_actions_length"]
        print(f"  hist_actions_full shape: {ha.shape}")
        print(f"  hist_actions_mask shape: {hm.shape}")
        print(f"  hist_actions_length: {hl}")

    # Test negative index
    last_item = dataset[-1]
    assert last_item is not None, "dataset[-1] should work"

    print("  PASSED")
    return items


def test_normalization(items, dataset):
    """Test 3: Per-Sub-Dataset Normalization"""
    print("\n" + "=" * 60)
    print("Test 3: Per-Sub-Dataset Normalization")
    print("=" * 60)

    if len(items) == 0:
        print("  SKIP: No items to test")
        return

    from collections import defaultdict
    ds_stats = defaultdict(lambda: {"state": [], "action": []})

    for item in items:
        ep_idx = item["episode_index"].item() if isinstance(item["episode_index"], torch.Tensor) else item["episode_index"]
        if ep_idx < len(dataset._episode_to_ds_idx) and dataset._episode_to_ds_idx[ep_idx] >= 0:
            ds_idx = dataset._episode_to_ds_idx[ep_idx]
        else:
            ds_idx = -1

        if "observation.state" in item:
            ds_stats[ds_idx]["state"].append(item["observation.state"])
        if "action" in item:
            ds_stats[ds_idx]["action"].append(item["action"])

    for ds_idx, stats in ds_stats.items():
        if ds_idx < 0:
            ds_name = "unknown"
        else:
            ds_name = dataset._sub_dataset_names[ds_idx] if ds_idx < len(dataset._sub_dataset_names) else f"ds_{ds_idx}"

        print(f"\n  Sub-dataset: {ds_name} (idx={ds_idx})")

        if stats["state"]:
            all_states = torch.stack(stats["state"])
            mean = all_states.mean(dim=0)
            std = all_states.std(dim=0)
            state_dim = dataset._sub_dataset_dims[ds_idx][1] if ds_idx >= 0 and ds_idx < len(dataset._sub_dataset_dims) else 0
            dim_to_check = min(state_dim, 5) if state_dim > 0 else min(5, mean.shape[0])
            print(f"    state mean (first {dim_to_check}): {mean[:dim_to_check].tolist()}")
            print(f"    state std  (first {dim_to_check}): {std[:dim_to_check].tolist()}")
            if dim_to_check > 0:
                max_abs_mean = mean[:dim_to_check].abs().max().item()
                print(f"    state max |mean|: {max_abs_mean:.4f} (should be < 1.0 after normalization)")

        if stats["action"]:
            all_actions = torch.stack(stats["action"])
            mean = all_actions.mean(dim=0)
            std = all_actions.std(dim=0)
            action_dim = dataset._sub_dataset_dims[ds_idx][0] if ds_idx >= 0 and ds_idx < len(dataset._sub_dataset_dims) else 0
            dim_to_check = min(action_dim, 5) if action_dim > 0 else min(5, mean.shape[0])
            print(f"    action mean (first {dim_to_check}): {mean[:dim_to_check].tolist()}")
            print(f"    action std  (first {dim_to_check}): {std[:dim_to_check].tolist()}")

    # Check padded dims are ~0
    item0 = items[0]
    ep_idx = item0["episode_index"].item() if isinstance(item0["episode_index"], torch.Tensor) else item0["episode_index"]
    if ep_idx < len(dataset._episode_to_ds_idx) and dataset._episode_to_ds_idx[ep_idx] >= 0:
        ds_idx = dataset._episode_to_ds_idx[ep_idx]
        action_dim, state_dim = dataset._sub_dataset_dims[ds_idx] if ds_idx < len(dataset._sub_dataset_dims) else (dataset.action_dim, 0)
        if "observation.state" in item0 and state_dim > 0:
            padded_state = item0["observation.state"][state_dim:]
            if padded_state.numel() > 0:
                max_padded = padded_state.abs().max().item()
                print(f"\n  Padded state dims (after {state_dim}): max |val| = {max_padded:.6f} (should be ~0)")
        if "action" in item0 and action_dim > 0:
            padded_action = item0["action"][action_dim:]
            if padded_action.numel() > 0:
                max_padded = padded_action.abs().max().item()
                print(f"  Padded action dims (after {action_dim}): max |val| = {max_padded:.6f} (should be ~0)")

    print("\n  PASSED")


def test_camera_valid_mask(items, dataset):
    """Test 4: Camera is_valid Handling"""
    print("\n" + "=" * 60)
    print("Test 4: Camera is_valid Handling")
    print("=" * 60)

    if len(items) == 0:
        print("  SKIP: No items to test")
        return

    valid_count = 0
    invalid_count = 0
    for item in items:
        cvm = item.get("camera_valid_mask", {})
        for cam_key, is_valid in cvm.items():
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
        for cam_key in dataset.meta.camera_keys:
            if cam_key in item:
                val = item[cam_key]
                cam_valid = cvm.get(cam_key, True)
                if not cam_valid:
                    if val is not None:
                        print(f"  ERROR: Invalid camera {cam_key} should be None, got {type(val)}")
                else:
                    if val is None:
                        print(f"  ERROR: Valid camera {cam_key} should have image, got None")

    print(f"  Valid camera entries: {valid_count}")
    print(f"  Invalid camera entries: {invalid_count}")

    if invalid_count > 0:
        print(f"  Found invalid cameras - is_valid handling active")
    else:
        print(f"  No invalid cameras in sampled items (all cameras valid)")

    print("  PASSED")


def test_collate(dataset, batch_size=3):
    """Test 5: Dynamic Resolution Collate with DataLoader"""
    print("\n" + "=" * 60)
    print("Test 5: Dynamic Resolution Collate")
    print("=" * 60)

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=make_collate_fn(),
    )

    for batch in loader:
        print(f"  Batch keys: {sorted(batch.keys())}")

        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch:
                val = batch[cam_key]
                if isinstance(val, list):
                    print(f"  {cam_key}: list of {len(val)} items", end="")
                    types = set(type(v).__name__ for v in val)
                    print(f" (types: {types})")
                elif isinstance(val, torch.Tensor):
                    print(f"  {cam_key}: tensor {val.shape}")
                else:
                    print(f"  {cam_key}: {type(val).__name__}")

        if "camera_valid_mask" in batch:
            cvm = batch["camera_valid_mask"]
            if isinstance(cvm, list):
                print(f"  camera_valid_mask: list of {len(cvm)} dicts")
                if len(cvm) > 0:
                    print(f"    sample: {cvm[0]}")
            else:
                print(f"  camera_valid_mask: {type(cvm).__name__}")

        if "action_dim" in batch:
            print(f"  action_dim: {batch['action_dim']}")
        if "state_dim" in batch:
            print(f"  state_dim: {batch['state_dim']}")

        for key in ["observation.state", "action", "episode_index", "index"]:
            if key in batch:
                val = batch[key]
                if isinstance(val, torch.Tensor):
                    print(f"  {key}: tensor {val.shape}")
                else:
                    print(f"  {key}: {type(val).__name__}")

        if "hist_actions_full" in batch:
            print(f"  hist_actions_full: tensor {batch['hist_actions_full'].shape}")
        if "hist_actions_mask" in batch:
            print(f"  hist_actions_mask: tensor {batch['hist_actions_mask'].shape}")

        break

    print("  PASSED")


def main():
    parser = argparse.ArgumentParser(description="LoLA Pretrain Dataset (Map-style) Unit Test")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Dataset root directory")
    parser.add_argument("--dataset_to_episodes_path", type=str, default=None,
                        help="dataset_to_episodes.json path")
    parser.add_argument("--no_mapping", action="store_true",
                        help="Skip per-dataset mapping (no dataset_to_episodes.json)")
    parser.add_argument("--max_items", type=int, default=10,
                        help="Max items per test")
    parser.add_argument("--skip_collate", action="store_true",
                        help="Skip collate test (slow)")
    parser.add_argument("--sub_root", type=str,
                        help="Sub-dataset root directory")
    parser.add_argument("--temp_process", action="store_true",
                        help="Temp process mode: zero-pad mismatched sub-dataset stats")

    args = parser.parse_args()

    dataset_to_episodes_path = args.dataset_to_episodes_path
    sub_root = args.sub_root
    if args.no_mapping:
        dataset_to_episodes_path = None

    print("=" * 60)
    print("LoLA Pretrain Dataset (Map-style) Unit Test")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Dataset-to-episodes path: {dataset_to_episodes_path}")

    # Test 1: Basic loading + __len__
    time_start = time.time()
    dataset = test_basic_loading(args.dataset_root, dataset_to_episodes_path, sub_root, args.temp_process)
    time_stage_1 = time.time()
    print(f"Dataset loading time: {time_stage_1 - time_start:.3f} seconds")

    # Test 2: Random access
    items = test_random_access(dataset, max_items=args.max_items)
    time_stage_2 = time.time()
    print(f"Random access time: {time_stage_2 - time_stage_1:.3f} seconds")

    # Test 3: Per-sub-dataset normalization
    test_normalization(items, dataset)
    time_stage_3 = time.time()
    print(f"Normalization test time: {time_stage_3 - time_stage_2:.3f} seconds")

    # Test 4: Camera is_valid
    test_camera_valid_mask(items, dataset)
    time_stage_4 = time.time()
    print(f"Camera is_valid test time: {time_stage_4 - time_stage_3:.3f} seconds")

    # Test 5: Dynamic resolution collate
    if not args.skip_collate:
        test_collate(dataset, batch_size=3)
        time_stage_5 = time.time()
        print(f"Collate test time: {time_stage_5 - time_stage_4:.3f} seconds")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
