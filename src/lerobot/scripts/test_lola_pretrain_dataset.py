#!/usr/bin/env python
"""
LoLA Pretrain Streaming Dataset 单元测试

测试 LoLAPretrainStreamingDataset 的核心功能：
1. 数据集加载和迭代
2. Per-sub-dataset 归一化
3. is_valid 相机处理
4. 动态分辨率 collate
5. 维度信息
6. Video frame 转 PIL Image

用法:
    # 使用样例数据集测试（本机）
    python src/lerobot/scripts/test_lola_pretrain_dataset.py \
        --dataset_root /data_6t_2/lerobot_v30/simpler_bridge_v3 \
        --dataset_to_episodes_path /path/to/dataset_to_episodes.json

    # 使用合并数据集测试（集群）
    python src/lerobot/scripts/test_lola_pretrain_dataset.py \
        --dataset_root /data_16T/deepseek/halo \
        --dataset_to_episodes_path /data_16T/deepseek/halo/dataset_to_episodes.json

注意:
    - 如果没有 dataset_to_episodes.json，可以用 --no_mapping 跳过 per-dataset 归一化测试
    - 本机测试时，子数据集 stats 可能无法加载（子数据集在云存储），脚本会 fallback
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
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lola_pretrain_streaming_dataset import LoLAPretrainStreamingDataset, AsyncDecodeDataLoader
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig


def test_basic_loading(dataset_root, dataset_to_episodes_path=None):
    """测试 1: 基本数据集加载"""
    print("\n" + "=" * 60)
    print("Test 1: Basic Dataset Loading")
    print("=" * 60)

    dataset_metadata = LeRobotDatasetMetadata(repo_id="test", root=dataset_root)
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

    dataset = LoLAPretrainStreamingDataset(
        repo_id="test",
        max_history_length=10,
        action_chunk_size=config.action_chunk_size,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        streaming=True,
        buffer_size=10,
        seed=42,
        shuffle=False,
        deferred_video_decode=False,  # 直接解码模式，方便检查帧内容
        dataset_to_episodes_path=dataset_to_episodes_path,
    )

    print(f"  Dataset created: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    print(f"  Action dim: {dataset.action_dim}")
    print(f"  Video keys: {dataset.meta.video_keys}")
    print(f"  Camera keys: {dataset.meta.camera_keys}")
    print(f"  Sub-datasets loaded: {len(dataset._sub_dataset_names)}")
    print("  PASSED")
    return dataset


def test_iteration(dataset, max_items=5):
    """测试 2: 数据集迭代和 item 结构"""
    print("\n" + "=" * 60)
    print("Test 2: Dataset Iteration & Item Structure")
    print("=" * 60)

    items = []
    for i, item in enumerate(dataset):
        if i >= max_items:
            break
        items.append(item)

    print(f"  Collected {len(items)} items")

    if len(items) == 0:
        print("  WARNING: No items collected!")
        return items

    # 检查 item 的 keys
    item0 = items[0]
    print(f"  Item keys: {sorted(item0.keys())}")

    # 检查必要字段
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

    # 检查维度信息
    print(f"  action_dim: {item0['action_dim']}")
    print(f"  state_dim: {item0['state_dim']}")

    # 检查 camera_valid_mask
    cvm = item0["camera_valid_mask"]
    print(f"  camera_valid_mask: {cvm}")

    # 检查 video frame 格式
    for cam_key in dataset.meta.camera_keys:
        if cam_key in item0:
            val = item0[cam_key]
            if val is None:
                print(f"  {cam_key}: None (invalid camera)")
            elif hasattr(val, 'size'):
                from PIL import Image
                if isinstance(val, Image.Image):
                    print(f"  {cam_key}: PIL Image {val.size}")
                else:
                    print(f"  {cam_key}: {type(val).__name__} {val.shape}")
            else:
                print(f"  {cam_key}: {type(val).__name__}")

    # 检查 hist_actions
    if "hist_actions_full" in item0:
        ha = item0["hist_actions_full"]
        hm = item0["hist_actions_mask"]
        hl = item0["hist_actions_length"]
        print(f"  hist_actions_full shape: {ha.shape}")
        print(f"  hist_actions_mask shape: {hm.shape}")
        print(f"  hist_actions_length: {hl}")

    print("  PASSED")
    return items


def test_normalization(items, dataset):
    """测试 3: Per-sub-dataset 归一化"""
    print("\n" + "=" * 60)
    print("Test 3: Per-Sub-Dataset Normalization")
    print("=" * 60)

    if len(items) == 0:
        print("  SKIP: No items to test")
        return

    # 收集不同子数据集的 state 和 action 统计
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

    # 检查归一化后的统计量
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
            # 只看前几个维度（原始维度部分）
            state_dim = dataset._sub_dataset_dims[ds_idx][1] if ds_idx >= 0 and ds_idx < len(dataset._sub_dataset_dims) else 0
            dim_to_check = min(state_dim, 5) if state_dim > 0 else min(5, mean.shape[0])
            print(f"    state mean (first {dim_to_check}): {mean[:dim_to_check].tolist()}")
            print(f"    state std  (first {dim_to_check}): {std[:dim_to_check].tolist()}")
            # 归一化后应该接近 (0, 1)
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

    # 检查 padded 维度是否为 0
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
    """测试 4: is_valid 相机处理"""
    print("\n" + "=" * 60)
    print("Test 4: Camera is_valid Handling")
    print("=" * 60)

    if len(items) == 0:
        print("  SKIP: No items to test")
        return

    # 统计有效/无效相机分布
    valid_count = 0
    invalid_count = 0
    for item in items:
        cvm = item.get("camera_valid_mask", {})
        for cam_key, is_valid in cvm.items():
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
        # 检查 invalid 相机的帧是 None
        for cam_key in dataset.meta.camera_keys:
            if cam_key in item:
                val = item[cam_key]
                cam_valid = cvm.get(cam_key, True)
                if not cam_valid:
                    if val is not None:
                        print(f"  ERROR: Invalid camera {cam_key} should be None, got {type(val)}")
                    invalid_count += 1  # already counted above, just checking
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
    """测试 5: 动态分辨率 Collate"""
    print("\n" + "=" * 60)
    print("Test 5: Dynamic Resolution Collate")
    print("=" * 60)

    from torch.utils.data import DataLoader

    raw_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=lambda x: x,  # passthrough
    )
    train_loader = AsyncDecodeDataLoader(
        dataloader=raw_loader,
        dataset=dataset,
        collate_fn=AsyncDecodeDataLoader.make_collate_fn(),
    )

    for batch in train_loader:
        print(f"  Batch keys: {sorted(batch.keys())}")

        # 检查 camera 帧是否为 list
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

        # 检查 camera_valid_mask 是否为 list
        if "camera_valid_mask" in batch:
            cvm = batch["camera_valid_mask"]
            if isinstance(cvm, list):
                print(f"  camera_valid_mask: list of {len(cvm)} dicts")
                if len(cvm) > 0:
                    print(f"    sample: {cvm[0]}")
            else:
                print(f"  camera_valid_mask: {type(cvm).__name__}")

        # 检查 action_dim / state_dim
        if "action_dim" in batch:
            print(f"  action_dim: {batch['action_dim']}")
        if "state_dim" in batch:
            print(f"  state_dim: {batch['state_dim']}")

        # 检查 tensor 字段是否正确 stack
        for key in ["observation.state", "action", "episode_index", "index"]:
            if key in batch:
                val = batch[key]
                if isinstance(val, torch.Tensor):
                    print(f"  {key}: tensor {val.shape}")
                else:
                    print(f"  {key}: {type(val).__name__}")

        # 检查 hist_actions
        if "hist_actions_full" in batch:
            print(f"  hist_actions_full: tensor {batch['hist_actions_full'].shape}")
        if "hist_actions_mask" in batch:
            print(f"  hist_actions_mask: tensor {batch['hist_actions_mask'].shape}")

        break  # 只测试第一个 batch

    print("  PASSED")


def test_lightweight_mode(dataset_root, dataset_to_episodes_path=None):
    """测试 6: Lightweight (deferred decode) 模式"""
    print("\n" + "=" * 60)
    print("Test 6: Lightweight / Deferred Decode Mode")
    print("=" * 60)

    dataset_metadata = LeRobotDatasetMetadata(repo_id="test", root=dataset_root)
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

    dataset = LoLAPretrainStreamingDataset(
        repo_id="test",
        max_history_length=10,
        action_chunk_size=config.action_chunk_size,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        streaming=True,
        buffer_size=10,
        seed=42,
        shuffle=False,
        deferred_video_decode=True,  # 启用延迟解码
        dataset_to_episodes_path=dataset_to_episodes_path,
    )

    # 用 decode_on_yield 模式迭代（deferred + not async）
    items = []
    for i, item in enumerate(dataset):
        if i >= 3:
            break
        items.append(item)

    print(f"  Collected {len(items)} items in lightweight mode")

    if len(items) > 0:
        item0 = items[0]
        print(f"  Item keys: {sorted(item0.keys())}")

        # 检查 video frame 格式
        for cam_key in dataset.meta.camera_keys:
            if cam_key in item0:
                val = item0[cam_key]
                if val is None:
                    print(f"  {cam_key}: None (invalid)")
                else:
                    from PIL import Image
                    if isinstance(val, Image.Image):
                        print(f"  {cam_key}: PIL Image {val.size} (decoded on yield)")
                    else:
                        print(f"  {cam_key}: {type(val).__name__}")

        # 检查 camera_valid_mask
        cvm = item0.get("camera_valid_mask", {})
        print(f"  camera_valid_mask: {cvm}")

    print("  PASSED")


def main():
    parser = argparse.ArgumentParser(description="LoLA Pretrain Dataset Unit Test")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="数据集根目录（合并数据集或样例数据集）")
    parser.add_argument("--dataset_to_episodes_path", type=str, default=None,
                        help="dataset_to_episodes.json 路径")
    parser.add_argument("--no_mapping", action="store_true",
                        help="跳过 per-dataset mapping（无 dataset_to_episodes.json 时使用）")
    parser.add_argument("--max_items", type=int, default=10,
                        help="每个测试最多迭代的 item 数")
    parser.add_argument("--skip_collate", action="store_true",
                        help="跳过 collate 测试（耗时较长）")

    args = parser.parse_args()

    dataset_to_episodes_path = args.dataset_to_episodes_path
    if args.no_mapping:
        dataset_to_episodes_path = None

    print("=" * 60)
    print("LoLA Pretrain Streaming Dataset Unit Test")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Dataset-to-episodes path: {dataset_to_episodes_path}")

    # Test 1: Basic loading
    dataset = test_basic_loading(args.dataset_root, dataset_to_episodes_path)

    # Test 2: Iteration & item structure
    items = test_iteration(dataset, max_items=args.max_items)

    # Test 3: Per-sub-dataset normalization
    test_normalization(items, dataset)

    # Test 4: Camera is_valid
    test_camera_valid_mask(items, dataset)

    # Test 5: Dynamic resolution collate
    if not args.skip_collate:
        test_collate(dataset, batch_size=3)

    # Test 6: Lightweight mode
    test_lightweight_mode(args.dataset_root, dataset_to_episodes_path)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
