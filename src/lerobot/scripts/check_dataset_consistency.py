#!/usr/bin/env python
"""
数据集一致性检查脚本

检查同设置的两个数据集（LoLADataset vs LoLAStreamingDataset）顺序读取的数据是否完全一致。

由于两种数据集的遍历顺序可能不同（map-style 按全局索引，streaming 按分片迭代且可能跳帧），
本脚本按 episode 分批处理：逐 episode 收集 streaming items 并与 map 数据集对比，对比完释放，
避免一次性加载所有数据导致内存溢出。

使用方法:
    python src/lerobot/scripts/check_dataset_consistency.py \
        --dataset_repo_id lerobot/pusht \
        --max_history_length 100

    python src/lerobot/scripts/check_dataset_consistency.py \
        --dataset_root /mnt/data/lerobot-dataset \
        --max_history_length 100

    python src/lerobot/scripts/check_dataset_consistency.py \
        --dataset_repo_id lerobot/pusht \
        --max_check_frames 500

    python src/lerobot/scripts/check_dataset_consistency.py \
        --dataset_repo_id lerobot/pusht \
        --compare_keys action observation.state
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lola_dataset import LoLADataset
from lerobot.datasets.lola_streaming_dataset import LoLAStreamingDataset


def build_delta_timestamps(repo_id, root, n_obs_steps=1, action_chunk_size=10, pred_chunk_size=50):
    """根据数据集 FPS 构建 delta_timestamps"""
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = metadata.fps

    observation_delta_indices = list(range(1 - n_obs_steps, 1))
    action_delta_indices = list(range(1, 1 + pred_chunk_size))

    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in action_delta_indices]
    for key in metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in observation_delta_indices]

    return delta_timestamps, metadata


def _item_key(item):
    """提取 (episode_index, frame_index) 作为匹配键"""
    ep = item["episode_index"]
    fr = item["frame_index"]
    if isinstance(ep, torch.Tensor):
        ep = ep.item()
    if isinstance(fr, torch.Tensor):
        fr = fr.item()
    return (ep, fr)


def compare_tensors(t1, t2, atol=1e-6, rtol=1e-5):
    """比较两个 tensor 是否一致，返回 (match, detail)"""
    if t1.shape != t2.shape:
        return False, f"shape mismatch: {t1.shape} vs {t2.shape}"

    if t1.dtype != t2.dtype:
        if t1.is_floating_point() and t2.is_floating_point():
            t1 = t1.to(torch.float64)
            t2 = t2.to(torch.float64)
        elif t1.dtype != t2.dtype:
            return False, f"dtype mismatch: {t1.dtype} vs {t2.dtype}"

    if t1.is_floating_point():
        if not torch.allclose(t1, t2, atol=atol, rtol=rtol):
            diff = (t1 - t2).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            mismatch_ratio = (diff > atol).float().mean().item()
            return False, (
                f"values differ: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, "
                f"mismatch_ratio={mismatch_ratio:.4f}"
            )
    else:
        if not torch.equal(t1, t2):
            mismatch_count = (t1 != t2).sum().item()
            total = t1.numel()
            return False, f"values differ: {mismatch_count}/{total} elements mismatch"

    return True, "OK"


def compare_items(item_map, item_stream, keys_to_compare=None, atol=1e-6, rtol=1e-5):
    """
    比较两个数据项是否一致。

    Returns:
        list of (key, match, detail)
    """
    results = []

    common_keys = set(item_map.keys()) & set(item_stream.keys())
    if keys_to_compare:
        common_keys = common_keys & set(keys_to_compare)

    priority_keys = [
        "action",
        "observation.state",
        "hist_actions_full", "hist_actions_mask", "hist_actions_length",
        "action_is_pad",
    ]
    remaining_keys = sorted(common_keys - set(priority_keys) - {"episode_index", "frame_index", "timestamp", "index", "task"})
    ordered_keys = [k for k in priority_keys if k in common_keys] + remaining_keys

    for key in ordered_keys:
        v1 = item_map[key]
        v2 = item_stream[key]

        if isinstance(v1, str) and isinstance(v2, str):
            if v1 == v2:
                results.append((key, True, "OK (str)"))
            else:
                results.append((key, False, f"str mismatch: '{v1[:50]}' vs '{v2[:50]}'"))
            continue

        if isinstance(v1, list) and isinstance(v2, list):
            if v1 == v2:
                results.append((key, True, "OK (list)"))
            else:
                results.append((key, False, "list mismatch"))
            continue

        if not isinstance(v1, torch.Tensor) or not isinstance(v2, torch.Tensor):
            results.append((key, True, f"skipped ({type(v1).__name__} vs {type(v2).__name__})"))
            continue

        match, detail = compare_tensors(v1, v2, atol=atol, rtol=rtol)
        results.append((key, match, detail))

    return results


def main():
    parser = argparse.ArgumentParser(description="Check dataset consistency between LoLADataset and LoLAStreamingDataset")

    # 数据集参数
    parser.add_argument("--dataset_repo_id", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)

    # 比较参数
    parser.add_argument("--max_check_frames", type=int, default=1000,
                        help="Maximum number of frames to check from map dataset")
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--history_padding_side", type=str, default="left", choices=["left", "right"])

    # 模型参数（用于构建 delta_timestamps）
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--pred_chunk_size", type=int, default=50)

    # 流式数据集参数
    parser.add_argument("--buffer_size", type=int, default=1000)

    # 比较选项
    parser.add_argument("--compare_keys", type=str, nargs="*", default=None,
                        help="Only compare specified keys")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for float comparison")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for float comparison")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame details")
    parser.add_argument("--stop_on_mismatch", type=int, default=10,
                        help="Stop after N mismatches (0 = never stop)")

    args = parser.parse_args()

    if args.dataset_repo_id is None and args.dataset_root is None:
        raise ValueError("Either --dataset_repo_id or --dataset_root must be provided.")

    repo_id = args.dataset_repo_id
    root = args.dataset_root

    print("=" * 70)
    print("Dataset Consistency Check: LoLADataset vs LoLAStreamingDataset")
    print("=" * 70)
    print(f"Dataset: {repo_id or root}")
    print(f"Max check frames: {args.max_check_frames}")
    print(f"Max history length: {args.max_history_length}")
    print(f"Action chunk size: {args.action_chunk_size}")
    print(f"History padding side: {args.history_padding_side}")
    print(f"Tolerance: atol={args.atol}, rtol={args.rtol}")
    print()

    # 构建 delta_timestamps
    print("Building delta_timestamps...")
    delta_timestamps, metadata = build_delta_timestamps(
        repo_id, root,
        n_obs_steps=args.n_obs_steps,
        action_chunk_size=args.action_chunk_size,
        pred_chunk_size=args.pred_chunk_size,
    )
    print(f"  FPS: {metadata.fps}")
    print(f"  Total episodes: {metadata.total_episodes}")
    print(f"  Total frames: {metadata.total_frames}")
    print()

    # 创建 map-style 数据集
    print("Creating LoLADataset (map-style)...")
    t0 = time.time()
    map_dataset = LoLADataset(
        repo_id=repo_id,
        root=root,
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        history_padding_side=args.history_padding_side,
        delta_timestamps=delta_timestamps,
    )
    print(f"  Done in {time.time() - t0:.1f}s, total samples: {len(map_dataset)}")
    print()

    # 获取 episode 边界信息，用于按 episode 分批比较
    episode_ranges = {}  # ep_idx -> (start_idx, end_idx)
    for ep_idx, ep in enumerate(map_dataset.meta.episodes):
        episode_ranges[ep_idx] = (ep["dataset_from_index"], ep["dataset_to_index"])

    # 创建 streaming 数据集
    print("Creating LoLAStreamingDataset (streaming)...")
    t0 = time.time()
    stream_dataset = LoLAStreamingDataset(
        repo_id=repo_id,
        root=root,
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        history_padding_side=args.history_padding_side,
        delta_timestamps=delta_timestamps,
        streaming=True,
        buffer_size=args.buffer_size,
        seed=42,
        shuffle=False,
    )
    print(f"  Done in {time.time() - t0:.1f}s")
    print()

    # 按分批收集 streaming items 并比较
    # 策略：从 streaming 迭代器中逐条读取，按 episode_index 分组到桶中
    # 当某个 episode 的桶收集完成后（遇到下一个 episode），立即与 map 对比并释放
    print("Comparing frames (batched by episode, matched by episode_index + frame_index)...")
    print("-" * 70)

    total_mismatches = 0
    total_missing_in_stream = 0
    total_missing_in_map = 0
    total_frames_checked = 0
    total_stream_items = 0
    key_mismatch_counts = {}
    mismatch_details = []

    check_count = min(args.max_check_frames, len(map_dataset))
    # 计算需要检查的 episode 范围
    max_ep_to_check = -1
    for idx in range(check_count):
        try:
            item = map_dataset[idx]
            ep = item["episode_index"]
            if isinstance(ep, torch.Tensor):
                ep = ep.item()
            max_ep_to_check = max(max_ep_to_check, ep)
        except Exception:
            pass

    stream_iter = iter(stream_dataset)
    current_ep_bucket = {}  # (ep, fr) -> item，当前 episode 的 streaming items
    current_ep_idx = None
    finished_streaming = False

    def flush_episode(ep_idx, bucket):
        """将一个 episode 的 streaming bucket 与 map dataset 对比，然后释放"""
        nonlocal total_mismatches, total_missing_in_stream, total_missing_in_map
        nonlocal total_frames_checked, key_mismatch_counts, mismatch_details

        if ep_idx not in episode_ranges:
            return

        start_idx, end_idx = episode_ranges[ep_idx]

        # 遍历 map dataset 中该 episode 的所有帧
        for idx in range(start_idx, end_idx):
            if idx >= check_count:
                break

            try:
                item_map = map_dataset[idx]
            except Exception as e:
                if args.verbose:
                    print(f"    [Idx {idx}] LoLADataset error: {e}")
                continue

            key = _item_key(item_map)
            ep, fr = key

            if key not in bucket:
                total_missing_in_stream += 1
                if args.verbose or total_missing_in_stream <= 20:
                    print(f"  [Idx {idx}] ep={ep}, fr={fr} MISSING in streaming")
                continue

            item_stream = bucket[key]
            total_frames_checked += 1

            results = compare_items(
                item_map, item_stream,
                keys_to_compare=args.compare_keys,
                atol=args.atol, rtol=args.rtol,
            )

            frame_mismatches = []
            for k, match, detail in results:
                if not match:
                    frame_mismatches.append((k, detail))
                    key_mismatch_counts[k] = key_mismatch_counts.get(k, 0) + 1

            if frame_mismatches:
                total_mismatches += 1
                msg = f"  [Idx {idx}] ep={ep}, fr={fr} MISMATCH:"
                for k, detail in frame_mismatches:
                    msg += f"\n    {k}: {detail}"
                if args.verbose or len(mismatch_details) < 10:
                    print(msg)
                mismatch_details.append({
                    "idx": idx, "episode": ep, "frame": fr,
                    "mismatches": frame_mismatches,
                })

            if args.stop_on_mismatch > 0 and total_mismatches >= args.stop_on_mismatch:
                return True  # signal stop

        # 检查 streaming bucket 中有但 map 中没有的帧
        map_keys = set()
        for idx in range(start_idx, min(end_idx, check_count)):
            try:
                map_keys.add(_item_key(map_dataset[idx]))
            except Exception:
                pass

        extra = set(bucket.keys()) - map_keys
        total_missing_in_map += len(extra)
        if extra and (args.verbose or total_missing_in_map <= 20):
            for k in sorted(extra)[:5]:
                print(f"  Extra in streaming: ep={k[0]}, fr={k[1]} (not in map range)")

        return False

    # 主循环：逐条从 streaming 读取，按 episode 分桶
    should_stop = False
    for item in stream_iter:
        key = _item_key(item)
        ep_idx = key[0]

        # 跳过超出检查范围的 episode
        if ep_idx > max_ep_to_check:
            # flush 最后一个 episode
            if current_ep_bucket:
                should_stop = flush_episode(current_ep_idx, current_ep_bucket)
                current_ep_bucket = {}
            break

        # 新 episode 开始，flush 上一个
        if current_ep_idx is not None and ep_idx != current_ep_idx:
            should_stop = flush_episode(current_ep_idx, current_ep_bucket)
            current_ep_bucket = {}
            if should_stop:
                break

        current_ep_idx = ep_idx
        current_ep_bucket[key] = item
        total_stream_items += 1

    # flush 最后一个 episode
    if not should_stop and current_ep_bucket:
        flush_episode(current_ep_idx, current_ep_bucket)

    # 汇总报告
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Map dataset samples in range: {check_count}")
    print(f"Streaming dataset items read: {total_stream_items}")
    print(f"Frames matched and compared: {total_frames_checked}")
    print(f"Frames missing in streaming: {total_missing_in_stream}")
    print(f"Extra frames in streaming (not in map range): {total_missing_in_map}")
    print(f"Mismatches: {total_mismatches}")

    if total_frames_checked > 0:
        match_rate = (total_frames_checked - total_mismatches) / total_frames_checked * 100
        print(f"Match rate: {match_rate:.2f}%")

    if total_missing_in_stream > 0:
        print(f"\n  Missing rate (in streaming): {total_missing_in_stream}/{check_count} "
              f"({total_missing_in_stream / check_count * 100:.2f}%)")

    if key_mismatch_counts:
        print()
        print("Mismatch by key:")
        for k, count in sorted(key_mismatch_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {count} frames")

    if mismatch_details:
        print()
        print("First mismatches detail:")
        for i, detail in enumerate(mismatch_details[:5]):
            print(f"  [{i}] idx={detail['idx']}, ep={detail['episode']}, fr={detail['frame']}")
            for k, d in detail["mismatches"]:
                print(f"      {k}: {d}")

    if total_mismatches == 0 and total_missing_in_stream == 0:
        print()
        print("PASS: All checked frames are consistent!")
    elif total_mismatches == 0 and total_missing_in_stream > 0:
        print()
        print(f"PARTIAL PASS: Matched frames are consistent, but {total_missing_in_stream} frames missing in streaming.")
    else:
        print()
        print(f"FAIL: {total_mismatches} frames have mismatches.")

    print("=" * 70)


if __name__ == "__main__":
    main()
