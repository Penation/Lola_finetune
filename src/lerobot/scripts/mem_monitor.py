#!/usr/bin/env python
"""
内存监控：定位遍历结束时内存飙升的原因。

在每个 batch 之间采样 RSS 内存，绘制内存变化曲线。

关键设计：
  - PyTorch DataLoader 的 collate_fn 在 worker 进程中运行
  - 因此不能在 collate_fn 中调用 decode_item()
  - 视频解码在主进程中进行（worker 只 yield 轻量级帧）
"""

import argparse
import os
import sys
import time
import gc

import psutil
import torch
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lola_streaming_dataset import LoLAStreamingDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig


# ── 辅助函数 ──────────────────────────────────────────────────────────

VARIABLE_LENGTH_KEYS = {"hist_actions_full", "hist_actions_mask"}


def make_passthrough_collate():
    """创建直接传递 items 列表的 collate_fn（不含视频解码）。

    关键：PyTorch DataLoader 的 collate_fn 在 worker 进程中运行，
    不能在这里做视频解码（否则 worker 进程中会产生 decoded frames，
    失去延迟解码的意义，且导致 flush 阶段内存飙升）。
    """

    def collate_fn(batch):
        return batch

    return collate_fn


def decode_and_collate(items, dataset):
    """在主进程中解码视频并 collate 为 batch dict。

    自动根据 dataset 的 decode_device 和 decode_num_threads 配置
    选择最优解码方式（CUDA/多线程CPU/串行CPU）。
    """
    if dataset.deferred_video_decode:
        items = dataset.decode_items_batch(items)

    result = {}
    for key in items[0].keys():
        values = [item[key] for item in items]
        if key == "task":
            result[key] = values
        elif key in VARIABLE_LENGTH_KEYS and isinstance(values[0], torch.Tensor):
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


# ── 主函数 ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=15300)
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument("--no_deferred", action="store_true",
                        help="使用 make_frame 模式（不延迟解码）")
    args = parser.parse_args()

    # 构建配置
    dataset_metadata = LeRobotDatasetMetadata(None, root=args.dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    action_dim = features["action"].shape[0]

    config = LoLAConfig(
        vlm_model_name="Qwen/Qwen3.5-4B",
        vlm_path="/tmp/dummy",
        action_dim=action_dim,
        action_chunk_size=args.action_chunk_size,
        pred_chunk_size=50,
        n_obs_steps=1,
        input_features={k: v for k, v in features.items() if v.type != FeatureType.ACTION},
        output_features={k: v for k, v in features.items() if v.type == FeatureType.ACTION},
        train_vlm=False,
        load_full_history=True,
        max_history_length=args.max_history_length,
        history_padding_side="left",
    )

    fps = dataset_metadata.fps
    delta_timestamps = {
        "observation.state": [i / fps for i in config.observation_delta_indices],
        "action": [i / fps for i in config.action_delta_indices],
    }
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    dataset = LoLAStreamingDataset(
        repo_id=None,
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        history_padding_side="left",
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        streaming=True,
        buffer_size=args.buffer_size,
        seed=42,
        shuffle=True,
        deferred_video_decode=not args.no_deferred,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=make_passthrough_collate(),
        pin_memory=False,
    )

    process = psutil.Process()
    total_frames = dataset_metadata.total_frames
    total_batches_est = total_frames // args.batch_size

    def get_total_rss_gb():
        """获取主进程 + 所有子进程的总 RSS。"""
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / 1024**3

    print(f"数据集: {total_frames} frames, 预计 ~{total_batches_est} batches")
    print(f"num_workers={args.num_workers}, batch_size={args.batch_size}")
    print(f"buffer_size={args.buffer_size}, deferred_video_decode={not args.no_deferred}")
    print()

    # 内存采样
    mem_samples = []
    batch_count = 0
    sample_count = 0
    start_time = time.time()
    peak_rss = 0
    peak_batch = 0

    # 采样初始内存
    gc.collect()
    mem_before = get_total_rss_gb()
    print(f"初始内存 (主进程+子进程): {mem_before:.2f} GB")

    for items in loader:
        batch_count += 1
        sample_count += args.batch_size

        # 在主进程中解码视频并 collate
        batch = decode_and_collate(items, dataset)

        # 强制释放 batch（不持有引用）
        del batch, items

        # 密集采样：每 100 batch，接近结束时每 10 batch
        sample_interval = 100
        if batch_count > total_batches_est - 1000:
            sample_interval = 10
        if batch_count > total_batches_est - 200:
            sample_interval = 5

        if batch_count % sample_interval == 0 or batch_count <= 5:
            gc.collect()
            rss_gb = get_total_rss_gb()
            pct = sample_count / total_frames * 100
            mem_samples.append((batch_count, rss_gb, pct))
            if rss_gb > peak_rss:
                peak_rss = rss_gb
                peak_batch = batch_count
            print(f"  batch {batch_count:>6d}, {sample_count:>7d} samples "
                  f"({pct:5.1f}%), RSS={rss_gb:.2f} GB")

        # 在接近结束时加密采样
        if batch_count > total_batches_est - 500 and batch_count % 50 == 0:
            gc.collect()
            rss_gb = get_total_rss_gb()
            pct = sample_count / total_frames * 100
            mem_samples.append((batch_count, rss_gb, pct))
            if rss_gb > peak_rss:
                peak_rss = rss_gb
                peak_batch = batch_count
            print(f"  batch {batch_count:>6d}, {sample_count:>7d} samples "
                  f"({pct:5.1f}%), RSS={rss_gb:.2f} GB  *** NEAR END ***")

        if batch_count >= args.max_batches:
            break

    # 最终内存
    gc.collect()
    mem_after = get_total_rss_gb()
    elapsed = time.time() - start_time

    print()
    print(f"遍历完成: {batch_count} batches, {sample_count} samples, {elapsed:.1f}s")
    print(f"初始 RSS: {mem_before:.2f} GB, 最终 RSS: {mem_after:.2f} GB")
    print(f"峰值 RSS: {peak_rss:.2f} GB (batch {peak_batch})")

    # 分析内存变化
    if mem_samples:
        # 找到内存增长最快的区间
        for i in range(1, len(mem_samples)):
            delta = mem_samples[i][1] - mem_samples[i-1][1]
            if delta > 1.0:
                print(f"  内存跳升: batch {mem_samples[i-1][0]}→{mem_samples[i][0]}, "
                      f"{mem_samples[i-1][1]:.2f}→{mem_samples[i][1]:.2f} GB "
                      f"(+{delta:.2f} GB)")


if __name__ == "__main__":
    main()
