#!/usr/bin/env python
"""
重新计算数据集的统计信息。

当数据集的 stats.json 与实际特征定义不匹配时，使用此脚本重新计算。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.compute_stats import get_feature_stats, RunningQuantileStats


def recompute_stats(dataset_root: str, output_path: str | None = None, sample_rate: float = 0.1):
    """
    重新计算数据集的统计信息。

    Args:
        dataset_root: 数据集根目录
        output_path: 输出路径（默认覆盖原 stats.json）
        sample_rate: 采样率（默认 10%）
    """
    dataset_root = Path(dataset_root)

    print(f"Loading dataset from {dataset_root}...")

    # 加载数据集（不使用 delta_timestamps，加载原始数据）
    dataset = LeRobotDataset(
        repo_id="local_dataset",
        root=dataset_root,
        delta_timestamps=None,
    )

    print(f"Dataset info:")
    print(f"  - Total episodes: {dataset.num_episodes}")
    print(f"  - Total frames: {dataset.num_frames}")
    print(f"  - Features: {list(dataset.features.keys())}")

    # 打印特征形状
    print("\nFeature shapes:")
    for key, feat in dataset.features.items():
        print(f"  {key}: {feat['shape']}, dtype={feat['dtype']}")

    # 计算采样数量
    num_samples = max(100, int(len(dataset) * sample_rate))
    num_samples = min(num_samples, len(dataset))
    print(f"\nSampling {num_samples} frames for stats computation...")

    # 采样索引
    sample_indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    # 初始化统计收集器
    stats_collectors = {}
    for key, feat in dataset.features.items():
        if feat['dtype'] in ['image', 'video']:
            continue  # 跳过图像/视频
        if feat['dtype'] == 'string':
            continue  # 跳过字符串

        shape = feat['shape']
        if len(shape) == 0:
            continue  # 跳过标量

        # 创建统计收集器
        stats_collectors[key] = RunningQuantileStats()

    # 收集统计数据
    for idx in tqdm(sample_indices, desc="Computing stats"):
        item = dataset[idx]

        for key in stats_collectors:
            if key in item:
                data = item[key]
                if isinstance(data, torch.Tensor):
                    data = data.numpy()

                # 跳过标量（0维或shape为空的数组）
                if data.ndim == 0 or (data.ndim == 1 and len(data) == 1):
                    # 对于标量，跳过或者特殊处理
                    continue

                # 确保是 2D 数组 (samples, features)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                elif data.ndim > 2:
                    data = data.reshape(-1, data.shape[-1])

                stats_collectors[key].update(data)

    # 获取统计结果
    stats = {}
    for key, collector in stats_collectors.items():
        try:
            stats[key] = collector.get_statistics()
            # 转换 numpy 数组为列表以便 JSON 序列化
            stats[key] = {k: v.tolist() for k, v in stats[key].items()}
        except ValueError as e:
            print(f"Warning: Could not compute stats for {key}: {e}")

    # 加载原有 stats 文件中的图像统计（保留图像统计不变）
    stats_path = dataset_root / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            original_stats = json.load(f)

        # 保留图像统计
        for key in original_stats:
            if key.startswith('observation.images.') or key.startswith('observation.videos.'):
                if key not in stats:
                    stats[key] = original_stats[key]

    # 输出路径
    if output_path is None:
        output_path = stats_path
    else:
        output_path = Path(output_path)

    # 保存统计信息
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to {output_path}")

    # 打印新统计的形状
    print("\nNew stats shapes:")
    for key, stat in stats.items():
        if 'mean' in stat:
            mean_shape = np.array(stat['mean']).shape
            print(f"  {key}: mean shape = {mean_shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute dataset statistics")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for stats.json")
    parser.add_argument("--sample_rate", type=float, default=0.1, help="Sampling rate (default: 0.1)")

    args = parser.parse_args()

    recompute_stats(args.dataset_root, args.output_path, args.sample_rate)
