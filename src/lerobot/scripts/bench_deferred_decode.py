#!/usr/bin/env python
"""
Benchmark: 延迟视频解码 vs 当前方案的速度对比。

测试矩阵：
  - 方案 A: 当前方案（make_frame 中解码视频，buffer 存完整帧）
  - 方案 B: 延迟解码（buffer 存轻量级帧，yield 时解码视频）

网络延迟模拟：
  - 无延迟（本地磁盘基线）
  - 模拟文件打开延迟（新 video file 打开时 sleep）
  - 模拟帧 seek 延迟（每次 decode 帧时 sleep）

使用方法：
    python src/lerobot/scripts/bench_deferred_decode.py \
        --dataset_root /data_6t_2/lerobot_v30/simpler_bridge_v3/ \
        --num_workers 8 \
        --batch_size 16 \
        --max_batches 200 \
        --file_open_delay_ms 100 \
        --frame_seek_delay_ms 5
"""

import argparse
import os
import sys
import time
import tracemalloc

import numpy as np
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
from lerobot.datasets.utils import dataset_to_policy_features, item_to_torch
from lerobot.policies.lola import LoLAConfig


VARIABLE_LENGTH_KEYS = {"hist_actions_full", "hist_actions_mask"}


def make_passthrough_collate():
    """创建直接传递 items 列表的 collate_fn（不含视频解码）。

    关键：PyTorch DataLoader 的 collate_fn 在 worker 进程中运行，
    不能在这里做视频解码。视频解码在主进程中通过 decode_and_collate() 执行。
    """

    def collate_fn(batch):
        return batch

    return collate_fn


def decode_and_collate(items, dataset):
    """在主进程中解码视频并 collate 为 batch dict。"""
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


def create_dataset_and_loader(args, delta_timestamps, mode="current"):
    """创建数据集和 DataLoader。

    Args:
        mode: "current" 使用当前方案, "deferred" 使用延迟解码方案
    """
    dataset = LoLAStreamingDataset(
        repo_id=args.dataset_repo_id,
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        history_padding_side=args.history_padding_side,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        streaming=True,
        buffer_size=args.buffer_size,
        seed=42,
        shuffle=True,
        deferred_video_decode=(mode == "deferred"),
    )

    # 注入延迟模拟（如果指定）
    if args.file_open_delay_ms > 0 or args.frame_seek_delay_ms > 0:
        _inject_network_delay(dataset, args.file_open_delay_ms, args.frame_seek_delay_ms)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=make_passthrough_collate(),
        pin_memory=False,  # benchmark 不需要 pin_memory
    )
    return dataset, loader


def _inject_network_delay(dataset, file_open_delay_ms, frame_seek_delay_ms):
    """注入模拟的网络延迟到视频解码流程中。

    通过 monkey-patch BoundedVideoDecoderCache.get_decoder 和
    decode_video_frames_torchcodec 来模拟：
    - file_open_delay_ms: 新文件打开时的延迟（模拟网络传输文件头）
    - frame_seek_delay_ms: 每次帧解码的延迟（模拟网络 seek + read）
    """
    from lerobot.datasets.lola_streaming_dataset import BoundedVideoDecoderCache
    from lerobot.datasets.video_utils import decode_video_frames_torchcodec

    original_get_decoder = BoundedVideoDecoderCache.get_decoder
    original_decode = decode_video_frames_torchcodec

    file_open_delay_s = file_open_delay_ms / 1000.0
    frame_seek_delay_s = frame_seek_delay_ms / 1000.0

    def delayed_get_decoder(self, video_path):
        # 检查是否是新文件（不在缓存中）
        with self._lock:
            is_new = video_path not in self._cache
        if is_new and file_open_delay_s > 0:
            time.sleep(file_open_delay_s)
        return original_get_decoder(self, video_path)

    def delayed_decode(video_path, timestamps, tolerance_s, decoder_cache=None):
        if frame_seek_delay_s > 0:
            time.sleep(frame_seek_delay_s)
        return original_decode(video_path, timestamps, tolerance_s, decoder_cache=decoder_cache)

    # Monkey-patch
    BoundedVideoDecoderCache.get_decoder = delayed_get_decoder
    import lerobot.datasets.lola_streaming_dataset as lola_module
    lola_module.decode_video_frames_torchcodec = delayed_decode


def run_benchmark(args, mode, delta_timestamps, label=""):
    """运行单个 benchmark 配置。"""
    import gc
    gc.collect()

    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"  Mode: {mode}")
    print(f"  num_workers: {args.num_workers}, batch_size: {args.batch_size}")
    print(f"  max_batches: {args.max_batches}")
    print(f"  file_open_delay: {args.file_open_delay_ms}ms")
    print(f"  frame_seek_delay: {args.frame_seek_delay_ms}ms")
    print(f"{'='*60}")

    dataset, loader = create_dataset_and_loader(args, delta_timestamps, mode=mode)

    batch_times = []
    sample_count = 0
    start_time = time.time()

    for batch_idx, items in enumerate(loader):
        t0 = time.time()
        # 在主进程中解码视频并 collate
        batch = decode_and_collate(items, dataset)
        _ = batch  # 消费 batch
        t1 = time.time()

        bs = None
        for key in batch:
            val = batch[key]
            if isinstance(val, torch.Tensor):
                bs = val.shape[0]
                break
            elif isinstance(val, list):
                bs = len(val)
                break
        sample_count += bs if bs is not None else 0
        batch_times.append(t1 - t0)

        if (batch_idx + 1) % max(1, args.max_batches // 10) == 0:
            elapsed = time.time() - start_time
            pct = (batch_idx + 1) / args.max_batches * 100
            speed = (batch_idx + 1) / max(elapsed, 1e-6)
            print(f"  [{pct:5.1f}%] batch {batch_idx + 1}/{args.max_batches}, "
                  f"{sample_count} samples, {speed:.1f} batch/s")

        if batch_idx + 1 >= args.max_batches:
            break

    total_time = time.time() - start_time
    avg_batch_time = total_time / max(len(batch_times), 1)
    throughput = sample_count / max(total_time, 1e-6)

    result = {
        "label": label,
        "mode": mode,
        "total_time": total_time,
        "sample_count": sample_count,
        "batch_count": len(batch_times),
        "avg_batch_time": avg_batch_time,
        "throughput": throughput,
        "p50_batch_time": np.median(batch_times),
        "p95_batch_time": np.percentile(batch_times, 95),
    }

    print(f"\n  结果:")
    print(f"    总时间: {result['total_time']:.1f}s")
    print(f"    样本数: {result['sample_count']}")
    print(f"    吞吐量: {result['throughput']:.1f} samples/s")
    print(f"    平均 batch 时间: {result['avg_batch_time']*1000:.1f}ms")
    print(f"    P50 batch 时间: {result['p50_batch_time']*1000:.1f}ms")
    print(f"    P95 batch 时间: {result['p95_batch_time']*1000:.1f}ms")

    # 清理
    del loader, dataset
    gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark: 延迟视频解码 vs 当前方案")
    parser.add_argument("--dataset_repo_id", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument("--history_padding_side", type=str, default="left")
    parser.add_argument("--file_open_delay_ms", type=int, default=0,
                        help="模拟新视频文件打开的网络延迟 (ms)")
    parser.add_argument("--frame_seek_delay_ms", type=int, default=0,
                        help="模拟每帧解码的网络延迟 (ms)")
    args = parser.parse_args()

    if args.dataset_repo_id is None and args.dataset_root is None:
        print("错误: 必须提供 --dataset_repo_id 或 --dataset_root")
        sys.exit(1)

    # 构建配置
    dataset_metadata = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
    )
    features = dataset_to_policy_features(dataset_metadata.features)
    action_dim = features["action"].shape[0]

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

    fps = dataset_metadata.fps
    delta_timestamps = {
        "observation.state": [i / fps for i in config.observation_delta_indices],
        "action": [i / fps for i in config.action_delta_indices],
    }
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    # ── 运行 benchmark ──────────────────────────────────────────────────

    results = []

    # Test 1: 当前方案（无延迟）
    r = run_benchmark(args, "current", delta_timestamps,
                      label="当前方案 (无延迟)")
    results.append(r)

    # Test 2: 延迟解码方案（无延迟）
    r = run_benchmark(args, "deferred", delta_timestamps,
                      label="延迟解码 (无延迟)")
    results.append(r)

    # Test 3: 当前方案 + 模拟网络延迟
    if args.file_open_delay_ms > 0 or args.frame_seek_delay_ms > 0:
        r = run_benchmark(args, "current", delta_timestamps,
                          label=f"当前方案 (延迟: open={args.file_open_delay_ms}ms, seek={args.frame_seek_delay_ms}ms)")
        results.append(r)

        r = run_benchmark(args, "deferred", delta_timestamps,
                          label=f"延迟解码 (延迟: open={args.file_open_delay_ms}ms, seek={args.frame_seek_delay_ms}ms)")
        results.append(r)

    # ── 汇总对比 ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Benchmark 汇总对比")
    print(f"{'='*70}")
    print(f"{'配置':<45} {'吞吐量':>12} {'P50(ms)':>10} {'P95(ms)':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<45} {r['throughput']:>10.1f}/s {r['p50_batch_time']*1000:>10.1f} {r['p95_batch_time']*1000:>10.1f}")

    if len(results) >= 2:
        base = results[0]
        deferred = results[1]
        ratio = deferred["throughput"] / max(base["throughput"], 1e-6)
        print(f"\n延迟解码 vs 当前方案吞吐量比: {ratio:.2f}x")
        if ratio < 0.9:
            print(f"  延迟解码吞吐量下降 {(1-ratio)*100:.1f}%")
        elif ratio > 1.1:
            print(f"  延迟解码吞吐量提升 {(ratio-1)*100:.1f}%")
        else:
            print(f"  两种方案吞吐量基本持平")


if __name__ == "__main__":
    main()
