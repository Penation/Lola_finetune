#!/usr/bin/env python
"""
Benchmark: 主进程视频解码加速方案对比。

测试矩阵：
  - 串行 CPU 解码 (decode_device="cpu", decode_num_threads=1)
  - Worker 进程解码 (worker_decode=True, 8 worker = 8× 真多进程并行)
  - 异步解码管线 (async_decode=True, 持久化大缓存)
  - CUDA 解码 (decode_device="cuda", decode_num_threads=1)

使用方法：
    python src/lerobot/scripts/bench_decode_modes.py \
        --dataset_root /data_6t_2/lerobot_v30/simpler_bridge_v3/ \
        --num_workers 10 \
        --batch_size 16 \
        --max_batches 200
"""

import argparse
import gc
import os
import sys
import time

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
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig


VARIABLE_LENGTH_KEYS = {"hist_actions_full", "hist_actions_mask"}


def make_passthrough_collate():
    def collate_fn(batch):
        return batch
    return collate_fn


def collate_decoded(items):
    """将解码后的 items 列表 collate 为 batch dict。"""
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


def build_config_and_timestamps(args):
    """构建 LoLAConfig 和 delta_timestamps。"""
    dataset_metadata = LeRobotDatasetMetadata(None, root=args.dataset_root)
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

    return delta_timestamps


def run_benchmark(args, decode_device, decode_num_threads, label=""):
    """运行同步解码的 benchmark 配置。"""
    gc.collect()

    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"  decode_device: {decode_device}")
    print(f"  decode_num_threads: {decode_num_threads}")
    print(f"  num_workers: {args.num_workers}, batch_size: {args.batch_size}")
    print(f"  max_batches: {args.max_batches}")
    print(f"{'='*60}")

    delta_timestamps = build_config_and_timestamps(args)

    dataset = LoLAStreamingDataset(
        repo_id=None,
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        history_padding_side=args.history_padding_side,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        streaming=True,
        buffer_size=args.buffer_size,
        seed=42,
        shuffle=True,
        deferred_video_decode=True,
        decode_device=decode_device,
        decode_num_threads=decode_num_threads,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=make_passthrough_collate(),
        pin_memory=False,
    )

    # 预热
    warmup_count = min(5, args.max_batches)
    items_iter = iter(loader)
    for _ in range(warmup_count):
        items = next(items_iter)
        decoded = dataset.decode_items_batch(items)
        _ = collate_decoded(decoded)

    del items_iter, loader
    gc.collect()

    # 正式测试
    dataset2 = LoLAStreamingDataset(
        repo_id=None,
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        history_padding_side=args.history_padding_side,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        streaming=True,
        buffer_size=args.buffer_size,
        seed=42,
        shuffle=True,
        deferred_video_decode=True,
        decode_device=decode_device,
        decode_num_threads=decode_num_threads,
    )

    loader2 = DataLoader(
        dataset2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=make_passthrough_collate(),
        pin_memory=False,
    )

    fetch_times = []
    decode_times = []
    collate_times = []
    sample_count = 0
    start_time = time.time()
    prev_end = start_time

    for batch_idx, items in enumerate(loader2):
        t_get = time.time()
        fetch_time = t_get - prev_end

        decoded = dataset2.decode_items_batch(items)

        t_decode = time.time()

        batch = collate_decoded(decoded)
        t_collate = time.time()

        fetch_times.append(fetch_time)
        decode_times.append(t_decode - t_get)
        collate_times.append(t_collate - t_decode)

        bs = len(items)
        sample_count += bs

        del batch, decoded, items
        prev_end = time.time()

        if (batch_idx + 1) % max(1, args.max_batches // 5) == 0:
            elapsed = time.time() - start_time
            speed = (batch_idx + 1) / max(elapsed, 1e-6)
            print(f"  [{(batch_idx+1)/args.max_batches*100:5.1f}%] batch {batch_idx+1}/{args.max_batches}, "
                  f"{sample_count} samples, {speed:.1f} batch/s, "
                  f"fetch={np.median(fetch_times[-20:])*1000:.1f}ms, "
                  f"decode={np.median(decode_times[-20:])*1000:.1f}ms", flush=True)

        if batch_idx + 1 >= args.max_batches:
            break

    total_time = time.time() - start_time
    throughput = sample_count / max(total_time, 1e-6)

    result = {
        "label": label,
        "mode": "sync",
        "decode_device": decode_device,
        "decode_num_threads": decode_num_threads,
        "total_time": total_time,
        "sample_count": sample_count,
        "throughput": throughput,
        "avg_fetch_time": np.mean(fetch_times),
        "p50_fetch_time": np.median(fetch_times),
        "avg_decode_time": np.mean(decode_times),
        "p50_decode_time": np.median(decode_times),
        "p95_decode_time": np.percentile(decode_times, 95),
        "avg_collate_time": np.mean(collate_times),
        "avg_batch_time": total_time / max(len(decode_times), 1),
    }

    print(f"\n  结果:")
    print(f"    总时间: {result['total_time']:.1f}s")
    print(f"    吞吐量: {result['throughput']:.1f} samples/s")
    print(f"    P50 fetch 时间: {result['p50_fetch_time']*1000:.1f}ms")
    print(f"    P50 decode 时间: {result['p50_decode_time']*1000:.1f}ms")
    print(f"    P95 decode 时间: {result['p95_decode_time']*1000:.1f}ms")
    print(f"    平均 collate 时间: {result['avg_collate_time']*1000:.1f}ms")
    print(f"    平均 batch 时间: {result['avg_batch_time']*1000:.1f}ms")

    # 清理
    del loader2, dataset2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_worker_decode_benchmark(args, label=""):
    """运行 worker 进程解码的 benchmark 配置。

    与主进程解码不同：每个 worker 进程独立解码视频帧，
    8 worker = 8× 真多进程并行（无 GIL 瓶颈）。
    主进程收到的已是解码后的帧，无需再调用 decode_items_batch()。
    """
    gc.collect()

    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"  mode: worker_decode (多进程并行)")
    print(f"  num_workers: {args.num_workers}, batch_size: {args.batch_size}")
    print(f"  max_batches: {args.max_batches}")
    print(f"{'='*60}")

    delta_timestamps = build_config_and_timestamps(args)

    dataset = LoLAStreamingDataset(
        repo_id=None,
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        history_padding_side=args.history_padding_side,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        streaming=True,
        buffer_size=args.buffer_size,
        seed=42,
        shuffle=True,
        deferred_video_decode=True,
        worker_decode=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=make_passthrough_collate(),
        pin_memory=False,
    )

    # 预热
    warmup_count = min(5, args.max_batches)
    items_iter = iter(loader)
    for _ in range(warmup_count):
        items = next(items_iter)
        # worker_decode 模式下，items 已解码
        _ = collate_decoded(items)

    del items_iter, loader
    gc.collect()

    # 正式测试
    dataset2 = LoLAStreamingDataset(
        repo_id=None,
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        history_padding_side=args.history_padding_side,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        streaming=True,
        buffer_size=args.buffer_size,
        seed=42,
        shuffle=True,
        deferred_video_decode=True,
        worker_decode=True,
    )

    loader2 = DataLoader(
        dataset2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=make_passthrough_collate(),
        pin_memory=False,
    )

    fetch_times = []
    collate_times = []
    sample_count = 0
    start_time = time.time()
    prev_end = start_time

    for batch_idx, items in enumerate(loader2):
        t_get = time.time()
        fetch_time = t_get - prev_end

        # items 已在 worker 中解码，直接 collate
        batch = collate_decoded(items)
        t_collate = time.time()

        fetch_times.append(fetch_time)
        collate_times.append(t_collate - t_get)

        bs = len(items)
        sample_count += bs

        del batch, items
        prev_end = time.time()

        if (batch_idx + 1) % max(1, args.max_batches // 5) == 0:
            elapsed = time.time() - start_time
            speed = (batch_idx + 1) / max(elapsed, 1e-6)
            print(f"  [{(batch_idx+1)/args.max_batches*100:5.1f}%] batch {batch_idx+1}/{args.max_batches}, "
                  f"{sample_count} samples, {speed:.1f} batch/s, "
                  f"fetch+decode={np.median(fetch_times[-20:])*1000:.1f}ms", flush=True)

        if batch_idx + 1 >= args.max_batches:
            break

    total_time = time.time() - start_time
    throughput = sample_count / max(total_time, 1e-6)

    result = {
        "label": label,
        "mode": "worker_decode",
        "decode_device": "cpu",
        "decode_num_threads": 0,
        "total_time": total_time,
        "sample_count": sample_count,
        "throughput": throughput,
        "avg_fetch_time": np.mean(fetch_times),
        "p50_fetch_time": np.median(fetch_times),
        "avg_decode_time": 0,
        "p50_decode_time": 0,
        "p95_decode_time": 0,
        "avg_collate_time": np.mean(collate_times),
        "avg_batch_time": total_time / max(len(fetch_times), 1),
    }

    print(f"\n  结果:")
    print(f"    总时间: {result['total_time']:.1f}s")
    print(f"    吞吐量: {result['throughput']:.1f} samples/s")
    print(f"    P50 fetch+decode 时间: {result['p50_fetch_time']*1000:.1f}ms")
    print(f"    平均 collate 时间: {result['avg_collate_time']*1000:.1f}ms")
    print(f"    平均 batch 时间: {result['avg_batch_time']*1000:.1f}ms")

    # 清理
    del loader2, dataset2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_async_benchmark(args, decode_device, label=""):
    """运行异步解码管线的 benchmark。"""
    gc.collect()

    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"  mode: async decode pipeline")
    print(f"  decode_device: {decode_device}")
    print(f"  num_workers: {args.num_workers}, batch_size: {args.batch_size}")
    print(f"  max_batches: {args.max_batches}")
    print(f"  cache_size: 2 * {args.num_workers} * n_cameras")
    print(f"{'='*60}")

    delta_timestamps = build_config_and_timestamps(args)

    dataset = LoLAStreamingDataset(
        repo_id=None,
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        history_padding_side=args.history_padding_side,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        streaming=True,
        buffer_size=args.buffer_size,
        seed=42,
        shuffle=True,
        deferred_video_decode=True,
        decode_device=decode_device,
        async_decode=True,
        num_dataloader_workers=args.num_workers,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=make_passthrough_collate(),
        pin_memory=False,
    )

    # 预热：使用 async decode_iter
    warmup_count = min(5, args.max_batches)
    decoded_iter = dataset.decode_iter(loader)
    for _ in range(warmup_count):
        decoded = next(decoded_iter)
        _ = collate_decoded(decoded)

    # 清理预热
    dataset.shutdown_decode_pipeline()
    del decoded_iter, loader
    gc.collect()

    # 正式测试
    dataset2 = LoLAStreamingDataset(
        repo_id=None,
        max_history_length=args.max_history_length,
        action_chunk_size=args.action_chunk_size,
        history_padding_side=args.history_padding_side,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        streaming=True,
        buffer_size=args.buffer_size,
        seed=42,
        shuffle=True,
        deferred_video_decode=True,
        decode_device=decode_device,
        async_decode=True,
        num_dataloader_workers=args.num_workers,
    )

    loader2 = DataLoader(
        dataset2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=make_passthrough_collate(),
        pin_memory=False,
    )

    batch_times = []
    sample_count = 0
    start_time = time.time()

    for batch_idx, decoded_items in enumerate(dataset2.decode_iter(loader2)):
        # decoded_items 已经在后台线程中解码完成
        batch = collate_decoded(decoded_items)

        bs = len(decoded_items)
        sample_count += bs
        batch_times.append(time.time() - start_time if batch_idx == 0 else time.time() - prev_end)
        prev_end = time.time()

        del batch, decoded_items

        if (batch_idx + 1) % max(1, args.max_batches // 5) == 0:
            elapsed = time.time() - start_time
            speed = (batch_idx + 1) / max(elapsed, 1e-6)
            print(f"  [{(batch_idx+1)/args.max_batches*100:5.1f}%] batch {batch_idx+1}/{args.max_batches}, "
                  f"{sample_count} samples, {speed:.1f} batch/s", flush=True)

        if batch_idx + 1 >= args.max_batches:
            break

    dataset2.shutdown_decode_pipeline()
    total_time = time.time() - start_time
    throughput = sample_count / max(total_time, 1e-6)

    result = {
        "label": label,
        "mode": "async",
        "decode_device": decode_device,
        "decode_num_threads": 0,
        "total_time": total_time,
        "sample_count": sample_count,
        "throughput": throughput,
        "avg_fetch_time": 0,
        "p50_fetch_time": 0,
        "avg_decode_time": 0,
        "p50_decode_time": 0,
        "p95_decode_time": 0,
        "avg_collate_time": 0,
        "avg_batch_time": total_time / max(len(batch_times), 1),
    }

    print(f"\n  结果:")
    print(f"    总时间: {result['total_time']:.1f}s")
    print(f"    吞吐量: {result['throughput']:.1f} samples/s")
    print(f"    平均 batch 时间: {result['avg_batch_time']*1000:.1f}ms")

    # 清理
    del loader2, dataset2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark: 主进程视频解码加速方案对比")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument("--history_padding_side", type=str, default="left")
    parser.add_argument("--skip_cuda", action="store_true",
                        help="跳过 CUDA 解码测试")
    parser.add_argument("--skip_async", action="store_true",
                        help="跳过异步解码管线测试")
    parser.add_argument("--skip_sync_parallel", action="store_true",
                        help="跳过同步多线程解码测试（只跑 baseline + async）")
    parser.add_argument("--skip_worker_decode", action="store_true",
                        help="跳过 worker 进程解码测试")
    args = parser.parse_args()

    results = []

    # Test 1: 串行 CPU 解码 (baseline)
    r = run_benchmark(args, "cpu", 1, label="串行 CPU (baseline)")
    results.append(r)

    if not args.skip_sync_parallel:
        # Test 2: 4线程 CPU 解码
        r = run_benchmark(args, "cpu", 4, label="4线程 CPU")
        results.append(r)

        # Test 3: 8线程 CPU 解码
        r = run_benchmark(args, "cpu", 8, label="8线程 CPU")
        results.append(r)

    # Test 4: Worker 进程解码（推荐：8 worker = 8× 真多进程并行）
    if not args.skip_worker_decode:
        r = run_worker_decode_benchmark(args, label="Worker 进程解码 (8× 并行)")
        results.append(r)

    # Test 5: 异步解码管线 (CPU)
    if not args.skip_async:
        r = run_async_benchmark(args, "cpu", label="异步管线 CPU (大缓存)")
        results.append(r)

    # Test 6: 异步解码管线 (CUDA)
    if not args.skip_cuda and not args.skip_async:
        try:
            r = run_async_benchmark(args, "cuda", label="异步管线 CUDA (大缓存)")
            results.append(r)
        except RuntimeError as e:
            if "Unsupported device" in str(e):
                print(f"\n[跳过] CUDA 解码不支持: {e}")
            else:
                raise

    # Test 7: CUDA 解码（同步）
    if not args.skip_cuda:
        try:
            r = run_benchmark(args, "cuda", 1, label="CUDA (NVDEC)")
            results.append(r)
        except RuntimeError as e:
            if "Unsupported device" in str(e):
                print(f"\n[跳过] CUDA 解码不支持: {e}")
                print("  需要 torchcodec >= 0.8 版本")
            else:
                raise

    # ── 汇总对比 ────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("Benchmark 汇总对比")
    print(f"{'='*100}")
    print(f"{'配置':<30} {'吞吐量':>10} {'fetch P50':>10} {'decode P50':>10} {'decode P95':>10} {'batch 时间':>10}")
    print("-" * 100)
    for r in results:
        if r['mode'] == 'worker_decode':
            fetch_str = f"{r['p50_fetch_time']*1000:>8.1f}ms"
            decode_str = " (worker)"
            decode95_str = " (worker)"
        elif r['mode'] == 'async':
            fetch_str = "  (async)"
            decode_str = "  (async)"
            decode95_str = "  (async)"
        else:
            fetch_str = f"{r['p50_fetch_time']*1000:>8.1f}ms"
            decode_str = f"{r['p50_decode_time']*1000:>8.1f}ms"
            decode95_str = f"{r['p95_decode_time']*1000:>8.1f}ms"
        print(f"{r['label']:<30} {r['throughput']:>8.1f}/s "
              f"{fetch_str} "
              f"{decode_str} "
              f"{decode95_str} "
              f"{r['avg_batch_time']*1000:>8.1f}ms")

    if len(results) >= 2:
        base = results[0]
        print(f"\n相对加速比（vs 串行 CPU baseline）:")
        for r in results[1:]:
            ratio = r["throughput"] / max(base["throughput"], 1e-6)
            print(f"  {r['label']}: 吞吐量 {ratio:.2f}x")


if __name__ == "__main__":
    main()
