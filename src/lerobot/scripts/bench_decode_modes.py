#!/usr/bin/env python
"""
Benchmark: 视频解码方案速度对比。

测试矩阵：
  - Yield-time decode (deferred=True, async=False) — 推荐模式
  - 串行 CPU 解码 (deferred=True, async=False, 主进程 decode_items_batch)
  - 多线程 CPU 解码 (deferred=True, async=False, decode_num_threads=4/8)
  - 异步解码管线 (deferred=True, async=True, 独立子进程)
  - CUDA 解码 (decode_device="cuda")

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
from lerobot.datasets.lola_streaming_dataset import LoLAStreamingDataset, AsyncDecodeDataLoader
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig


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
        input_features={k: v for k, v in features.items() if k.type != FeatureType.ACTION},
        output_features={k: v for k, v in features.items() if k.type == FeatureType.ACTION},
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


def run_benchmark(args, label="", deferred_video_decode=True, async_decode=False,
                  decode_device="cpu", decode_num_threads=1):
    """运行 benchmark 配置。

    Args:
        deferred_video_decode: True = buffer 存轻量帧, False = worker 内解码
        async_decode: True = 独立子进程解码管线
        decode_device: "cpu" or "cuda"
        decode_num_threads: 主进程解码线程数
    """
    gc.collect()

    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"  deferred_video_decode: {deferred_video_decode}")
    print(f"  async_decode: {async_decode}")
    print(f"  decode_device: {decode_device}")
    print(f"  decode_num_threads: {decode_num_threads}")
    print(f"  num_workers: {args.num_workers}, batch_size: {args.batch_size}")
    print(f"  max_batches: {args.max_batches}")
    print(f"{'='*60}")

    delta_timestamps = build_config_and_timestamps(args)

    # 预热
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
        deferred_video_decode=deferred_video_decode,
        decode_device=decode_device,
        decode_num_threads=decode_num_threads,
        async_decode=async_decode,
        num_dataloader_workers=args.num_workers,
    )

    raw_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
        pin_memory=False,
    )
    warmup_loader = AsyncDecodeDataLoader(
        dataloader=raw_loader,
        dataset=dataset,
        collate_fn=AsyncDecodeDataLoader.make_collate_fn(),
    )

    warmup_count = min(5, args.max_batches)
    for i, batch in enumerate(warmup_loader):
        if i + 1 >= warmup_count:
            break
    del warmup_loader, raw_loader, dataset
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
        deferred_video_decode=deferred_video_decode,
        decode_device=decode_device,
        decode_num_threads=decode_num_threads,
        async_decode=async_decode,
        num_dataloader_workers=args.num_workers,
    )

    raw_loader2 = DataLoader(
        dataset2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=None,
        pin_memory=False,
    )
    async_loader2 = AsyncDecodeDataLoader(
        dataloader=raw_loader2,
        dataset=dataset2,
        collate_fn=AsyncDecodeDataLoader.make_collate_fn(),
    )

    batch_times = []
    sample_count = 0
    start_time = time.time()
    prev_end = start_time

    for batch_idx, batch in enumerate(async_loader2):
        t_now = time.time()
        batch_time = t_now - prev_end if batch_idx > 0 else t_now - start_time
        batch_times.append(batch_time)

        # 获取 batch size
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

        del batch
        prev_end = time.time()

        if (batch_idx + 1) % max(1, args.max_batches // 5) == 0:
            elapsed = time.time() - start_time
            speed = (batch_idx + 1) / max(elapsed, 1e-6)
            print(f"  [{(batch_idx+1)/args.max_batches*100:5.1f}%] batch {batch_idx+1}/{args.max_batches}, "
                  f"{sample_count} samples, {speed:.1f} batch/s, "
                  f"p50_batch={np.median(batch_times[-20:])*1000:.1f}ms", flush=True)

        if batch_idx + 1 >= args.max_batches:
            break

    total_time = time.time() - start_time
    throughput = sample_count / max(total_time, 1e-6)

    result = {
        "label": label,
        "mode": "async" if async_decode else ("yield" if deferred_video_decode else "worker"),
        "deferred_video_decode": deferred_video_decode,
        "async_decode": async_decode,
        "decode_device": decode_device,
        "decode_num_threads": decode_num_threads,
        "total_time": total_time,
        "sample_count": sample_count,
        "throughput": throughput,
        "p50_batch_time": np.median(batch_times) if batch_times else 0,
        "p95_batch_time": np.percentile(batch_times, 95) if batch_times else 0,
        "avg_batch_time": total_time / max(len(batch_times), 1),
    }

    print(f"\n  结果:")
    print(f"    总时间: {result['total_time']:.1f}s")
    print(f"    吞吐量: {result['throughput']:.1f} samples/s")
    print(f"    P50 batch 时间: {result['p50_batch_time']*1000:.1f}ms")
    print(f"    P95 batch 时间: {result['p95_batch_time']*1000:.1f}ms")
    print(f"    平均 batch 时间: {result['avg_batch_time']*1000:.1f}ms")

    # 清理
    del async_loader2, raw_loader2, dataset2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark: 视频解码方案速度对比")
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
    parser.add_argument("--skip_yield", action="store_true",
                        help="跳过 yield-time decode 测试")
    parser.add_argument("--skip_multithread", action="store_true",
                        help="跳过多线程 CPU 解码测试")
    args = parser.parse_args()

    results = []

    # Test 1: Yield-time decode (推荐模式，默认行为)
    # deferred=True + async=False: buffer 存轻量帧，yield 时解码
    if not args.skip_yield:
        r = run_benchmark(args, label="Yield-time decode (推荐)",
                          deferred_video_decode=True, async_decode=False,
                          decode_device="cpu", decode_num_threads=1)
        results.append(r)

    # Test 2: Worker decode (最快，但 flush 阶段内存峰值)
    # deferred=False: worker 内 make_frame 解码
    r = run_benchmark(args, label="Worker decode (最快)",
                      deferred_video_decode=False, async_decode=False,
                      decode_device="cpu", decode_num_threads=1)
    results.append(r)

    if not args.skip_multithread:
        # Test 3: 异步管线 + 4线程 CPU 解码
        if not args.skip_async:
            r = run_benchmark(args, label="异步管线 CPU 4线程",
                              deferred_video_decode=True, async_decode=True,
                              decode_device="cpu", decode_num_threads=4)
            results.append(r)

    # Test 4: 异步解码管线 - 独立子进程 (CPU)
    if not args.skip_async:
        r = run_benchmark(args, label="异步管线 CPU (子进程)",
                          deferred_video_decode=True, async_decode=True,
                          decode_device="cpu")
        results.append(r)

    # Test 5: 异步解码管线 - 独立子进程 (CUDA)
    if not args.skip_cuda and not args.skip_async:
        try:
            r = run_benchmark(args, label="异步管线 CUDA (大缓存)",
                              deferred_video_decode=True, async_decode=True,
                              decode_device="cuda")
            results.append(r)
        except RuntimeError as e:
            if "Unsupported device" in str(e):
                print(f"\n[跳过] CUDA 解码不支持: {e}")
            else:
                raise

    # Test 6: CUDA 解码（同步）
    if not args.skip_cuda:
        try:
            r = run_benchmark(args, label="CUDA (NVDEC)",
                              deferred_video_decode=True, async_decode=False,
                              decode_device="cuda", decode_num_threads=1)
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
    print(f"{'配置':<30} {'吞吐量':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'batch(ms)':>10}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:<30} {r['throughput']:>8.1f}/s "
              f"{r['p50_batch_time']*1000:>10.1f} "
              f"{r['p95_batch_time']*1000:>10.1f} "
              f"{r['avg_batch_time']*1000:>10.1f}")

    if len(results) >= 2:
        base = results[0]
        print(f"\n相对加速比（vs {base['label']}）:")
        for r in results[1:]:
            ratio = r["throughput"] / max(base["throughput"], 1e-6)
            print(f"  {r['label']}: 吞吐量 {ratio:.2f}x")


if __name__ == "__main__":
    main()
