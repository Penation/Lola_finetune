#!/usr/bin/env python
"""
校验 LoLAStreamingDataset + DataLoader 是否能正确遍历数据的脚本。

验证项：
1. 数据集能否成功创建并迭代
2. 每条样本是否包含预期的 key（observation, action, hist_actions 等）
3. 每条样本的 tensor shape 和 dtype 是否符合预期
4. collate_fn 能否正确将一个 batch 的数据合并
5. DataLoader 多 worker 是否能正常工作（先于单 worker 测试，避免 torchcodec fork 死锁）
6. DataLoader 单 worker 是否能正常工作
7. 多批次迭代的稳定性
7. 解码图像质量检查与保存（全黑/全白/值范围/padding mask）

图像保存：
    默认保存到 ./validate_images/ 目录，按阶段分子目录：
      step4_single/       - 多批次遍历的相机图像
      step5_single/       - 单条样本的相机图像

    每个目录下包含：
      {camera_key}_sample{i}.png  - 各 camera 的单帧图像
      all_cameras_sample{i}.png   - 所有 camera 水平拼接图

    使用 --save_images_dir "" 可禁用图像保存。

使用方法：
    # 最小参数（自动使用默认值）
    python src/lerobot/scripts/validate_stream_dataloader.py \
        --dataset_root /mnt/data/lerobot-dataset

    # 完整参数
    python src/lerobot/scripts/validate_stream_dataloader.py \
        --dataset_repo_id <repo_id> \
        --dataset_root /mnt/data/lerobot-dataset \
        --batch_size 4 \
        --num_workers 2 \
        --max_batches 20 \
        --max_history_length 100 \
        --action_chunk_size 10 \
        --save_images_dir ./validate_images \
        --num_images_per_stage 3
"""

import argparse
import os
import sys
import time
import traceback

import torch
from torch.utils.data import DataLoader

try:
    from torchvision.utils import save_image as tv_save_image
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

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
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lola_streaming_dataset import LoLAStreamingDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig


# ── 与 train_lola_azure_stream.py 保持一致的 collate_fn ──────────────────────

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


# ── 校验辅助函数 ──────────────────────────────────────────────────────────────

class ValidationResult:
    """收集校验结果"""
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
            "校验结果汇总",
            "=" * 60,
            f"通过: {len(self.passed)}/{total}",
            f"失败: {len(self.failed)}/{total}",
            f"警告: {len(self.warnings)}",
        ]
        if self.failed:
            lines.append("--- 失败项 ---")
            for f in self.failed:
                lines.append(f"  [FAIL] {f}")
        if self.warnings:
            lines.append("--- 警告项 ---")
            for w in self.warnings:
                lines.append(f"  [WARN] {w}")
        lines.append("")
        if not self.failed:
            lines.append("全部校验通过!")
        else:
            lines.append("存在校验失败项，请检查!")
        return "\n".join(lines)


def check_keys(item, expected_keys, result: ValidationResult, label=""):
    """检查 item 是否包含所有预期 key"""
    actual_keys = set(item.keys())
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    prefix = f"[{label}] " if label else ""

    if missing:
        result.fail(f"{prefix}缺少 key: {missing}")
    else:
        result.ok(f"{prefix}包含所有预期 key")

    if extra:
        result.warn(f"{prefix}额外 key (不一定是错误): {extra}")


def check_tensor_props(item, key, result: ValidationResult, expected_dtype=None, expected_ndim=None, label=""):
    """检查 tensor 的 dtype 和 ndim"""
    prefix = f"[{label}][{key}] " if label else f"[{key}] "
    val = item.get(key)
    if val is None:
        result.fail(f"{prefix}key 不存在")
        return

    if not isinstance(val, torch.Tensor):
        result.fail(f"{prefix}不是 torch.Tensor, 实际类型: {type(val).__name__}")
        return

    if expected_dtype is not None and val.dtype != expected_dtype:
        result.fail(f"{prefix}dtype 不匹配: 期望 {expected_dtype}, 实际 {val.dtype}")
    else:
        result.ok(f"{prefix}dtype={val.dtype}")

    if expected_ndim is not None and val.dim() != expected_ndim:
        result.fail(f"{prefix}ndim 不匹配: 期望 {expected_ndim}, 实际 {val.dim()}")
    else:
        result.ok(f"{prefix}ndim={val.dim()}")


def check_no_nan_inf(item, result: ValidationResult, label=""):
    """检查所有 tensor 是否包含 NaN 或 Inf"""
    prefix = f"[{label}] " if label else ""
    has_issue = False
    for key, val in item.items():
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            if torch.isnan(val).any():
                result.fail(f"{prefix}[{key}] 包含 NaN")
                has_issue = True
            if torch.isinf(val).any():
                result.fail(f"{prefix}[{key}] 包含 Inf")
                has_issue = True
    if not has_issue:
        result.ok(f"{prefix}所有 float tensor 无 NaN/Inf")


def save_camera_images(batch, camera_keys, save_dir, stage_name, max_samples=3):
    """从 batch 中保存相机图像到磁盘。

    对于每个 camera key，保存 max_samples 张图像。
    支持多种图像 tensor 形状:
      - [B, T, C, H, W]: 取每个 sample 的第 0 帧 (当前帧)
      - [B, C, H, W]: 直接保存
      - [C, H, W]: 单张 (未 batch)

    Args:
        batch: DataLoader 产出的 batch dict
        camera_keys: 需要保存的 camera key 列表
        save_dir: 保存根目录
        stage_name: 阶段名 (用于子目录)
        max_samples: 每个 camera 最多保存多少个 sample 的图像
    """
    if not HAS_TORCHVISION and not HAS_PIL:
        print("    [跳过图像保存] 需要 torchvision 或 PIL")
        return

    stage_dir = os.path.join(save_dir, stage_name)
    os.makedirs(stage_dir, exist_ok=True)

    saved_count = 0

    for cam_key in camera_keys:
        if cam_key not in batch:
            print(f"    [跳过] camera key '{cam_key}' 不在 batch 中")
            continue

        img_tensor = batch[cam_key]
        if not isinstance(img_tensor, torch.Tensor):
            print(f"    [跳过] camera key '{cam_key}' 不是 Tensor (type={type(img_tensor).__name__})")
            continue

        # 统一为 [N, C, H, W] 用于逐张保存
        if img_tensor.dim() == 5:
            # [B, T, C, H, W] -> 取每个 sample 的第 0 帧
            img_tensor = img_tensor[:, 0, :, :, :]
        elif img_tensor.dim() == 3:
            # [C, H, W] -> 增加 batch 维度
            img_tensor = img_tensor.unsqueeze(0)

        if img_tensor.dim() != 4:
            print(f"    [跳过] camera key '{cam_key}' shape={batch[cam_key].shape} 无法处理为 [N,C,H,W]")
            continue

        n_samples = min(img_tensor.shape[0], max_samples)
        cam_name = cam_key.replace("/", "_").replace(".", "_")

        for i in range(n_samples):
            frame = img_tensor[i]  # [C, H, W]

            # 检查值范围，确保在 [0,1]
            if frame.min() < 0 or frame.max() > 1:
                # 如果值在 [0, 255] 整数范围，自动归一化
                if frame.max() > 1:
                    frame = frame / 255.0
                frame = frame.clamp(0, 1)

            fname = os.path.join(stage_dir, f"{cam_name}_sample{i}.png")

            if HAS_TORCHVISION:
                tv_save_image(frame, fname)
            elif HAS_PIL:
                # CHW -> HWC, 转 numpy
                arr = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                Image.fromarray(arr).save(fname)

            saved_count += 1

    # 保存一个所有 camera 的拼接网格图 (仅 torchvision 可用时)
    if HAS_TORCHVISION and len(camera_keys) > 1:
        grid_tensors = []
        valid_cam_keys = [k for k in camera_keys if k in batch and isinstance(batch[k], torch.Tensor)]
        if valid_cam_keys:
            for cam_key in valid_cam_keys:
                t = batch[cam_key]
                if t.dim() == 5:
                    t = t[:, 0, :, :, :]
                elif t.dim() == 3:
                    t = t.unsqueeze(0)
                if t.dim() == 4:
                    grid_tensors.append(t[:max_samples])

            if grid_tensors:
                # 确保 all tensors have same sample count
                min_n = min(t.shape[0] for t in grid_tensors)
                grid_tensors = [t[:min_n] for t in grid_tensors]
                # concat along channel? 不，沿 batch dim 拼接后做 grid
                # 每个 sample 显示所有 camera: 先做 [n_cams, C, H, W] 的 grid
                for i in range(min_n):
                    frames = [t[i] for t in grid_tensors]  # list of [C, H, W]
                    row = torch.cat(frames, dim=2)  # [C, H, W*len(cameras)] 水平拼接
                    grid_fname = os.path.join(stage_dir, f"all_cameras_sample{i}.png")
                    tv_save_image(row, grid_fname)
                    saved_count += 1

    if saved_count > 0:
        print(f"    已保存 {saved_count} 张图像到 {stage_dir}/")


def check_image_quality(batch, camera_keys, result: ValidationResult, label=""):
    """检查解码图像的质量：全黑、全白、值范围、通道数、is_pad 诊断。"""
    prefix = f"[{label}]" if label else ""

    for cam_key in camera_keys:
        if cam_key not in batch:
            continue
        img_tensor = batch[cam_key]
        if not isinstance(img_tensor, torch.Tensor):
            continue

        cam_prefix = f"{prefix}[{cam_key}]"

        # 值范围检查
        if img_tensor.is_floating_point():
            if img_tensor.min() < -0.5 or img_tensor.max() > 1.5:
                result.warn(f"{cam_prefix} 值范围异常: [{img_tensor.min():.3f}, {img_tensor.max():.3f}], 期望 [0,1]")
            else:
                result.ok(f"{cam_prefix} 值范围正常: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

        # 全黑检查
        if img_tensor.is_floating_point() and img_tensor.max() < 1e-6:
            result.fail(f"{cam_prefix} 图像全黑 (max={img_tensor.max():.6f})")
        elif img_tensor.is_floating_point() and img_tensor.mean() < 0.01:
            result.warn(f"{cam_prefix} 图像几乎全黑 (mean={img_tensor.mean():.4f})")
        else:
            result.ok(f"{cam_prefix} 图像非全黑 (mean={img_tensor.mean():.4f})")

        # 全白检查
        if img_tensor.is_floating_point() and img_tensor.min() > 0.99:
            result.fail(f"{cam_prefix} 图像全白 (min={img_tensor.min():.6f})")

        # 通道数检查 (期望 3 通道 RGB)
        if img_tensor.dim() >= 3:
            c_dim = 1 if img_tensor.dim() == 4 else (1 if img_tensor.dim() == 5 else 0)
            if c_dim > 0 and img_tensor.shape[c_dim] not in (1, 3):
                result.warn(f"{cam_prefix} 通道数={img_tensor.shape[c_dim]}, 期望 1 或 3")
            elif c_dim > 0:
                result.ok(f"{cam_prefix} 通道数={img_tensor.shape[c_dim]}")

        # is_pad 诊断：打印详细分布，帮助定位是 clamp 导致的误判还是真的缺帧
        pad_key = f"{cam_key}_is_pad"
        if pad_key in batch and isinstance(batch[pad_key], torch.Tensor):
            pad_mask = batch[pad_key]
            # _is_pad 中 True=padding, False=有效
            valid_ratio = (~pad_mask).float().mean().item()
            n_pad = pad_mask.sum().item()
            n_total = pad_mask.numel()

            # 逐 sample 打印 is_pad 分布 (仅 batch 维度 < 8 时)
            if pad_mask.dim() >= 1 and pad_mask.shape[0] <= 8:
                per_sample_info = []
                for i in range(pad_mask.shape[0]):
                    if pad_mask.dim() == 1:
                        per_sample_info.append(f"s{i}={'T' if pad_mask[i].item() else 'F'}")
                    else:
                        # [B, T] 展示每行
                        row = pad_mask[i]
                        per_sample_info.append(f"s{i}={row.tolist()}")
                print(f"    {cam_prefix} is_pad 详细: {'; '.join(per_sample_info)}")

            if valid_ratio < 0.5:
                result.warn(
                    f"{cam_prefix} is_pad 有效帧比例仅 {valid_ratio:.2%} "
                    f"({n_total - n_pad}/{n_total} 有效). "
                    f"可能原因: episode 边界附近 clamp 导致 timestamp 重复映射, "
                    f"或 delta_timestamps 中 lookahead 超出 episode 范围"
                )
            else:
                result.ok(f"{cam_prefix} is_pad 有效帧比例 {valid_ratio:.2%} ({n_total - n_pad}/{n_total} 有效)")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="校验 LoLAStreamingDataset + DataLoader")
    parser.add_argument("--dataset_repo_id", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=20,
                        help="最多遍历多少个 batch 后停止")
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument("--streaming_seed", type=int, default=42)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--history_padding_side", type=str, default="left",
                        choices=["left", "right"])
    parser.add_argument("--save_images_dir", type=str,
                        default="./validate_images",
                        help="保存解码图像的目录 (设为空字符串则不保存)")
    parser.add_argument("--num_images_per_stage", type=int, default=3,
                        help="每个阶段每个 camera 最多保存多少张图像")
    parser.add_argument("--decode_device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="视频解码设备 (cpu 或 cuda)")
    parser.add_argument("--async_decode", action="store_true",
                        help="启用异步解码管线 (后台线程解码 + 持久化大缓存)")
    parser.add_argument("--worker_decode", action="store_true",
                        help="启用 worker 进程解码 (每个 worker 独立解码视频帧，"
                             "8 worker = 8× 真多进程并行，主进程收到的已是解码后的帧)")
    args = parser.parse_args()

    if args.dataset_repo_id is None and args.dataset_root is None:
        print("错误: 必须提供 --dataset_repo_id 或 --dataset_root")
        sys.exit(1)

    result = ValidationResult()

    # ── Step 1: 加载元数据 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: 加载数据集元数据")
    print("=" * 60)

    try:
        dataset_metadata = LeRobotDatasetMetadata(
            args.dataset_repo_id,
            root=args.dataset_root,
        )
        print(f"  数据集: {dataset_metadata.total_episodes} episodes, "
              f"{dataset_metadata.total_frames} frames")
        print(f"  camera_keys: {dataset_metadata.camera_keys}")
        print(f"  fps: {dataset_metadata.fps}")
        result.ok("数据集元数据加载成功")
    except Exception as e:
        result.fail(f"数据集元数据加载失败: {e}")
        print(result.summary())
        sys.exit(1)

    # ── Step 2: 构建 LoLAConfig ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: 构建 LoLAConfig")
    print("=" * 60)

    features = dataset_to_policy_features(dataset_metadata.features)
    if "action" in features:
        action_dim = features["action"].shape[0]
    else:
        action_dim = args.action_dim
        result.warn(f"features 中无 action key, 使用默认 action_dim={action_dim}")

    try:
        config = LoLAConfig(
            vlm_model_name="Qwen/Qwen3.5-4B",
            vlm_path="/tmp/dummy",  # 校验时不需要真实 VLM 权重
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
        result.ok("LoLAConfig 创建成功")
    except Exception as e:
        result.fail(f"LoLAConfig 创建失败: {e}")
        print(result.summary())
        sys.exit(1)

    # ── Step 3: 创建 LoLAStreamingDataset ────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: 创建 LoLAStreamingDataset")
    print("=" * 60)

    fps = dataset_metadata.fps
    delta_timestamps = {
        "observation.state": [i / fps for i in config.observation_delta_indices],
        "action": [i / fps for i in config.action_delta_indices],
    }
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    try:
        dataset = LoLAStreamingDataset(
            repo_id=args.dataset_repo_id,
            max_history_length=args.max_history_length,
            action_chunk_size=args.action_chunk_size,
            history_padding_side=args.history_padding_side,
            root=args.dataset_root,
            episodes=args.episodes,
            image_transforms=None,
            delta_timestamps=delta_timestamps,
            streaming=True,
            buffer_size=args.buffer_size,
            seed=args.streaming_seed,
            shuffle=False if args.no_shuffle else True,
            decode_device=args.decode_device,
            async_decode=args.async_decode,
            num_dataloader_workers=args.num_workers,
            worker_decode=args.worker_decode,
        )
        result.ok("LoLAStreamingDataset 创建成功")
    except Exception as e:
        result.fail(f"LoLAStreamingDataset 创建失败: {e}")
        traceback.print_exc()
        print(result.summary())
        sys.exit(1)

    # ── Step 4: 多批次遍历校验（先于单 worker 步骤，避免 torchcodec fork 死锁）
    # 重要：torchcodec 在主进程初始化后，fork 的子进程调用 torchcodec 会死锁。
    # 因此必须先跑 num_workers > 0 的步骤，再跑 num_workers=0 的步骤。
    print("\n" + "=" * 60)
    print(f"Step 4: 多批次遍历校验 (num_workers={args.num_workers}, "
          f"batch_size={args.batch_size}, max_batches={args.max_batches})")
    print("=" * 60)

    try:
        dataset = LoLAStreamingDataset(
            repo_id=args.dataset_repo_id,
            max_history_length=args.max_history_length,
            action_chunk_size=args.action_chunk_size,
            history_padding_side=args.history_padding_side,
            root=args.dataset_root,
            episodes=args.episodes,
            image_transforms=None,
            delta_timestamps=delta_timestamps,
            streaming=True,
            buffer_size=args.buffer_size,
            seed=args.streaming_seed,
            shuffle=False if args.no_shuffle else True,
            decode_device=args.decode_device,
            async_decode=args.async_decode,
            num_dataloader_workers=args.num_workers,
            worker_decode=args.worker_decode,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=make_passthrough_collate(),
            pin_memory=False,
        )

        batch_count = 0
        sample_count = 0
        errors_in_iteration = []
        start_time = time.time()

        # 选择迭代模式：异步管线 vs worker 解码 vs 同步解码
        if args.async_decode:
            iter_source = dataset.decode_iter(loader)
        else:
            iter_source = loader

        for batch_idx, items_or_decoded in enumerate(iter_source):
            if args.worker_decode:
                # worker_decode 模式：帧已在 worker 中解码，直接 collate
                batch = decode_and_collate(items_or_decoded, dataset)
            elif args.async_decode:
                # decode_iter 已经在后台线程中解码完成
                batch = decode_and_collate(items_or_decoded, dataset)
            else:
                # 同步模式：在主进程中解码视频并 collate
                batch = decode_and_collate(items_or_decoded, dataset)
            batch_count += 1
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

            # 每个 batch 做快速检查
            try:
                # 检查 batch 中所有 tensor 的 batch 维度一致
                batch_sizes = set()
                for key, val in batch.items():
                    if isinstance(val, torch.Tensor):
                        batch_sizes.add(val.shape[0])
                    elif isinstance(val, list):
                        batch_sizes.add(len(val))
                if len(batch_sizes) > 1:
                    result.fail(f"[batch {batch_idx}] batch 维度不一致: {batch_sizes}")
                elif len(batch_sizes) == 1:
                    actual_bs = batch_sizes.pop()
                    if actual_bs != args.batch_size and batch_idx == 0:
                        result.warn(f"[batch {batch_idx}] batch_size={actual_bs} < {args.batch_size}, "
                                    "数据量可能不足一个完整 batch")
            except Exception as e:
                errors_in_iteration.append(f"batch {batch_idx}: {e}")

            # 检查 hist_actions_mask 和 hist_actions_full shape 匹配
            if "hist_actions_full" in batch and "hist_actions_mask" in batch:
                haf = batch["hist_actions_full"]
                ham = batch["hist_actions_mask"]
                if haf.shape[0] != ham.shape[0]:
                    result.fail(f"[batch {batch_idx}] hist_actions_full 和 hist_actions_mask batch 维度不匹配: "
                                f"{haf.shape[0]} vs {ham.shape[0]}")
                if haf.shape[1] != ham.shape[1]:
                    result.fail(f"[batch {batch_idx}] hist_actions_full 和 hist_actions_mask 序列长度不匹配: "
                                f"{haf.shape[1]} vs {ham.shape[1]}")

            # 检查 hist_actions_length 不超过 max_history_length
            if "hist_actions_length" in batch:
                hal = batch["hist_actions_length"]
                max_hal = hal.max().item()
                if max_hal > args.max_history_length:
                    result.fail(f"[batch {batch_idx}] hist_actions_length 最大值 {max_hal} "
                                f"超过 max_history_length {args.max_history_length}")

            # 打印前几个 batch 的详细信息
            if batch_idx < 3:
                print(f"  batch {batch_idx}:")
                for key, val in batch.items():
                    if isinstance(val, torch.Tensor):
                        print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
                    elif isinstance(val, list) and len(val) <= 4:
                        print(f"    {key}: {val}")
                    else:
                        print(f"    {key}: type={type(val).__name__}, len={len(val)}")

            # 图像质量检查 (前 3 个 batch)
            if batch_idx < 3:
                check_image_quality(batch, dataset_metadata.camera_keys, result, label=f"step4_batch{batch_idx}")

            # 保存解码图像 (前 3 个 batch，每个 batch 最多 num_images_per_stage 张)
            if args.save_images_dir and batch_idx < 3:
                save_camera_images(
                    batch, dataset_metadata.camera_keys,
                    args.save_images_dir, f"step4_batch{batch_idx}",
                    max_samples=args.num_images_per_stage,
                )

            # 周期性进度打印
            if (batch_idx + 1) % max(1, args.max_batches // 20) == 0 or batch_idx == 0:
                elapsed = time.time() - start_time
                pct = (batch_idx + 1) / args.max_batches * 100
                speed = (batch_idx + 1) / max(elapsed, 1e-6)
                eta = (args.max_batches - batch_idx - 1) / max(speed, 1e-6)
                print(f"  [{pct:5.1f}%] batch {batch_idx + 1}/{args.max_batches}, "
                      f"{sample_count} samples, {speed:.1f} batch/s, ETA {eta:.0f}s")

            if batch_count >= args.max_batches:
                break

        elapsed = time.time() - start_time
        print(f"成功遍历 {batch_count} 个 batch ({sample_count} 条样本), 耗时 {elapsed:.1f}s, 平均 {elapsed/max(batch_count,1):.2f}s/batch")
        result.ok(f"成功遍历 {batch_count} 个 batch ({sample_count} 条样本), "
                  f"耗时 {elapsed:.1f}s, 平均 {elapsed/max(batch_count,1):.2f}s/batch")

        if errors_in_iteration:
            for e in errors_in_iteration:
                result.fail(f"迭代中出错: {e}")

        # 清理异步解码管线
        if args.async_decode:
            dataset.shutdown_decode_pipeline()

    except Exception as e:
        result.fail(f"多批次遍历失败: {e}")
        traceback.print_exc()

    # ── Step 5: 单条样本校验（num_workers=0，在多 worker 之后运行）
    # torchcodec 已在主进程的 DataLoader worker 中初始化，但 num_workers=0
    # 不会 fork，因此不会死锁。
    print("\n" + "=" * 60)
    print("Step 5: 单条样本校验 (num_workers=0)")
    print("=" * 60)

    try:
        dataset = LoLAStreamingDataset(
            repo_id=args.dataset_repo_id,
            max_history_length=args.max_history_length,
            action_chunk_size=args.action_chunk_size,
            history_padding_side=args.history_padding_side,
            root=args.dataset_root,
            episodes=args.episodes,
            image_transforms=None,
            delta_timestamps=delta_timestamps,
            streaming=True,
            buffer_size=args.buffer_size,
            seed=args.streaming_seed,
            shuffle=False if args.no_shuffle else True,
            decode_device=args.decode_device,
            async_decode=False,  # 单条样本不需要异步管线
            worker_decode=False,  # 单条样本不需要 worker 解码
        )

        single_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=make_passthrough_collate(),
        )
        items = next(iter(single_loader))
        sample_batch = decode_and_collate(items, dataset)
        result.ok("单条样本成功获取")

        # 确定预期 key
        expected_keys = {
            "action", "episode_index", "frame_index", "timestamp",
            "index", "task_index", "task",
            "observation.state",
            "hist_actions_full", "hist_actions_mask", "hist_actions_length",
        }
        # 添加 camera key
        for cam_key in dataset_metadata.camera_keys:
            expected_keys.add(cam_key)
            expected_keys.add(f"{cam_key}_is_pad")  # streaming 使用 _is_pad 后缀

        # 添加 action / state is_pad
        expected_keys.add("action_is_pad")
        expected_keys.add("observation.state_is_pad")

        check_keys(sample_batch, expected_keys, result, label="batch")

        # 检查关键 tensor 的 shape 和 dtype
        for key in sample_batch:
            val = sample_batch[key]
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

        # is_pad 全量诊断：打印每个 _is_pad key 的具体值
        print("  --- is_pad 诊断 ---")
        for key in sorted(sample_batch.keys()):
            if key.endswith("_is_pad"):
                val = sample_batch[key]
                if isinstance(val, torch.Tensor):
                    print(f"  {key}: shape={val.shape}, values={val.tolist()}")

        # hist_actions_full: [B, padded_len, action_dim]
        check_tensor_props(
            sample_batch, "hist_actions_full",
            expected_dtype=torch.float32, expected_ndim=3,
            result=result, label="batch",
        )
        if "hist_actions_full" in sample_batch and isinstance(sample_batch["hist_actions_full"], torch.Tensor):
            haf = sample_batch["hist_actions_full"]
            if haf.shape[-1] != action_dim:
                result.fail(f"[batch][hist_actions_full] 最后一维={haf.shape[-1]}, 期望 action_dim={action_dim}")
            else:
                result.ok(f"[batch][hist_actions_full] 最后一维={action_dim} (与 action_dim 一致)")
            if haf.shape[0] != 1:
                result.fail(f"[batch][hist_actions_full] batch 维度={haf.shape[0]}, 期望 1")
            else:
                result.ok("[batch][hist_actions_full] batch 维度=1")

        # hist_actions_mask: [B, padded_len]
        check_tensor_props(
            sample_batch, "hist_actions_mask",
            expected_dtype=torch.bool, expected_ndim=2,
            result=result, label="batch",
        )

        # hist_actions_length: [B]
        check_tensor_props(
            sample_batch, "hist_actions_length",
            expected_dtype=torch.int64, expected_ndim=1,
            result=result, label="batch",
        )

        # action: [B, chunk_size, action_dim]
        check_tensor_props(
            sample_batch, "action",
            expected_ndim=3,
            result=result, label="batch",
        )
        if "action" in sample_batch and isinstance(sample_batch["action"], torch.Tensor):
            act = sample_batch["action"]
            if act.shape[-1] != action_dim:
                result.fail(f"[batch][action] 最后一维={act.shape[-1]}, 期望 action_dim={action_dim}")
            else:
                result.ok(f"[batch][action] 最后一维={action_dim} (与 action_dim 一致)")

        # NaN/Inf 检查
        check_no_nan_inf(sample_batch, result, label="batch")

        # 图像质量检查
        check_image_quality(sample_batch, dataset_metadata.camera_keys, result, label="step5_single")

        # 保存解码图像
        if args.save_images_dir:
            print(f"  保存 Step 5 单条样本图像...")
            save_camera_images(
                sample_batch, dataset_metadata.camera_keys,
                args.save_images_dir, "step5_single",
                max_samples=args.num_images_per_stage,
            )

    except Exception as e:
        result.fail(f"单条样本获取失败: {e}")
        traceback.print_exc()

    # ── Step 6: collate_fn 一致性校验 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: collate_fn 一致性校验")
    print("=" * 60)

    try:
        # 手动构造模拟数据来验证 collate_fn 逻辑
        fake_item_1 = {
            "action": torch.randn(args.action_chunk_size, action_dim),
            "episode_index": torch.tensor(0),
            "hist_actions_full": torch.randn(40, action_dim),
            "hist_actions_mask": torch.ones(40, dtype=torch.bool),
            "hist_actions_length": torch.tensor(40, dtype=torch.long),
            "task": "pick_place",
        }
        fake_item_2 = {
            "action": torch.randn(args.action_chunk_size, action_dim),
            "episode_index": torch.tensor(0),
            "hist_actions_full": torch.randn(30, action_dim),  # 不同长度
            "hist_actions_mask": torch.ones(30, dtype=torch.bool),
            "hist_actions_length": torch.tensor(30, dtype=torch.long),
            "task": "pick_place",
        }
        collated = decode_and_collate([fake_item_1, fake_item_2], dataset)
        result.ok("collate_fn 对变长 hist_actions_full/hist_actions_mask 能正常工作")

        # 验证 padding 后长度一致
        if collated["hist_actions_full"].shape[1] == collated["hist_actions_mask"].shape[1]:
            result.ok(f"collate_fn padding 后序列长度一致: {collated['hist_actions_full'].shape[1]}")
        else:
            result.fail("collate_fn padding 后序列长度不一致")

        # 验证 batch 维度
        if collated["hist_actions_full"].shape[0] == 2:
            result.ok("collate_fn batch 维度=2 (符合输入 2 条样本)")
        else:
            result.fail(f"collate_fn batch 维度={collated['hist_actions_full'].shape[0]}, 期望 2")

        # 验证较短样本的 padding 区域为 0
        padded_len = collated["hist_actions_full"].shape[1]
        # fake_item_2 有 30 个真实值，应该被 left-padded 到 padded_len
        if args.history_padding_side == "left":
            # 左 padding: 前 (padded_len - 30) 个应该是 0
            padding_region = collated["hist_actions_full"][1, :padded_len - 30]
            if (padding_region == 0).all():
                result.ok("left padding 区域全为 0")
            else:
                result.warn("left padding 区域并非全为 0 (可能非零值来自原始数据截断)")

    except Exception as e:
        result.fail(f"collate_fn 一致性校验失败: {e}")
        traceback.print_exc()

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print("\n" + result.summary())
    sys.exit(1 if result.failed else 0)


if __name__ == "__main__":
    main()