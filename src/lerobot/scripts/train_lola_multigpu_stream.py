#!/usr/bin/env python
"""
LoLA 多卡分布式流式训练脚本 - 使用 StreamingLeRobotDataset

本脚本使用 LoLAStreamingDataset 从远程存储（如 Azure Blob 挂载）流式加载数据。
适用于数据通过挂载方式访问的场景，无需将整个数据集下载到本地。

与 train_lola_multigpu.py 的区别：
- 使用 LoLAStreamingDataset（IterableDataset）替代 LoLADataset（map-style）
- 不需要 DistributedSampler（IterableDataset 自带分片）
- 适用于远程挂载存储（Azure Blob 等）

使用方法:
    # 使用流式数据集训练
    torchrun --nproc_per_node=4 src/lerobot/scripts/train_lola_multigpu_stream.py \
        --dataset_repo_id lerobot/pusht \
        --strategy fsdp

    # DeepSpeed 训练
    torchrun --nproc_per_node=4 src/lerobot/scripts/train_lola_multigpu_stream.py \
        --dataset_repo_id lerobot/pusht \
        --strategy deepspeed

    # 单卡测试
    python src/lerobot/scripts/train_lola_multigpu_stream.py --devices 1 --dataset_repo_id lerobot/pusht
"""

import argparse
import datetime
import os
import sys
import time
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy, DeepSpeedStrategy

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lola_streaming_dataset import LoLAStreamingDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig, LoLAPolicy
from lerobot.policies.factory import make_pre_post_processors


# ----------------------------------------------------------------------
# LoLA Lightning Module - 包装实际模型
# ----------------------------------------------------------------------
class LoLALightningModule(pl.LightningModule):
    """LoLA 的 PyTorch Lightning 模块 - 包装实际的 LoLAPolicy"""

    def __init__(
        self,
        config: LoLAConfig,
        dataset_stats: dict | None = None,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 30000,
        warmup_ratio: float = 0.03,
        train_vlm: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config", "dataset_stats"])

        self.config = config
        self.dataset_stats = dataset_stats
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.train_vlm = train_vlm

        # 延迟加载模型到 setup() 阶段
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None

        # 计时
        self._step_start_time = None

    def setup(self, stage=None):
        """在 distributed 环境初始化后加载模型"""
        if self.policy is not None:
            return

        # 获取当前进程的 local_rank
        if hasattr(self.trainer.strategy, 'local_rank') and self.trainer.strategy.local_rank is not None:
            local_rank = self.trainer.strategy.local_rank
        else:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        print(f"[Rank {self.global_rank}] Loading LoLA Policy on {device}...")
        print(f"[Rank {self.global_rank}] VLM Path: {self.config.vlm_path}")

        self.policy = LoLAPolicy(self.config)
        self.policy._device = device
        self.policy.model = self.policy.model.to(device)
        self.policy.vlm = self.policy.vlm.to(device)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            dataset_stats=self.dataset_stats,
        )

        print(f"[Rank {self.global_rank}] LoLA Policy loaded on {device}!")
        print(f"[Rank {self.global_rank}] VLM device: {next(self.policy.vlm.parameters()).device}")
        print(f"[Rank {self.global_rank}] DiT device: {next(self.policy.model.parameters()).device}")

        if not self.train_vlm and hasattr(self.policy, 'vlm'):
            print(f"[Rank {self.global_rank}] Freezing VLM parameters...")
            for param in self.policy.vlm.parameters():
                param.requires_grad = False
            self.policy.vlm.eval()

        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.policy.parameters())
        print(f"[Rank {self.global_rank}] Trainable params: {trainable_params:,} / {total_params:,}")

    def forward(self, batch):
        return self.policy(batch)

    def _extract_special_fields(self, batch):
        """提取特殊字段，避免被preprocessor处理"""
        special_data = {}
        keys_to_extract = ["hist_actions_full", "hist_actions_mask", "hist_actions_length"]
        for key in keys_to_extract:
            if key in batch:
                special_data[key] = batch.pop(key)

        if "action" in batch:
            special_data["action"] = batch.pop("action")

        return special_data

    def _restore_special_fields(self, batch, special_data):
        """恢复特殊字段到batch"""
        batch.update(special_data)
        return batch

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        self._step_start_time = time.monotonic()

        special_data = self._extract_special_fields(batch)
        batch = self.preprocessor(batch)
        batch = self._restore_special_fields(batch, special_data)

        loss, loss_dict = self(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if k != "loss":
                self.log(f"train_{k}", v, prog_bar=False, sync_dist=True)

        if self._step_start_time is not None:
            update_s = round(time.monotonic() - self._step_start_time, 2)
            self.log("train_update_s", update_s, prog_bar=False, sync_dist=False)

        return loss

    def configure_optimizers(self):
        """配置优化器"""
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        from torch.optim.lr_scheduler import OneCycleLR
        warmup_ratio = min(self.hparams.warmup_ratio, 0.1)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.max_steps,
            pct_start=warmup_ratio,
            anneal_strategy='cos',
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


# ----------------------------------------------------------------------
# 数据集工具函数
# ----------------------------------------------------------------------
def create_lola_streaming_dataset(
    repo_id: str,
    config: LoLAConfig,
    root: str | None = None,
    episodes: list | None = None,
    image_transforms=None,
    max_history_length: int = 100,
    history_padding_side: str = "left",
    streaming: bool = True,
    buffer_size: int = 1000,
    seed: int = 42,
    shuffle: bool = True,
):
    """创建 LoLA 流式数据集"""
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = dataset_metadata.fps

    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in config.action_delta_indices]
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    print(f"[Dataset] delta_timestamps: {delta_timestamps}")

    dataset = LoLAStreamingDataset(
        repo_id=repo_id,
        max_history_length=max_history_length,
        action_chunk_size=config.action_chunk_size,
        history_padding_side=history_padding_side,
        root=root,
        episodes=episodes,
        image_transforms=image_transforms,
        delta_timestamps=delta_timestamps,
        streaming=streaming,
        buffer_size=buffer_size,
        seed=seed,
        shuffle=shuffle,
    )

    return dataset


def collate_fn(batch):
    """自定义 collate 函数，处理变长序列"""
    result = {}
    variable_length_keys = {"hist_actions_full", "hist_actions_mask"}

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if key == "task":
            result[key] = values
        elif key in variable_length_keys and isinstance(values[0], torch.Tensor):
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


# ----------------------------------------------------------------------
# FSDP 配置
# ----------------------------------------------------------------------
def get_fsdp_strategy():
    """获取 FSDP 策略配置"""
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, StateDictType
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    auto_wrap_policy = lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
        module, recurse, nonwrapped_numel,
        transformer_layer_cls={Qwen3_5DecoderLayer}
    )

    strategy = FSDPStrategy(
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        cpu_offload=False,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=True,
        state_dict_type=StateDictType.FULL_STATE_DICT,
    )
    return strategy


def get_deepspeed_strategy():
    """获取 DeepSpeed 策略配置"""
    strategy = DeepSpeedStrategy(
        stage=2,
        offload_optimizer=False,
        offload_parameters=False,
    )
    return strategy


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LoLA Multi-GPU Streaming Training")

    # 数据集参数
    parser.add_argument("--dataset_repo_id", type=str, default=None,
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Local dataset root directory (mount path)")
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="Specific episodes to load (optional)")

    # 训练策略参数
    parser.add_argument("--strategy", type=str, default="fsdp", choices=["fsdp", "deepspeed", "ddp"])
    parser.add_argument("--devices", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--precision", type=str, default="bf16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--save_every_n_steps", type=int, default=500,
                        help="Save checkpoint every N training steps")

    # 模型参数
    parser.add_argument("--vlm_path", type=str, default="/data_16T/deepseek/qwen3_5/Qwen3.5-4B/",
                        help="Path to local Qwen3.5-4B model")
    parser.add_argument("--train_vlm", action="store_true", help="Whether to train VLM")
    parser.add_argument("--ckpt_dir", type=str, default="/data_16T/deepseek/checkpoints/lola",
                        help="Path to save LoLA checkpoints.")

    # LoLA 特定参数
    parser.add_argument("--action_dim", type=int, default=14, help="Action dimension")
    parser.add_argument("--action_chunk_size", type=int, default=10, help="Action chunk size")
    parser.add_argument("--pred_chunk_size", type=int, default=50, help="Prediction chunk size")
    parser.add_argument("--n_obs_steps", type=int, default=1, help="Number of observation steps")

    # 历史 action 参数
    parser.add_argument("--max_history_length", type=int, default=100,
                        help="Maximum history length for padding/truncation")
    parser.add_argument("--history_padding_side", type=str, default="left", choices=["left", "right"],
                        help="Padding side for history actions")

    # 流式数据集参数
    parser.add_argument("--buffer_size", type=int, default=1000, help="Streaming shuffle buffer size")
    parser.add_argument("--streaming_seed", type=int, default=42, help="Streaming dataset seed")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable streaming shuffle")

    # DataLoader 参数
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    if args.dataset_repo_id is None and args.dataset_root is None:
        raise ValueError("Either --dataset_repo_id or --dataset_root must be provided.")

    # 设置策略
    if args.strategy == "fsdp":
        strategy = get_fsdp_strategy()
    elif args.strategy == "deepspeed":
        strategy = get_deepspeed_strategy()
    else:
        strategy = "auto"

    # 获取数据集元数据
    print(f"Loading dataset metadata from {args.dataset_repo_id or args.dataset_root}...")
    dataset_metadata = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
    )

    features = dataset_to_policy_features(dataset_metadata.features)
    if "action" in features:
        action_dim = features["action"].shape[0]
    else:
        action_dim = args.action_dim

    print(f"Dataset info:")
    print(f"  - Total episodes: {dataset_metadata.total_episodes}")
    print(f"  - Total frames: {dataset_metadata.total_frames}")
    print(f"  - FPS: {dataset_metadata.fps}")
    print(f"  - Action dim: {action_dim}")

    # 创建 LoLA 配置
    config = LoLAConfig(
        vlm_model_name="Qwen/Qwen3.5-4B",
        vlm_path=args.vlm_path,
        action_dim=action_dim,
        action_chunk_size=args.action_chunk_size,
        pred_chunk_size=args.pred_chunk_size,
        n_obs_steps=args.n_obs_steps,
        input_features={key: ft for key, ft in features.items() if ft.type != FeatureType.ACTION},
        output_features={key: ft for key, ft in features.items() if ft.type == FeatureType.ACTION},
        train_vlm=args.train_vlm,
        load_full_history=True,
        max_history_length=args.max_history_length,
        history_padding_side=args.history_padding_side,
    )

    # 创建流式数据集
    print("Creating streaming dataset...")
    train_dataset = create_lola_streaming_dataset(
        repo_id=args.dataset_repo_id,
        config=config,
        root=args.dataset_root,
        episodes=args.episodes,
        max_history_length=args.max_history_length,
        history_padding_side=args.history_padding_side,
        buffer_size=args.buffer_size,
        seed=args.streaming_seed,
        shuffle=not args.no_shuffle,
    )

    # IterableDataset 不使用 DistributedSampler，由数据集内部处理分片
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 创建模型
    model = LoLALightningModule(
        config=config,
        dataset_stats=dataset_metadata.stats,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        train_vlm=args.train_vlm,
    )

    # 回调函数
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(args.ckpt_dir, f"lola-stream-{time_str}")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="lola-stream-{step:06d}",
            save_top_k=-1,
            every_n_train_steps=args.save_every_n_steps,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    logger_name = args.dataset_repo_id.replace('/', '-') if args.dataset_repo_id else os.path.basename(args.dataset_root)
    logger = WandbLogger(
        project="lola-multigpu-stream",
        name=f"lola-{args.strategy}-{logger_name}",
        save_dir="logs",
    )

    # 创建 Trainer
    gradient_clip_val = 1.0 if args.strategy != "fsdp" else None

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        max_steps=args.max_steps,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=1,
        benchmark=True,
        enable_progress_bar=True,
        sync_batchnorm=True,
    )

    # 打印配置信息
    print("=" * 60)
    print("LoLA Multi-GPU Streaming Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset_repo_id or args.dataset_root}")
    print(f"Strategy: {args.strategy}")
    print(f"Devices: {args.devices}")
    print(f"Num Nodes: {args.num_nodes}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Save Every N Steps: {args.save_every_n_steps}")
    print(f"Precision: {args.precision}")
    print(f"VLM Path: {args.vlm_path}")
    print(f"Train VLM: {args.train_vlm}")
    print(f"Action Dim: {action_dim}")
    print(f"Action Chunk Size: {args.action_chunk_size}")
    print(f"Pred Chunk Size: {args.pred_chunk_size}")
    print(f"Max History Length: {args.max_history_length}")
    print(f"History Padding Side: {args.history_padding_side}")
    print(f"Streaming: True")
    print(f"Buffer Size: {args.buffer_size}")
    print("=" * 60)

    # 开始训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
    )

    print("Training completed!")
    print(f"Last checkpoint: {callbacks[0].last_model_path}")


if __name__ == "__main__":
    main()
