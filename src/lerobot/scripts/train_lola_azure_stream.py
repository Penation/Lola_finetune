#!/usr/bin/env python
"""
LoLA Azure 分布式流式训练脚本

使用 LoLAStreamingDataset 从 Azure Blob 等远程存储流式加载数据。
适用于数据通过挂载方式访问的场景，无需将整个数据集下载到本地。

与 train_lola_azure.py 的区别：
- 使用 LoLAStreamingDataset（IterableDataset）替代 LoLADataset（map-style）
- 不需要 DistributedSampler（IterableDataset 自带分片）
- 适用于远程挂载存储（Azure Blob 等）

使用方法:
    python src/lerobot/scripts/train_lola_azure_stream.py \
        --dataset_root /mnt/data/lerobot-dataset \
        --strategy ddp
"""

import argparse
import datetime
import logging
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lola_streaming_dataset import LoLAStreamingDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig, LoLAPolicy
from lerobot.policies.factory import make_pre_post_processors

logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] [Rank {os.environ.get('RANK', '0')}] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """从环境变量初始化分布式训练"""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world_size > 1:
        master_uri = f"tcp://{master_addr}:{master_port}"
        dist.init_process_group(
            backend="nccl",
            init_method=master_uri,
            world_size=world_size,
            timeout=timedelta(minutes=60),
            rank=world_rank,
        )
        logger.info(f"Distributed initialized: rank={world_rank}, local_rank={local_rank}, world_size={world_size}")
    else:
        logger.info(f"Single GPU mode: local_rank={local_rank}")

    return {
        "world_size": world_size,
        "local_rank": local_rank,
        "world_rank": world_rank,
        "node_rank": node_rank,
        "device": device,
        "is_distributed": world_size > 1,
    }


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


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

    logger.info(f"delta_timestamps: {delta_timestamps}")

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
    """自定义 collate 函数"""
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


# 导入训练器（与 train_lola_azure.py 完全相同）
from train_lola_azure import LoLATrainer


def main():
    dist_info = setup_distributed()

    parser = argparse.ArgumentParser(description="LoLA Azure Streaming Training")

    # 数据集参数
    parser.add_argument("--dataset_repo_id", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)

    # 训练参数
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    # 模型参数
    parser.add_argument("--vlm_path", type=str, default="/data_16T/deepseek/qwen3_5/Qwen3.5-4B/")
    parser.add_argument("--train_vlm", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="/data_16T/deepseek/checkpoints/lola")
    parser.add_argument("--resume", type=str, default=None)

    # LoLA 参数
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--action_chunk_size", type=int, default=10)
    parser.add_argument("--pred_chunk_size", type=int, default=50)
    parser.add_argument("--n_obs_steps", type=int, default=1)

    # 历史 action 参数
    parser.add_argument("--max_history_length", type=int, default=100)
    parser.add_argument("--history_padding_side", type=str, default="left", choices=["left", "right"])

    # 流式数据集参数
    parser.add_argument("--buffer_size", type=int, default=1000, help="Streaming shuffle buffer size")
    parser.add_argument("--streaming_seed", type=int, default=42, help="Streaming dataset seed")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable streaming shuffle")

    # Wandb 参数
    parser.add_argument("--wandb_project", type=str, default="lola-azure-stream")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")

    # DataLoader 参数
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    if args.dataset_repo_id is None and args.dataset_root is None:
        raise ValueError("Either --dataset_repo_id or --dataset_root must be provided.")

    if dist_info["world_rank"] == 0:
        logger.info("=" * 60)
        logger.info("LoLA Azure Streaming Training")
        logger.info("=" * 60)
        logger.info(f"Dataset: {args.dataset_repo_id or args.dataset_root}")
        logger.info(f"Strategy: {args.strategy}")
        logger.info(f"World Size: {dist_info['world_size']}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Streaming: True")
        logger.info(f"Buffer Size: {args.buffer_size}")
        logger.info(f"VLM Path: {args.vlm_path}")
        logger.info("=" * 60)

    # 获取数据集元数据
    logger.info("Loading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
    )

    features = dataset_to_policy_features(dataset_metadata.features)
    if "action" in features:
        action_dim = features["action"].shape[0]
    else:
        action_dim = args.action_dim

    logger.info(f"Dataset: {dataset_metadata.total_episodes} episodes, {dataset_metadata.total_frames} frames")

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
    logger.info("Creating streaming dataset...")
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

    # 创建训练器
    trainer = LoLATrainer(
        config=config,
        dataset_stats=dataset_metadata.stats,
        dist_info=dist_info,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        train_vlm=args.train_vlm,
        strategy=args.strategy,
        gradient_clip_val=args.gradient_clip_val,
        ckpt_dir=args.ckpt_dir,
        save_every_n_steps=args.save_every_n_steps,
        log_every_n_steps=args.log_every_n_steps,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
        wandb_id=args.wandb_id,
    )

    if args.disable_wandb:
        trainer.use_wandb = False

    trainer.setup_model()
    trainer.setup_optimizer()

    start_step = 0
    if args.resume:
        trainer.load_checkpoint(args.resume)
        start_step = trainer.global_step

    trainer.train(train_loader, start_step=start_step)

    cleanup_distributed()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
