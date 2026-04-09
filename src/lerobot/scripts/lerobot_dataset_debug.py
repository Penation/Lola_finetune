import argparse
import os
import sys
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy, DeepSpeedStrategy

from tqdm import tqdm
# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 关键：让 PyTorch 延迟初始化 CUDA，避免在 import 时占用 GPU
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig, LoLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# ----------------------------------------------------------------------
# 数据集工具函数
# ----------------------------------------------------------------------
def create_lola_dataset(
    repo_id: str,
    n_obs_steps: int = 1,
    n_hist_steps: int = 1,
    pred_chunk_size: int = 50,
    root: str | None = None,
    episodes: list | None = None,
    image_transforms=None,
    video_backend: str | None = None,
) -> LeRobotDataset:
    """
    创建 LoLA 训练用的 LeRobotDataset。
    
    根据 LoLA 的配置设置 delta_timestamps：
    - observation.state: 用于历史动作输入
    - action: 用于预测目标动作
    - observation.images.*: 用于视觉输入（如果存在）
    
    Args:
        repo_id: 数据集仓库ID (如 "lerobot/pusht")
        config: LoLA 配置
        root: 本地数据集根目录
        episodes: 指定加载的 episode 列表
        image_transforms: 图像变换
        video_backend: 视频后端
        
    Returns:
        配置好的 LeRobotDataset
    """
    # 获取数据集元数据以确定 fps
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = dataset_metadata.fps
    observation_delta_indices = list(range(-n_obs_steps + 1, 1))
    action_delta_indices = list(range(-n_hist_steps, pred_chunk_size))
    
    # 构建 delta_timestamps
    # 将帧索引转换为时间戳（秒）
    delta_timestamps = {}
    
    # 观测状态：使用 n_obs_steps 个历史帧
    delta_timestamps["observation.state"] = [
        i / fps for i in observation_delta_indices
    ]
    
    # 动作：预测 pred_chunk_size 步
    delta_timestamps["action"] = [
        i / fps for i in action_delta_indices
    ]
    
    # 图像/视频观测：与状态同步
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [
            i / fps for i in observation_delta_indices
        ]
    
    print(f"[Dataset] delta_timestamps: {delta_timestamps}")
    
    # 创建数据集
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        image_transforms=image_transforms,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )
    
    return dataset

# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LoLA Multi-GPU Training with LeRobotDataset")
    
    # 数据集参数
    parser.add_argument("--dataset_repo_id", type=str, default=None,
                        help="HuggingFace dataset repo ID (e.g., lerobot/pusht)")
    parser.add_argument("--dataset_root", type=str, default="/data_16T/deepseek/lerobot30/stanford_hydra_dataset",
                        help="Local dataset root directory (optional)")
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
    parser.add_argument("--val_check_interval", type=int, default=100)
    
    # 模型参数
    parser.add_argument("--vlm_path", type=str, default="/data_16T/deepseek/qwen3_5/Qwen3.5-4B/",
                        help="Path to local Qwen3.5-4B model")
    parser.add_argument("--train_vlm", action="store_true", help="Whether to train VLM (default: False)")
    
    # LoLA 特定参数
    parser.add_argument("--action_dim", type=int, default=14, help="Action dimension")
    parser.add_argument("--action_chunk_size", type=int, default=10, help="Action chunk size")
    parser.add_argument("--pred_chunk_size", type=int, default=50, help="Prediction chunk size")
    parser.add_argument("--n_obs_steps", type=int, default=1, help="Number of observation steps")
    
    # 验证集比例
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    
    args = parser.parse_args()

    # 获取数据集元数据
    print(f"Loading dataset metadata from {args.dataset_repo_id}...")
    dataset_metadata = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
    )

    # 获取数据集的 features 并转换为 policy features
    features = dataset_to_policy_features(dataset_metadata.features)
    
    # 获取 action_dim 从数据集
    if "action" in features:
        action_dim = features["action"].shape[0]
    else:
        action_dim = args.action_dim

    print(f"Dataset info:")
    print(f"  - Total episodes: {dataset_metadata.total_episodes}")
    print(f"  - Total frames: {dataset_metadata.total_frames}")
    print(f"  - FPS: {dataset_metadata.fps}")
    print(f"  - Action dim: {action_dim}")
    print(f"  - Features: {list(features.keys())}")

    # 创建训练和验证数据集
    print("Creating datasets...")
    train_dataset = create_lola_dataset(
        repo_id=args.dataset_repo_id,
        n_obs_steps=10,
        n_hist_steps=10,
        pred_chunk_size=50,
        root=args.dataset_root,
        episodes=args.episodes,
    )

    print(train_dataset)

    indice_list = torch.randint(0, len(train_dataset), (10,))
    indice_list = indice_list.tolist()
    for i in tqdm(indice_list):
        data = train_dataset[i]
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
        if key == "action_is_pad":
            print(value)
    # data = train_dataset[121]
    # for key, value in data.items():
    #     if isinstance(value, torch.Tensor):
    #         print(f"{key}: {value.shape}")
    #     elif isinstance(value, list):
    #         print(f"{key}: {len(value)}")
    #     elif isinstance(value, str):
    #         print(f"{key}: {value}")
    #     else:
    #         print(f"{key}: {type(value)}")
        
if __name__ == "__main__":
    main()
    