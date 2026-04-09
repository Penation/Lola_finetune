#!/usr/bin/env python
"""
LoLA 多卡分布式训练测试脚本

支持两种分布式策略:
1. FSDP (Fully Sharded Data Parallel)
2. DeepSpeed

使用方法:
    # FSDP 测试 (4卡)
    torchrun --nproc_per_node=4 src/lerobot/scripts/test_lola_multigpu.py --strategy fsdp

    # DeepSpeed 测试 (4卡)
    torchrun --nproc_per_node=4 src/lerobot/scripts/test_lola_multigpu.py --strategy deepspeed

    # 或者使用 pytorch-lightning 的 CLI
    python src/lerobot/scripts/test_lola_multigpu.py --strategy fsdp --devices 4
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy, DeepSpeedStrategy

# 环境变量设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ----------------------------------------------------------------------
# 模拟数据集
# ----------------------------------------------------------------------
class MockLoLADataset(Dataset):
    """模拟 LoLA 训练数据集"""
    
    def __init__(
        self,
        num_samples: int = 1000,
        action_dim: int = 20,
        action_chunk_size: int = 10,
        pred_chunk_size: int = 50,
        vlm_hidden_size: int = 2560,
        vlm_seq_len: int = 64,
        num_vlm_layers: int = 33,
    ):
        self.num_samples = num_samples
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.pred_chunk_size = pred_chunk_size
        self.vlm_hidden_size = vlm_hidden_size
        self.vlm_seq_len = vlm_seq_len
        self.num_vlm_layers = num_vlm_layers
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 模拟 hidden_states (所有 VLM 层)
        # 实际训练时由 VLM 前向传播生成
        hidden_states = [
            torch.randn(self.vlm_seq_len, self.vlm_hidden_size)
            for _ in range(self.num_vlm_layers)
        ]
        
        # 模拟历史动作 (最近的动作序列)
        hist_actions = torch.randn(self.action_chunk_size, self.action_dim)
        
        # 模拟目标动作 (预测的动作序列)
        target_actions = torch.randn(self.pred_chunk_size, self.action_dim)
        
        # 模拟 input_ids
        input_ids = torch.randint(0, 100000, (self.vlm_seq_len,))
        
        return {
            "hidden_states_all_layers": hidden_states,
            "hist_actions": hist_actions,
            "target_actions": target_actions,
            "input_ids": input_ids,
        }


def collate_fn(batch):
    """自定义 collate 函数处理 hidden_states 元组"""
    # 合并 hidden_states
    batch_size = len(batch)
    num_layers = len(batch[0]["hidden_states_all_layers"])
    
    # 将 hidden_states 转换为 [B, Seq, Hidden] 格式
    hidden_states_batched = []
    for layer_idx in range(num_layers):
        layer_hidden = torch.stack([item["hidden_states_all_layers"][layer_idx] for item in batch])
        hidden_states_batched.append(layer_hidden)
    hidden_states_all_layers = tuple(hidden_states_batched)
    
    # 其他数据正常 batch
    return {
        "hidden_states_all_layers": hidden_states_all_layers,
        "hist_actions": torch.stack([item["hist_actions"] for item in batch]),
        "target_actions": torch.stack([item["target_actions"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
    }


# ----------------------------------------------------------------------
# LoLA Lightning Module
# ----------------------------------------------------------------------
class LoLALightningModule(pl.LightningModule):
    """LoLA 的 PyTorch Lightning 模块"""
    
    def __init__(
        self,
        action_dim: int = 20,
        action_chunk_size: int = 10,
        pred_chunk_size: int = 50,
        dit_hidden_size: int = 1536,
        vlm_hidden_size: int = 2560,
        vlm_extract_layers: tuple = (8, 16, 24),
        time_sampling_beta_alpha: float = 1.5,
        time_sampling_beta_beta: float = 1.0,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 30000,
        warmup_ratio: float = 0.03,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.pred_chunk_size = pred_chunk_size
        self.dit_hidden_size = dit_hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # 构建 DiT 模型组件
        self._build_model()
        
    def _build_model(self):
        """构建模型组件"""
        vlm_concat_dim = self.hparams.vlm_hidden_size * len(self.hparams.vlm_extract_layers)
        
        # VLM 特征投影
        self.vlm_feature_proj = nn.Linear(vlm_concat_dim, self.dit_hidden_size)
        self.empty_token_proj = nn.Linear(vlm_concat_dim, self.dit_hidden_size)
        
        # Action Encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_chunk_size * self.action_dim, self.dit_hidden_size),
            nn.LayerNorm(self.dit_hidden_size, eps=1e-6),
            nn.SiLU(),
            nn.Linear(self.dit_hidden_size, self.dit_hidden_size)
        )
        
        # Action Decoder
        self.action_decoder = nn.Sequential(
            nn.LayerNorm(self.dit_hidden_size, eps=1e-6),
            nn.Linear(self.dit_hidden_size, self.dit_hidden_size),
            nn.SiLU(),
            nn.Linear(self.dit_hidden_size, self.action_chunk_size * self.action_dim)
        )
        
        # 简化的 DiT 主干 (实际实现使用 FLUX.2 blocks)
        self.dit_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.dit_hidden_size,
                nhead=12,
                dim_feedforward=self.dit_hidden_size * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(4)  # 简化为 4 层用于测试
        ])
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, self.dit_hidden_size),
            nn.SiLU(),
            nn.Linear(self.dit_hidden_size, self.dit_hidden_size)
        )
        
    def create_sinusoidal_pos_embedding(self, time: torch.Tensor, dimension: int) -> torch.Tensor:
        """生成 Timestep 的正弦位置编码"""
        import math
        half_dim = dimension // 2
        fraction = torch.linspace(0.0, 1.0, half_dim, dtype=torch.float32, device=time.device)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * time[:, None]
        return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    
    def forward(self, hidden_states_all_layers, hist_actions, target_actions, time=None, noise=None):
        """前向传播 - Flow Matching x-pred + v-loss"""
        batch_size = target_actions.shape[0]
        device = target_actions.device
        
        # 获取目标数据类型（兼容 DeepSpeed BF16）
        target_dtype = self.vlm_feature_proj.weight.dtype
        
        # 1. 提取 VLM 特征
        extract_layers = [l + 1 for l in self.hparams.vlm_extract_layers]  # +1 跳过 embedding
        selected_hiddens = [hidden_states_all_layers[i].to(target_dtype) for i in extract_layers]
        stacked_features = torch.cat(selected_hiddens, dim=-1)  # [B, Seq, Hidden*3]
        
        # 分离 VLM 特征和 empty token
        vlm_features = stacked_features[:, :-1, :]
        empty_token = stacked_features[:, -1, :]
        
        vlm_emb = self.vlm_feature_proj(vlm_features)
        empty_emb = self.empty_token_proj(empty_token)
        
        # 2. 编码动作 - 转换为目标 dtype（兼容 DeepSpeed BF16）
        b, seq_len, d = hist_actions.shape
        hist_actions = hist_actions.to(target_dtype)
        hist_chunks = self.action_encoder(
            hist_actions.view(b, seq_len // self.action_chunk_size, -1)
        )
        
        b, seq_len, d = target_actions.shape
        target_actions = target_actions.to(target_dtype)
        target_chunks = self.action_encoder(
            target_actions.view(b, seq_len // self.action_chunk_size, -1)
        )
        
        # 3. Flow Matching 加噪
        if noise is None:
            noise = torch.randn_like(target_chunks)
        if time is None:
            dist = torch.distributions.Beta(
                self.hparams.time_sampling_beta_alpha,
                self.hparams.time_sampling_beta_beta
            )
            time = dist.sample((batch_size,)).to(device)
        
        # 转换 time 为目标 dtype（兼容 DeepSpeed BF16）
        t_expand = time[:, None, None].to(target_dtype)
        x_t = t_expand * noise + (1 - t_expand) * target_chunks
        
        # Ground Truth Flow
        u_t = noise - target_chunks
        
        # 4. Time embedding - 转换为目标 dtype
        time_emb = self.create_sinusoidal_pos_embedding(time, 256).to(device).to(target_dtype)
        time_emb = self.time_embed(time_emb) + empty_emb
        
        # 5. DiT 前向传播
        # 简化版：直接拼接所有序列
        combined = torch.cat([vlm_emb, hist_chunks, x_t], dim=1)
        
        for layer in self.dit_layers:
            combined = layer(combined)
        
        # 提取目标部分
        target_len = x_t.shape[1]
        pred_x0_chunks = combined[:, -target_len:, :]
        
        # 6. 计算 v-loss
        t_expand_clamped = t_expand.clamp(min=1e-5)
        v_pred = (x_t - pred_x0_chunks) / t_expand_clamped
        
        loss = F.mse_loss(v_pred, u_t, reduction="none")
        return loss.mean()
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        loss = self(
            hidden_states_all_layers=batch["hidden_states_all_layers"],
            hist_actions=batch["hist_actions"],
            target_actions=batch["target_actions"],
        )
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        loss = self(
            hidden_states_all_layers=batch["hidden_states_all_layers"],
            hist_actions=batch["hist_actions"],
            target_actions=batch["target_actions"],
        )
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        # Cosine decay scheduler
        from torch.optim.lr_scheduler import OneCycleLR
        # 使用 warmup_ratio 确保 pct_start 在 0-1 之间
        warmup_ratio = min(self.hparams.warmup_ratio, 0.1)  # 最大 10%
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
# FSDP 配置
# ----------------------------------------------------------------------
def get_fsdp_strategy():
    """获取 FSDP 策略配置"""
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
    
    # 正确的 MixedPrecision 配置
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    strategy = FSDPStrategy(
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
        cpu_offload=False,  # 不使用 CPU offload
        mixed_precision=mixed_precision,  # 混合精度
        auto_wrap_policy=size_based_auto_wrap_policy,  # 自动分片策略
        # 注意：PyTorch Lightning 2.6+ 会自动处理 state_dict_type
        # 不要显式设置 state_dict_type，避免兼容性问题
    )
    return strategy


def get_deepspeed_strategy():
    """获取 DeepSpeed 策略配置"""
    strategy = DeepSpeedStrategy(
        stage=2,  # ZeRO Stage 2
        offload_optimizer=False,  # 不使用 CPU offload
        # 注意：precision 由 Trainer 的 precision 参数控制
    )
    return strategy


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LoLA Multi-GPU Training Test")
    parser.add_argument("--strategy", type=str, default="fsdp", choices=["fsdp", "deepspeed", "ddp"])
    parser.add_argument("--devices", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--precision", type=str, default="bf16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--val_check_interval", type=int, default=100)
    args = parser.parse_args()
    
    # 设置策略
    if args.strategy == "fsdp":
        strategy = get_fsdp_strategy()
    elif args.strategy == "deepspeed":
        strategy = get_deepspeed_strategy()
    else:
        strategy = "auto"  # DDP or auto
    
    # 创建数据集
    train_dataset = MockLoLADataset(
        num_samples=args.num_samples,
        action_dim=20,
        action_chunk_size=10,
        pred_chunk_size=50,
    )
    
    val_dataset = MockLoLADataset(
        num_samples=args.num_samples // 5,
        action_dim=20,
        action_chunk_size=10,
        pred_chunk_size=50,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # 创建模型
    model = LoLALightningModule(
        action_dim=20,
        action_chunk_size=10,
        pred_chunk_size=50,
        dit_hidden_size=1536,
        vlm_hidden_size=2560,
        vlm_extract_layers=(8, 16, 24),
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
    )
    
    # 回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/lola",
            filename="lola-{step:06d}-{train_loss:.4f}",
            save_top_k=3,
            monitor="train_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Logger
    logger = WandbLogger(
        project="lola-multigpu",
        name=f"lola-{args.strategy}",
        save_dir="logs",
    )
    
    # 创建 Trainer
    # 注意：FSDP 不支持 gradient_clip_algorithm='norm'，需要使用 'value'
    gradient_clip_val = 1.0 if args.strategy != "fsdp" else None
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        max_steps=args.max_steps,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=gradient_clip_val,  # FSDP 不支持 norm clipping
        accumulate_grad_batches=1,
        benchmark=True,
        enable_progress_bar=True,
        sync_batchnorm=True,
    )
    
    # 打印配置信息
    print("=" * 60)
    print("LoLA Multi-GPU Training Test")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Devices: {args.devices}")
    print(f"Num Nodes: {args.num_nodes}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Precision: {args.precision}")
    print("=" * 60)
    
    # 开始训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    print("Training completed!")
    print(f"Best model path: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
