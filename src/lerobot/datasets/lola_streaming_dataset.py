#!/usr/bin/env python

# Copyright 2025 LoLA Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LoLA 流式数据集，支持从 Azure Blob 等远程存储流式加载数据。

与 LoLADataset 的区别：
- LoLADataset 继承 LeRobotDataset（map-style），通过索引随机访问
- LoLAStreamingDataset 继承 StreamingLeRobotDataset（iterable），通过迭代器顺序访问
- LoLAStreamingDataset 使用 Backtrackable 迭代器的 peek_back 功能加载历史 action

使用方法：
    dataset = LoLAStreamingDataset(
        repo_id="lerobot/pusht",
        max_history_length=100,
        action_chunk_size=10,
        delta_timestamps={...},
        streaming=True,
    )

    for item in dataset:
        # item["hist_actions_full"]: [padded_length, action_dim]
        # item["hist_actions_mask"]: [padded_length] (1=真实, 0=padding)
"""

import numpy as np
import torch

from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.utils import item_to_torch, safe_shard


class LoLAStreamingDataset(StreamingLeRobotDataset):
    """
    支持流式加载完整历史 action 的 LoLA 数据集。

    在 StreamingLeRobotDataset 基础上，额外提供：
    - hist_actions_full: 从 episode 开始到当前帧的所有 action
    - hist_actions_mask: 标识哪些是真实 action vs padding
    - hist_actions_length: 标量，真实 action 数量
    """

    def __init__(
        self,
        repo_id: str,
        max_history_length: int = 100,
        action_chunk_size: int = 10,
        history_padding_side: str = "left",
        root: str | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        streaming: bool = True,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        rng=None,
        shuffle: bool = True,
    ):
        """
        Args:
            repo_id: 数据集仓库 ID
            max_history_length: 历史 action 最大长度，超过则截断
            action_chunk_size: action 块大小，历史长度补齐到该值的整数倍
            history_padding_side: padding 方向，"left" 或 "right"
            root: 本地数据集根目录（挂载路径）
            episodes: 指定加载的 episode 列表
            image_transforms: 图像变换
            delta_timestamps: 时间戳偏移配置
            tolerance_s: 时间戳容差
            revision: 版本
            force_cache_sync: 是否强制同步缓存
            streaming: 是否流式加载
            buffer_size: 流式 shuffle 缓冲区大小
            max_num_shards: 最大分片数
            seed: 随机种子
            rng: 随机数生成器
            shuffle: 是否 shuffle
        """
        # 确保 lookback 足够大以支持历史 action 加载
        # max_history_length 决定了需要回看多少帧
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            streaming=streaming,
            buffer_size=buffer_size,
            max_num_shards=max_num_shards,
            seed=seed,
            rng=rng,
            shuffle=shuffle,
        )

        self.max_history_length = max_history_length
        self.action_chunk_size = action_chunk_size
        self.history_padding_side = history_padding_side

        # 获取 action 维度
        if "action" in self.meta.features:
            self.action_dim = self.meta.features["action"]["shape"][0]
        else:
            self.action_dim = 1

        print(f"[LoLAStreamingDataset] max_history_length: {max_history_length}")
        print(f"[LoLAStreamingDataset] action_chunk_size: {action_chunk_size}")
        print(f"[LoLAStreamingDataset] history_padding_side: {history_padding_side}")
        print(f"[LoLAStreamingDataset] action_dim: {self.action_dim}")

    def _get_window_steps(self, delta_timestamps=None, dynamic_bounds=False):
        """覆盖父类方法，确保 lookback 足够大以支持历史 action 加载"""
        lookback, lookahead = super()._get_window_steps(delta_timestamps, dynamic_bounds)
        # lookback 需要至少 max_history_length 以支持完整历史加载
        lookback = max(lookback, self.max_history_length)
        return lookback, lookahead

    def __iter__(self):
        """覆盖父类 __iter__，添加分布式数据分片支持。

        父类 StreamingLeRobotDataset.__iter__ 中所有 rank/worker 看到相同的数据，
        在分布式训练中会导致每个 GPU 重复处理相同样本，等效信息量不增加。

        本方法根据 rank × num_workers 分配不同的 shard 子集给每个并行单元，
        确保不同 GPU、不同 DataLoader worker 处理不同的数据分片。
        """
        if self.video_decoder_cache is None:
            from lerobot.datasets.video_utils import VideoDecoderCache
            self.video_decoder_cache = VideoDecoderCache()

        # 获取分布式信息
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        # 获取 DataLoader worker 信息
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # 总并行度 = world_size × num_workers
        # 每个并行单元处理不同的 shard 子集，避免数据重复
        total_parallel = world_size * num_workers
        parallel_id = rank * num_workers + worker_id

        # 取模轮询分配 shard
        assigned_shards = [idx for idx in range(self.num_shards) if idx % total_parallel == parallel_id]

        if not assigned_shards:
            # shard 数量少于并行度，当前单元无分配则跳过
            return

        # 为每个 rank 使用不同的种子偏移，保证各 rank 的 shuffle 顺序不同
        rng = np.random.default_rng(self.seed + rank) if not self.shuffle else np.random.default_rng(self.rng.integers(0, 2**31) + rank)
        buffer_indices_generator = self._iter_random_indices(rng, self.buffer_size)

        idx_to_backtrack_dataset = {
            idx: self._make_backtrackable_dataset(safe_shard(self.hf_dataset, idx, self.num_shards))
            for idx in assigned_shards
        }

        # 以下逻辑与父类相同：随机选择 shard → 随机采样 buffer 中的帧
        frames_buffer = []
        while available_shards := list(idx_to_backtrack_dataset.keys()):
            shard_key = next(self._infinite_generator_over_elements(rng, available_shards))
            backtrack_dataset = idx_to_backtrack_dataset[shard_key]

            try:
                for frame in self.make_frame(backtrack_dataset):
                    if len(frames_buffer) == self.buffer_size:
                        i = next(buffer_indices_generator)
                        yield frames_buffer[i]
                        frames_buffer[i] = frame
                    else:
                        frames_buffer.append(frame)
                    break
            except (RuntimeError, StopIteration):
                del idx_to_backtrack_dataset[shard_key]

        rng.shuffle(frames_buffer)
        yield from frames_buffer

    def make_frame(self, dataset_iterator):
        """生成帧数据，包含完整历史 action"""
        # 调用父类获取基础帧数据
        for result in super().make_frame(dataset_iterator):
            # 在结果上附加历史 action
            result = self._add_history_actions(result, dataset_iterator)
            yield result

    def _add_history_actions(self, item, dataset_iterator):
        """
        为当前帧添加完整历史 action。

        使用 Backtrackable 迭代器的 peek_back 功能回溯到 episode 开始。
        """
        current_episode_idx = item["episode_index"].item() if isinstance(item["episode_index"], torch.Tensor) else item["episode_index"]

        # 当前帧的 action
        current_action = item.get("action")
        if current_action is None:
            return item

        # 确保当前 action 是 [action_dim] 形状
        if current_action.dim() > 1:
            current_action = current_action[0] if current_action.shape[0] == 1 else current_action[-1]

        # 回溯收集历史 action
        past_actions = []
        past_masks = []

        # 从最远的历史帧开始回溯
        max_lookback = min(self.max_history_length - 1, len(dataset_iterator.history()))

        for steps_back in range(max_lookback, 0, -1):
            try:
                if dataset_iterator.can_peek_back(steps_back):
                    past_item = dataset_iterator.peek_back(steps_back)
                    past_item = item_to_torch(past_item)

                    if past_item["episode_index"] == current_episode_idx:
                        past_action = past_item.get("action")
                        if past_action is not None:
                            if past_action.dim() > 1:
                                past_action = past_action[0] if past_action.shape[0] == 1 else past_action[-1]
                            past_actions.append(past_action)
                            past_masks.append(True)
                        else:
                            # action 不可用，用 zero padding
                            past_actions.append(torch.zeros(self.action_dim, dtype=current_action.dtype))
                            past_masks.append(False)
                    else:
                        # 跨越 episode 边界，用 zero padding
                        past_actions.append(torch.zeros(self.action_dim, dtype=current_action.dtype))
                        past_masks.append(False)
                else:
                    # 无法回溯到足够远，用 zero padding
                    past_actions.append(torch.zeros(self.action_dim, dtype=current_action.dtype))
                    past_masks.append(False)
            except Exception:
                past_actions.append(torch.zeros(self.action_dim, dtype=current_action.dtype))
                past_masks.append(False)

        # 添加当前帧的 action
        past_actions.append(current_action)
        past_masks.append(True)

        # 转换为 tensor
        hist_actions = torch.stack(past_actions)  # [actual_length, action_dim]
        hist_actions_mask = torch.BoolTensor(past_masks)  # [actual_length]

        actual_history_length = len(past_actions)

        # 计算补齐后的长度（action_chunk_size 的整数倍）
        padded_length = ((actual_history_length + self.action_chunk_size - 1) // self.action_chunk_size) * self.action_chunk_size

        # 确保不超过 max_history_length
        if padded_length > self.max_history_length:
            padded_length = (self.max_history_length // self.action_chunk_size) * self.action_chunk_size
            if actual_history_length > padded_length:
                truncate_length = actual_history_length - padded_length
                hist_actions = hist_actions[truncate_length:]
                hist_actions_mask = hist_actions_mask[truncate_length:]
                actual_history_length = padded_length

        # Padding 到 action_chunk_size 的整数倍
        if actual_history_length < padded_length:
            pad_length = padded_length - actual_history_length
            padding_actions = torch.zeros(pad_length, self.action_dim, dtype=hist_actions.dtype)
            padding_mask = torch.zeros(pad_length, dtype=torch.bool)

            if self.history_padding_side == "left":
                hist_actions = torch.cat([padding_actions, hist_actions], dim=0)
                hist_actions_mask = torch.cat([padding_mask, hist_actions_mask], dim=0)
            else:
                hist_actions = torch.cat([hist_actions, padding_actions], dim=0)
                hist_actions_mask = torch.cat([hist_actions_mask, padding_mask], dim=0)

        # 添加到 item
        item["hist_actions_full"] = hist_actions  # [padded_length, action_dim]
        item["hist_actions_mask"] = hist_actions_mask  # [padded_length]
        item["hist_actions_length"] = torch.tensor(actual_history_length, dtype=torch.long)

        return item
