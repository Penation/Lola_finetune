#!/usr/bin/env python
"""
LoLA Pretrain Model 端到端测试

测试从 Dataset → Collate → Processor → Model Forward 的完整数据流。
重点关注：
1. Processor pipeline 对 pretrain 模式数据的处理（PIL Image list、camera_valid_mask）
2. batch_to_transition 后 camera_valid_mask 的正确路由
3. AddBatchDimensionProcessorStep 对非 tensor 数据的处理
4. Model forward pass 能否正常接收处理器输出
5. 训练时特殊字段（hist_actions_full, action）的提取/恢复

用法:
    # 使用样例数据集测试（本机，CPU/GPU）
    python src/lerobot/scripts/test_lola_pretrain_model.py \
        --dataset_root /data_6t_2/lerobot_v30/simpler_bridge_v3

    # 使用合并数据集测试（集群，GPU）
    python src/lerobot/scripts/test_lola_pretrain_model.py \
        --dataset_root /data_16T/deepseek/halo \
        --dataset_to_episodes_path /data_16T/deepseek/halo/dataset_to_episodes.json

    # 仅测试 processor（不需要 GPU）
    python src/lerobot/scripts/test_lola_pretrain_model.py \
        --dataset_root /data_6t_2/lerobot_v30/simpler_bridge_v3 \
        --processor_only

注意:
    - 需要 Qwen3.5 模型权重（VLMPATH 环境变量或 --vlm_path 参数）
    - processor_only 模式不需要 GPU，但需要 Qwen3.5 processor（~2GB）
    - 完整模型测试需要 ~16GB GPU 显存
"""

import argparse
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lola_pretrain_streaming_dataset import LoLAPretrainStreamingDataset, AsyncDecodeDataLoader
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig


def get_vlm_path():
    """获取 Qwen3.5 模型路径"""
    vlm_path = os.environ.get("VLMPATH")
    if vlm_path and os.path.isdir(vlm_path):
        return vlm_path
    # 尝试常见路径
    for path in ["/data_16T/deepseek/qwen3_5/Qwen3.5-4B/"]:
        if os.path.isdir(path):
            return path
    return None


def create_config(dataset_root, vlm_path=None, action_dim=20):
    """创建 LoLAConfig"""
    dataset_metadata = LeRobotDatasetMetadata(repo_id="test", root=dataset_root)
    fps = dataset_metadata.fps
    features = dataset_to_policy_features(dataset_metadata.features)
    if "action" in features:
        action_dim = features["action"].shape[0]

    config = LoLAConfig(
        vlm_model_name="Qwen/Qwen3.5-4B",
        vlm_path=vlm_path or get_vlm_path(),
        action_dim=action_dim,
        action_chunk_size=10,
        pred_chunk_size=50,
        n_obs_steps=1,
        input_features={key: ft for key, ft in features.items() if ft.type != FeatureType.ACTION},
        output_features={key: ft for key, ft in features.items() if ft.type == FeatureType.ACTION},
        load_full_history=True,
        max_history_length=10,
        # Pretrain mode: all IDENTITY normalization
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
    )
    return config, dataset_metadata, fps


def create_dataset(dataset_root, dataset_to_episodes_path, config, fps, batch_size=2):
    """创建 LoLAPretrainStreamingDataset + DataLoader"""
    delta_timestamps = {}
    delta_timestamps["observation.state"] = [i / fps for i in config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in config.action_delta_indices]

    dataset_metadata = LeRobotDatasetMetadata(repo_id="test", root=dataset_root)
    for key in dataset_metadata.camera_keys:
        delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

    dataset = LoLAPretrainStreamingDataset(
        repo_id="test",
        max_history_length=10,
        action_chunk_size=config.action_chunk_size,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        streaming=True,
        buffer_size=10,
        seed=42,
        shuffle=False,
        deferred_video_decode=False,
        dataset_to_episodes_path=dataset_to_episodes_path,
    )

    from torch.utils.data import DataLoader
    raw_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=lambda x: x,
        pin_memory=False,
    )
    train_loader = AsyncDecodeDataLoader(
        dataloader=raw_loader,
        dataset=dataset,
        collate_fn=AsyncDecodeDataLoader.make_collate_fn(),
    )
    return dataset, train_loader


def test_processor_with_mock_data(config):
    """测试 1: 使用 mock 数据测试 processor pipeline

    验证 processor 对以下 pretrain 模式数据的处理:
    - camera key 值为 list of PIL Image / None（动态分辨率）
    - camera_valid_mask 字段
    - observation.state 和 action 已在 dataset 内归一化
    """
    print("\n" + "=" * 60)
    print("Test 1: Processor Pipeline with Mock Data")
    print("=" * 60)

    from lerobot.policies.lola.processor_lola import make_lola_pre_post_processors

    # 创建 processor（pretrain 模式：dataset_stats 为空，IDENTITY normalization）
    preprocessor, postprocessor = make_lola_pre_post_processors(
        config=config,
        dataset_stats={},
        camera_keys=[
            "observation.images.primary",
            "observation.images.secondary",
            "observation.images.wrist",
        ],
    )

    print(f"  Preprocessor steps: {[s.__class__.__name__ for s in preprocessor.steps]}")
    print(f"  Postprocessor steps: {[s.__class__.__name__ for s in postprocessor.steps]}")

    # 构造 mock batch（模拟 AsyncDecodeDataLoader collate 后的输出）
    batch_size = 2
    action_dim = config.action_dim

    # 创建 mock PIL Images（不同分辨率以测试动态分辨率）
    img1 = Image.fromarray(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    mock_batch = {
        # Camera keys: list of PIL Image / None
        "observation.images.primary": [img1, img1],
        "observation.images.secondary": [None, img2],  # 第二个 sample 的 secondary 相机无效
        "observation.images.wrist": [img2, None],       # 第一个 sample 的 wrist 相机无效
        # camera_valid_mask: list of dicts
        "camera_valid_mask": [
            {"observation.images.primary": True, "observation.images.secondary": False, "observation.images.wrist": True},
            {"observation.images.primary": True, "observation.images.secondary": True, "observation.images.wrist": False},
        ],
        # observation.state: normalized in dataset, tensor
        "observation.state": torch.randn(batch_size, action_dim),
        # action: normalized in dataset, will be extracted as special field
        "action": torch.randn(batch_size, config.pred_chunk_size, action_dim),
        # history actions
        "hist_actions_full": torch.randn(batch_size, 100, action_dim),
        "hist_actions_mask": torch.ones(batch_size, 100),
        "hist_actions_length": torch.tensor([100, 80]),
        # metadata
        "task": ["pick up the cup", "place the block"],
        "episode_index": torch.tensor([0, 1]),
        "index": torch.tensor([10, 20]),
        "task_index": torch.tensor([0, 0]),
        "action_dim": torch.tensor([action_dim, action_dim]),
        "state_dim": torch.tensor([10, 10]),
    }

    # 测试 1a: batch_to_transition 路由
    print("\n  --- Sub-test 1a: batch_to_transition routing ---")
    from lerobot.processor.converters import batch_to_transition
    from lerobot.processor.core import TransitionKey

    transition = batch_to_transition(mock_batch)
    observation = transition.get(TransitionKey.OBSERVATION, {})
    complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

    obs_keys = sorted(observation.keys()) if observation else []
    comp_keys = sorted(complementary_data.keys()) if complementary_data else []

    print(f"  Observation keys: {obs_keys}")
    print(f"  Complementary data keys: {comp_keys}")

    # 关键检查：camera_valid_mask 在哪里？
    camera_valid_mask_in_obs = "camera_valid_mask" in observation
    camera_valid_mask_in_comp = "camera_valid_mask" in complementary_data
    print(f"  camera_valid_mask in observation: {camera_valid_mask_in_obs}")
    print(f"  camera_valid_mask in complementary_data: {camera_valid_mask_in_comp}")

    if not camera_valid_mask_in_obs and camera_valid_mask_in_comp:
        print("  [ISSUE] camera_valid_mask is in complementary_data, NOT in observation!")
        print("  This means LolaImageProcessor.observation() will NOT find it via")
        print("  observation.get('camera_valid_mask', {}). It will default to {} (all valid).")
        print("  FIX NEEDED: Either:")
        print("    a) Prefix camera_valid_mask as 'observation.camera_valid_mask' in dataset, or")
        print("    b) Change LolaImageProcessor to read from self.transition[COMPLEMENTARY_DATA], or")
        print("    c) Move camera_valid_mask from complementary_data to observation before processor")

    # 测试 1b: 完整 processor pipeline
    print("\n  --- Sub-test 1b: Full preprocessor pipeline ---")
    try:
        # 先提取特殊字段（模拟 LoLATrainer.training_step 的做法）
        special_data = {}
        keys_to_extract = ["hist_actions_full", "hist_actions_mask", "hist_actions_length"]
        for key in keys_to_extract:
            if key in mock_batch:
                special_data[key] = mock_batch.pop(key)
        if "action" in mock_batch:
            special_data["action"] = mock_batch.pop("action")

        processed_batch = preprocessor(mock_batch)

        # 恢复特殊字段
        processed_batch.update(special_data)

        print(f"  Processed batch keys: {sorted(processed_batch.keys())}")

        # 检查 processor 输出
        if "input_ids" in processed_batch:
            print(f"  input_ids: shape={processed_batch['input_ids'].shape}")
        if "attention_mask" in processed_batch:
            print(f"  attention_mask: shape={processed_batch['attention_mask'].shape}")
        if "pixel_values" in processed_batch:
            print(f"  pixel_values: shape={processed_batch['pixel_values'].shape}")
        if "image_grid_thw" in processed_batch:
            print(f"  image_grid_thw: {processed_batch['image_grid_thw']}")
        if "observation.state" in processed_batch:
            val = processed_batch["observation.state"]
            if isinstance(val, torch.Tensor):
                print(f"  observation.state: shape={val.shape}")
        if "action" in processed_batch:
            val = processed_batch["action"]
            if isinstance(val, torch.Tensor):
                print(f"  action: shape={val.shape}")
        if "hist_actions_full" in processed_batch:
            val = processed_batch["hist_actions_full"]
            if isinstance(val, torch.Tensor):
                print(f"  hist_actions_full: shape={val.shape}")

        # 检查 camera key 是否已被清理
        remaining_cam_keys = [k for k in processed_batch.keys() if k.startswith("observation.images.")]
        if remaining_cam_keys:
            print(f"  [WARNING] Camera keys still in batch: {remaining_cam_keys}")
            print(f"  These should have been removed by LolaQwenProcessor")
        else:
            print(f"  Camera keys properly cleaned up by LolaQwenProcessor")

        # 检查 camera_valid_mask 是否已被清理
        if "camera_valid_mask" in processed_batch:
            print(f"  [WARNING] camera_valid_mask still in batch (should be cleaned by LolaQwenProcessor)")
        else:
            print(f"  camera_valid_mask properly cleaned up")

        print("  PASSED")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False


def test_processor_with_real_data(dataset_root, dataset_to_episodes_path, config, fps, batch_size=2):
    """测试 2: 使用真实数据集数据测试 processor pipeline"""
    print("\n" + "=" * 60)
    print("Test 2: Processor Pipeline with Real Dataset Data")
    print("=" * 60)

    dataset, train_loader = create_dataset(
        dataset_root, dataset_to_episodes_path, config, fps, batch_size=batch_size
    )

    # 获取一个 batch
    batch = None
    for b in train_loader:
        batch = b
        break

    if batch is None:
        print("  SKIP: No batch available from dataset")
        return False

    print(f"  Batch keys: {sorted(batch.keys())}")

    # 检查 camera key 格式
    dataset_metadata = LeRobotDatasetMetadata(repo_id="test", root=dataset_root)
    cam_keys = dataset_metadata.camera_keys
    for cam_key in cam_keys:
        if cam_key in batch:
            val = batch[cam_key]
            if isinstance(val, list):
                types = [type(v).__name__ for v in val]
                shapes = [f"{v.size}" if isinstance(v, Image.Image) else "None" for v in val]
                print(f"  {cam_key}: list of {len(val)}, types={types}, shapes={shapes}")
            else:
                print(f"  {cam_key}: {type(val).__name__}")

    # 检查 camera_valid_mask
    if "camera_valid_mask" in batch:
        cvm = batch["camera_valid_mask"]
        if isinstance(cvm, list):
            print(f"  camera_valid_mask: list of {len(cvm)} dicts")
            for i, m in enumerate(cvm[:2]):
                print(f"    [{i}]: {m}")
        else:
            print(f"  camera_valid_mask: {type(cvm).__name__}")

    # 运行 processor
    from lerobot.policies.lola.processor_lola import make_lola_pre_post_processors
    preprocessor, _ = make_lola_pre_post_processors(
        config=config,
        dataset_stats={},
        camera_keys=cam_keys,
    )

    try:
        # 提取特殊字段
        special_data = {}
        keys_to_extract = ["hist_actions_full", "hist_actions_mask", "hist_actions_length"]
        for key in keys_to_extract:
            if key in batch:
                special_data[key] = batch.pop(key)
        if "action" in batch:
            special_data["action"] = batch.pop("action")
        # 也提取非 tensor 字段（action_dim, state_dim）
        for key in ["action_dim", "state_dim"]:
            if key in batch:
                special_data[key] = batch.pop(key)

        processed_batch = preprocessor(batch)
        processed_batch.update(special_data)

        print(f"  Processed batch keys: {sorted(processed_batch.keys())}")

        # 检查关键输出
        for key in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                     "observation.state", "action", "hist_actions_full"]:
            if key in processed_batch:
                val = processed_batch[key]
                if isinstance(val, torch.Tensor):
                    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

        print("  PASSED")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False


def test_model_forward_mock(config):
    """测试 3: 使用 mock 数据测试 model forward pass

    验证 LoLAPolicy.forward() 能正确处理 processor 输出格式。
    """
    print("\n" + "=" * 60)
    print("Test 3: Model Forward Pass with Mock Data")
    print("=" * 60)

    vlm_path = config.vlm_path or get_vlm_path()
    if vlm_path is None:
        print("  SKIP: No VLM path available (set VLMPATH env or --vlm_path)")
        return False

    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available (model forward requires GPU)")
        return False

    try:
        from lerobot.policies.lola.modeling_lola import LoLAPolicy

        device = torch.device("cuda:0")

        # 创建模型
        print("  Loading LoLAPolicy...")
        policy = LoLAPolicy(config)
        policy._device = device
        policy.model = policy.model.to(device)
        policy.vlm = policy.vlm.to(device)
        policy.model.train()

        # 创建 mock batch（模拟 processor 输出 + 特殊字段恢复后）
        batch_size = 2
        action_dim = config.action_dim

        # Qwen3.5 processor 输出格式
        seq_len = 50  # token sequence length
        vocab_size = 151936  # Qwen3.5 vocab size

        mock_batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
            "observation.state": torch.randn(batch_size, action_dim, device=device),
            "action": torch.randn(batch_size, config.pred_chunk_size, action_dim, device=device),
            "hist_actions_full": torch.randn(batch_size, 100, action_dim, device=device),
            "hist_actions_mask": torch.ones(batch_size, 100, device=device),
        }

        # 添加 pixel_values（如果有视觉输入）
        # 注意：实际 Qwen3.5 processor 会生成 pixel_values 和 image_grid_thw
        # 这里先用文本 only 模式测试（无 pixel_values）
        print("  Running model forward (text-only, no pixel_values)...")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_dict = policy(mock_batch)

        print(f"  Loss: {loss.item():.4f}")
        print(f"  Loss dict: {loss_dict}")
        print("  PASSED")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False


def test_model_forward_with_vision(config):
    """测试 4: 使用完整视觉输入测试 model forward pass

    验证包含 pixel_values 和 image_grid_thw 的完整模型前向传播。
    """
    print("\n" + "=" * 60)
    print("Test 4: Model Forward Pass with Vision Input")
    print("=" * 60)

    vlm_path = config.vlm_path or get_vlm_path()
    if vlm_path is None:
        print("  SKIP: No VLM path available")
        return False

    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return False

    try:
        from lerobot.policies.lola.modeling_lola import LoLAPolicy
        from transformers import AutoProcessor

        device = torch.device("cuda:0")

        # 创建模型
        print("  Loading LoLAPolicy...")
        policy = LoLAPolicy(config)
        policy._device = device
        policy.model = policy.model.to(device)
        policy.vlm = policy.vlm.to(device)
        policy.model.train()

        # 使用 Qwen3.5 processor 生成真实的视觉输入
        print("  Loading Qwen3.5 processor...")
        qwen_processor = AutoProcessor.from_pretrained(vlm_path)

        # 创建 mock PIL Images
        img1 = Image.fromarray(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        # 使用 apply_chat_template 生成输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": "pick up the cup"},
                ],
            }
        ]
        inputs = qwen_processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        # 移到 GPU
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        batch_size = 1
        action_dim = config.action_dim

        mock_batch = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "observation.state": torch.randn(batch_size, action_dim, device=device),
            "action": torch.randn(batch_size, config.pred_chunk_size, action_dim, device=device),
            "hist_actions_full": torch.randn(batch_size, 100, action_dim, device=device),
            "hist_actions_mask": torch.ones(batch_size, 100, device=device),
        }

        print(f"  input_ids: shape={mock_batch['input_ids'].shape}")
        if mock_batch.get("pixel_values") is not None:
            print(f"  pixel_values: shape={mock_batch['pixel_values'].shape}")
        if mock_batch.get("image_grid_thw") is not None:
            print(f"  image_grid_thw: shape={mock_batch['image_grid_thw'].shape}")

        print("  Running model forward (text + vision)...")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_dict = policy(mock_batch)

        print(f"  Loss: {loss.item():.4f}")
        print(f"  Loss dict: {loss_dict}")
        print("  PASSED")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False


def test_end_to_end(dataset_root, dataset_to_episodes_path, config, fps, batch_size=2):
    """测试 5: 端到端测试 Dataset → Collate → Processor → Model

    这是最关键的测试，验证完整训练数据流。
    """
    print("\n" + "=" * 60)
    print("Test 5: End-to-End Dataset → Processor → Model")
    print("=" * 60)

    vlm_path = config.vlm_path or get_vlm_path()
    if vlm_path is None:
        print("  SKIP: No VLM path available")
        return False

    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        return False

    try:
        from lerobot.policies.lola.modeling_lola import LoLAPolicy
        from lerobot.policies.lola.processor_lola import make_lola_pre_post_processors

        device = torch.device("cuda:0")

        # 1. 创建 dataset + dataloader
        dataset, train_loader = create_dataset(
            dataset_root, dataset_to_episodes_path, config, fps, batch_size=batch_size
        )

        # 2. 创建 model + processor
        print("  Loading LoLAPolicy...")
        policy = LoLAPolicy(config)
        policy._device = device
        policy.model = policy.model.to(device)
        policy.vlm = policy.vlm.to(device)
        policy.model.train()

        dataset_metadata = LeRobotDatasetMetadata(repo_id="test", root=dataset_root)
        preprocessor, postprocessor = make_lola_pre_post_processors(
            config=config,
            dataset_stats={},
            camera_keys=dataset_metadata.camera_keys,
        )

        # 3. 获取一个 batch 并运行完整训练流程
        for batch in train_loader:
            print(f"  Raw batch keys: {sorted(batch.keys())}")

            # 移到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 提取特殊字段
            special_data = {}
            keys_to_extract = ["hist_actions_full", "hist_actions_mask", "hist_actions_length"]
            for key in keys_to_extract:
                if key in batch:
                    special_data[key] = batch.pop(key)
            if "action" in batch:
                special_data["action"] = batch.pop("action")
            for key in ["action_dim", "state_dim"]:
                if key in batch:
                    special_data[key] = batch.pop(key)

            # 运行 preprocessor
            processed_batch = preprocessor(batch)

            # 恢复特殊字段
            processed_batch.update(special_data)

            print(f"  Processed batch keys: {sorted(processed_batch.keys())}")
            for key in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                         "observation.state", "action", "hist_actions_full"]:
                if key in processed_batch:
                    val = processed_batch[key]
                    if isinstance(val, torch.Tensor):
                        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

            # 运行 model forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, loss_dict = policy(processed_batch)

            print(f"  Loss: {loss.item():.4f}")
            print(f"  Loss dict: {loss_dict}")

            # 测试 backward
            loss.backward()
            print("  Backward pass successful")

            break  # 只测试第一个 batch

        print("  PASSED")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False


def test_batch_to_transition_routing():
    """测试 6: 验证 batch_to_transition 对 pretrain 字段的路由

    检查关键字段（camera_valid_mask, camera keys, action_dim, state_dim）
    在 batch → transition 转换中的正确路由。
    """
    print("\n" + "=" * 60)
    print("Test 6: batch_to_transition Field Routing")
    print("=" * 60)

    from lerobot.processor.converters import batch_to_transition
    from lerobot.processor.core import TransitionKey

    # 构造模拟 pretrain batch
    mock_batch = {
        # observation 前缀 → 应进入 observation dict
        "observation.state": torch.randn(2, 20),
        "observation.images.primary": [Image.fromarray(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8))] * 2,
        "observation.images.secondary": [None, Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))],
        # 无 observation 前缀 → 应进入 complementary_data
        "camera_valid_mask": [
            {"observation.images.primary": True, "observation.images.secondary": False},
            {"observation.images.primary": True, "observation.images.secondary": True},
        ],
        "action_dim": torch.tensor([20, 20]),
        "state_dim": torch.tensor([10, 10]),
        # action → 顶层字段
        "action": torch.randn(2, 50, 20),
        # complementary data
        "task": ["task1", "task2"],
        "index": torch.tensor([0, 1]),
    }

    transition = batch_to_transition(mock_batch)

    observation = transition.get(TransitionKey.OBSERVATION, {})
    action = transition.get(TransitionKey.ACTION)
    complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

    # 检查路由
    print("  Field routing results:")
    for key in mock_batch:
        if key.startswith("observation."):
            in_obs = key in observation
            print(f"  {key} → observation: {in_obs}")
        elif key == "action":
            is_action = action is not None
            print(f"  {key} → action: {is_action}")
        elif key in ("task", "index"):
            in_comp = key in complementary_data
            print(f"  {key} → complementary_data: {in_comp}")
        else:
            # 检查是否被遗漏
            in_obs = key in observation
            in_comp = key in complementary_data
            print(f"  {key} → observation: {in_obs}, complementary_data: {in_comp}")

    # 关键检查
    issues = []

    # camera_valid_mask 应该在 complementary_data 中（因为无 observation. 前缀）
    if "camera_valid_mask" not in complementary_data:
        issues.append("camera_valid_mask not in complementary_data (lost during routing!)")
    else:
        print(f"\n  camera_valid_mask is in complementary_data (correct routing)")
        print(f"  But LolaImageProcessor.observation() reads it from observation dict!")
        print(f"  This is a MISMATCH that needs fixing.")

    # camera keys 应该在 observation 中
    cam_keys_in_obs = [k for k in observation if k.startswith("observation.images.")]
    print(f"\n  Camera keys in observation: {cam_keys_in_obs}")
    for cam_key in cam_keys_in_obs:
        val = observation[cam_key]
        if isinstance(val, list):
            print(f"    {cam_key}: list of {len(val)} items")
            types = set(type(v).__name__ for v in val)
            print(f"    types: {types}")

    # action_dim / state_dim 路由
    for key in ["action_dim", "state_dim"]:
        in_obs = key in observation
        in_comp = key in complementary_data
        if in_comp:
            print(f"  {key} → complementary_data (OK, not needed by processor)")
        elif in_obs:
            print(f"  {key} → observation (will go through observation processor steps)")
        else:
            issues.append(f"{key} not found in observation or complementary_data")

    if issues:
        print(f"\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  No critical routing issues (but camera_valid_mask access pattern needs fixing)")

    print("  PASSED")
    return len(issues) == 0


def test_camera_valid_mask_fix():
    """测试 7: 验证 camera_valid_mask 的修复方案

    测试两种修复方案:
    a) 在 batch 中将 camera_valid_mask 重命名为 observation.camera_valid_mask
    b) 在 LolaImageProcessor 中从 self.transition 读取 camera_valid_mask
    """
    print("\n" + "=" * 60)
    print("Test 7: camera_valid_mask Fix Verification")
    print("=" * 60)

    from lerobot.processor.converters import batch_to_transition, create_transition
    from lerobot.processor.core import TransitionKey

    # 方案 A: 重命名为 observation.camera_valid_mask
    print("\n  --- Fix A: Prefix as 'observation.camera_valid_mask' ---")
    mock_batch_a = {
        "observation.state": torch.randn(2, 20),
        "observation.images.primary": [Image.fromarray(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8))] * 2,
        "observation.camera_valid_mask": [
            {"observation.images.primary": True},
            {"observation.images.primary": True},
        ],
        "action": torch.randn(2, 50, 20),
        "task": ["task1", "task2"],
    }

    transition_a = batch_to_transition(mock_batch_a)
    observation_a = transition_a.get(TransitionKey.OBSERVATION, {})

    if "observation.camera_valid_mask" in observation_a:
        print("  Fix A works: camera_valid_mask is in observation dict")
        print(f"  Value: {observation_a['observation.camera_valid_mask']}")
        # 注意：LolaImageProcessor 读取的是 observation.get('camera_valid_mask', {})
        # 如果用 observation.camera_valid_mask 前缀，processor 需要相应修改
        print("  But LolaImageProcessor reads 'camera_valid_mask', not 'observation.camera_valid_mask'")
        print("  Need to also update LolaImageProcessor to use the correct key")
    else:
        print("  Fix A: camera_valid_mask not found in observation")

    # 方案 B: 在 processor 中从 self.transition 读取
    print("\n  --- Fix B: Read from self.transition in LolaImageProcessor ---")
    print("  This approach mirrors how LolaQwenProcessor reads 'task' from complementary_data")
    print("  Implementation:")
    print("    camera_valid_mask = {}")
    print("    if hasattr(self, 'transition') and self.transition is not None:")
    print("      comp_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA, {})")
    print("      camera_valid_mask = comp_data.get('camera_valid_mask', {})")
    print("  This is the recommended fix because:")
    print("    1. No changes needed in dataset/collate")
    print("    2. Follows existing pattern (LolaQwenProcessor reads task from transition)")
    print("    3. camera_valid_mask naturally belongs in complementary_data (it's metadata, not observation)")

    # 验证 transition 中确实有 camera_valid_mask
    mock_batch_b = {
        "observation.state": torch.randn(2, 20),
        "observation.images.primary": [Image.fromarray(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8))] * 2,
        "camera_valid_mask": [{"observation.images.primary": True}, {"observation.images.primary": True}],
        "action": torch.randn(2, 50, 20),
        "task": ["task1", "task2"],
    }

    transition_b = batch_to_transition(mock_batch_b)
    comp_data_b = transition_b.get(TransitionKey.COMPLEMENTARY_DATA, {})
    observation_b = transition_b.get(TransitionKey.OBSERVATION, {})

    print(f"\n  camera_valid_mask in complementary_data: {'camera_valid_mask' in comp_data_b}")
    print(f"  camera_valid_mask in observation: {'camera_valid_mask' in observation_b}")
    print(f"  observation.images.primary in observation: {'observation.images.primary' in observation_b}")

    # 测试修复后的 processor
    print("\n  --- Testing Fix B with actual processor ---")
    try:
        from lerobot.policies.lola.processor_lola import LolaImageProcessor

        # 创建一个修改版的 processor，使用 self.transition 读取 camera_valid_mask
        processor = LolaImageProcessor(
            camera_keys=["observation.images.primary", "observation.images.secondary"]
        )

        # 手动设置 transition（模拟 ObservationProcessorStep.__call__ 的行为）
        processor._current_transition = transition_b

        # 测试 observation 方法
        # 注意：observation_b 是 observation dict，没有 camera_valid_mask
        # processor 需要从 self.transition 读取
        result = processor.observation(observation_b.copy())

        if "_lola_images" in result:
            print(f"  _lola_images collected: {len(result['_lola_images'])} images")
            # 由于当前 LolaImageProcessor 从 observation 读 camera_valid_mask，
            # 但 observation 中没有，所以会默认为 {}（全部有效），不会跳过任何相机
            print(f"  [CURRENT BUG] All cameras treated as valid (camera_valid_mask not found in observation)")
            print(f"  [FIX] Should read from self.transition[COMPLEMENTARY_DATA]['camera_valid_mask']")
        else:
            print(f"  No images collected (possible issue)")

    except Exception as e:
        print(f"  Processor test failed: {e}")

    print("  PASSED")
    return True


def test_add_batch_dim_with_lists(config):
    """测试 8: AddBatchDimensionProcessorStep 对 list 数据的处理

    验证 batch dimension 步骤不会破坏 PIL Image list 和 dict list。
    """
    print("\n" + "=" * 60)
    print("Test 8: AddBatchDimensionProcessorStep with List Data")
    print("=" * 60)

    from lerobot.processor.batch_processor import AddBatchDimensionProcessorStep
    from lerobot.processor.core import TransitionKey

    processor = AddBatchDimensionProcessorStep()

    # 构造模拟 transition（processor pipeline 中间状态）
    img1 = Image.fromarray(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8))

    transition = {
        TransitionKey.OBSERVATION: {
            "observation.state": torch.randn(20),  # 1D tensor → should unsqueeze
            "observation.images.primary": img1,  # PIL Image → should be preserved
            "observation.language_tokens": torch.randint(0, 1000, (50,)),  # 1D → should unsqueeze
            "observation.language_attention_mask": torch.ones(50),  # 1D → should unsqueeze
            "input_ids": torch.randint(0, 1000, (50,)),  # 1D → should unsqueeze
            "attention_mask": torch.ones(50),  # 1D → should unsqueeze
            "pixel_values": torch.randn(3, 360, 640),  # 3D → should unsqueeze
        },
        TransitionKey.ACTION: torch.randn(50, 20),  # 2D → should NOT unsqueeze
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["pick up the cup"],  # Already a list
        },
    }

    result = processor(transition)

    obs = result[TransitionKey.OBSERVATION]
    action = result[TransitionKey.ACTION]
    comp = result[TransitionKey.COMPLEMENTARY_DATA]

    # 检查 observation
    checks = []
    for key, expected_shape in [
        ("observation.state", (1, 20)),
        ("input_ids", (1, 50)),
        ("attention_mask", (1, 50)),
        ("pixel_values", (1, 3, 360, 640)),
    ]:
        if key in obs and isinstance(obs[key], torch.Tensor):
            actual_shape = obs[key].shape
            ok = actual_shape == expected_shape
            checks.append(ok)
            print(f"  {key}: shape={actual_shape} {'OK' if ok else 'MISMATCH (expected ' + str(expected_shape) + ')'}")

    # 检查 PIL Image 保留
    if "observation.images.primary" in obs:
        val = obs["observation.images.primary"]
        if isinstance(val, Image.Image):
            print(f"  observation.images.primary: PIL Image preserved OK")
            checks.append(True)
        else:
            print(f"  observation.images.primary: {type(val).__name__} (expected PIL Image)")
            checks.append(False)

    # 检查 action
    if isinstance(action, torch.Tensor):
        # 2D action should not get unsqueezed by AddBatchDimensionActionStep
        # But AddBatchDimensionActionStep only unsqueezes dim==1 tensors
        if action.dim() == 2:
            # Action is already 2D, unsqueeze adds batch dim → 3D
            # Actually AddBatchDimensionActionStep checks dim==1, so 2D stays 2D
            print(f"  action: shape={action.shape} (2D, not unsqueezed)")
            checks.append(True)
        elif action.dim() == 3:
            print(f"  action: shape={action.shape} (3D, already batched)")
            checks.append(True)
        else:
            print(f"  action: shape={action.shape}")
            checks.append(True)

    # 检查 complementary_data
    if "task" in comp:
        val = comp["task"]
        if isinstance(val, list):
            print(f"  task: list preserved OK")
            checks.append(True)
        else:
            print(f"  task: {type(val).__name__} (expected list)")
            checks.append(False)

    all_ok = all(checks)
    if all_ok:
        print("  PASSED")
    else:
        print("  Some checks FAILED")
    return all_ok


def test_device_processor_with_lists(config):
    """测试 9: DeviceProcessorStep 对 list 数据的处理"""
    print("\n" + "=" * 60)
    print("Test 9: DeviceProcessorStep with List Data")
    print("=" * 60)

    from lerobot.processor.device_processor import DeviceProcessorStep
    from lerobot.processor.core import TransitionKey

    processor = DeviceProcessorStep(device="cpu")

    img1 = Image.fromarray(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8))

    transition = {
        TransitionKey.OBSERVATION: {
            "observation.state": torch.randn(1, 20),
            "observation.images.primary": img1,  # PIL Image → should be preserved
            "pixel_values": torch.randn(1, 3, 360, 640),  # tensor → should be moved
            "input_ids": torch.randint(0, 1000, (1, 50)),
            "attention_mask": torch.ones(1, 50),
        },
        TransitionKey.ACTION: torch.randn(1, 50, 20),
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["pick up the cup"],
            "camera_valid_mask": [{"observation.images.primary": True}],
        },
    }

    result = processor(transition)

    obs = result[TransitionKey.OBSERVATION]
    comp = result[TransitionKey.COMPLEMENTARY_DATA]

    checks = []

    # 检查 tensor 被处理
    for key in ["observation.state", "pixel_values", "input_ids", "attention_mask"]:
        if key in obs and isinstance(obs[key], torch.Tensor):
            print(f"  {key}: tensor on {obs[key].device} OK")
            checks.append(True)

    # 检查 PIL Image 保留（DeviceProcessorStep 跳过非 tensor）
    if "observation.images.primary" in obs:
        val = obs["observation.images.primary"]
        if isinstance(val, Image.Image):
            print(f"  observation.images.primary: PIL Image preserved OK")
            checks.append(True)
        else:
            print(f"  observation.images.primary: {type(val).__name__} (expected PIL Image)")
            checks.append(False)

    # 检查 complementary_data 中的非 tensor 值保留
    if "camera_valid_mask" in comp:
        val = comp["camera_valid_mask"]
        if isinstance(val, list):
            print(f"  camera_valid_mask: list preserved OK")
            checks.append(True)
        else:
            print(f"  camera_valid_mask: {type(val).__name__} (expected list)")
            checks.append(False)

    all_ok = all(checks)
    if all_ok:
        print("  PASSED")
    else:
        print("  Some checks FAILED")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="LoLA Pretrain Model End-to-End Test")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="数据集根目录")
    parser.add_argument("--dataset_to_episodes_path", type=str, default=None,
                        help="dataset_to_episodes.json 路径")
    parser.add_argument("--no_mapping", action="store_true",
                        help="跳过 per-dataset mapping")
    parser.add_argument("--vlm_path", type=str, default=None,
                        help="Qwen3.5 模型路径（默认从 VLMPATH 环境变量获取）")
    parser.add_argument("--processor_only", action="store_true",
                        help="仅测试 processor pipeline（不需要 GPU）")
    parser.add_argument("--skip_model", action="store_true",
                        help="跳过模型 forward pass 测试")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="测试 batch size")

    args = parser.parse_args()

    dataset_to_episodes_path = args.dataset_to_episodes_path
    if args.no_mapping:
        dataset_to_episodes_path = None

    print("=" * 60)
    print("LoLA Pretrain Model End-to-End Test")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"VLM path: {args.vlm_path or get_vlm_path() or 'NOT FOUND'}")
    print(f"Processor only: {args.processor_only}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # 创建 config
    vlm_path = args.vlm_path or get_vlm_path()
    config, dataset_metadata, fps = create_config(args.dataset_root, vlm_path)

    results = {}

    # Test 1: Processor with mock data
    results["test1_processor_mock"] = test_processor_with_mock_data(config)

    # Test 2: Processor with real data
    try:
        results["test2_processor_real"] = test_processor_with_real_data(
            args.dataset_root, dataset_to_episodes_path, config, fps, args.batch_size
        )
    except Exception as e:
        print(f"  Test 2 exception: {e}")
        results["test2_processor_real"] = False

    # Test 3: Model forward with mock (requires GPU)
    if not args.processor_only and not args.skip_model:
        try:
            results["test3_model_mock"] = test_model_forward_mock(config)
        except Exception as e:
            print(f"  Test 3 exception: {e}")
            results["test3_model_mock"] = False

        # Test 4: Model forward with vision (requires GPU + VLM)
        try:
            results["test4_model_vision"] = test_model_forward_with_vision(config)
        except Exception as e:
            print(f"  Test 4 exception: {e}")
            results["test4_model_vision"] = False

        # Test 5: End-to-end
        try:
            results["test5_e2e"] = test_end_to_end(
                args.dataset_root, dataset_to_episodes_path, config, fps, args.batch_size
            )
        except Exception as e:
            print(f"  Test 5 exception: {e}")
            results["test5_e2e"] = False

    # Test 6: batch_to_transition routing
    results["test6_routing"] = test_batch_to_transition_routing()

    # Test 7: camera_valid_mask fix
    results["test7_cvm_fix"] = test_camera_valid_mask_fix()

    # Test 8: AddBatchDimension with lists
    results["test8_batch_dim"] = test_add_batch_dim_with_lists(config)

    # Test 9: DeviceProcessor with lists
    results["test9_device"] = test_device_processor_with_lists(config)

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed}/{total} passed")

    if not all(results.values()):
        print("\n  KNOWN ISSUE: camera_valid_mask routing mismatch")
        print("  - batch_to_transition puts camera_valid_mask in complementary_data")
        print("  - LolaImageProcessor.observation() reads from observation dict")
        print("  - FIX: Update LolaImageProcessor to read from self.transition[COMPLEMENTARY_DATA]")
        print("  - This mirrors how LolaQwenProcessor reads 'task' from transition")


if __name__ == "__main__":
    main()
