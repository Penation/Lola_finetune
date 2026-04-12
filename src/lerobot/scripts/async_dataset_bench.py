import time, os, sys, gc
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, '.')
import logging; logging.disable(logging.WARNING)

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lola_streaming_dataset import LoLAStreamingDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.lola import LoLAConfig

VARIABLE_LENGTH_KEYS = {'hist_actions_full', 'hist_actions_mask'}

def collate_decoded(items):
    result = {}
    for key in items[0].keys():
        values = [item[key] for item in items]
        if key == 'task':
            result[key] = values
        elif key in VARIABLE_LENGTH_KEYS and isinstance(values[0], torch.Tensor):
            max_len = max(v.shape[0] for v in values)
            padded_values = []
            for v in values:
                if v.shape[0] < max_len:
                    pad_len = max_len - v.shape[0]
                    if key == 'hist_actions_full':
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

dataset_root = '/data_6t_2/lerobot_v30/simpler_bridge_v3/'
dataset_metadata = LeRobotDatasetMetadata(None, root=dataset_root)
features = dataset_to_policy_features(dataset_metadata.features)
action_dim = features['action'].shape[0]

config = LoLAConfig(
    vlm_model_name='Qwen/Qwen3.5-4B', vlm_path='/tmp/dummy',
    action_dim=action_dim, action_chunk_size=10, pred_chunk_size=50,
    n_obs_steps=1,
    input_features={k: v for k, v in features.items() if v.type != FeatureType.ACTION},
    output_features={k: v for k, v in features.items() if v.type == FeatureType.ACTION},
    train_vlm=False, load_full_history=True, max_history_length=100, history_padding_side='left',
)

fps = dataset_metadata.fps
delta_timestamps = {
    'observation.state': [i / fps for i in config.observation_delta_indices],
    'action': [i / fps for i in config.action_delta_indices],
}
for key in dataset_metadata.camera_keys:
    delta_timestamps[key] = [i / fps for i in config.observation_delta_indices]

# Test with different decode_num_threads values
for n_threads in [1, 4, 8]:
    gc.collect()
    print(f'=== Async pipeline with decode_num_threads={n_threads} ===')
    dataset = LoLAStreamingDataset(
        repo_id=None, max_history_length=100, action_chunk_size=10,
        history_padding_side='left', root=dataset_root,
        delta_timestamps=delta_timestamps, streaming=True, buffer_size=1000,
        seed=42, shuffle=True, deferred_video_decode=True,
        async_decode=True, num_dataloader_workers=8,
        decode_device='cpu', decode_num_threads=n_threads,
    )
    loader = DataLoader(dataset, batch_size=16, num_workers=8, collate_fn=lambda b: b, pin_memory=False)

    # Warmup
    it = dataset.decode_iter(loader)
    for _ in range(5):
        items = next(it)
        batch = collate_decoded(items)
    dataset.shutdown_decode_pipeline(); del it; gc.collect()

    # Measure
    dataset2 = LoLAStreamingDataset(
        repo_id=None, max_history_length=100, action_chunk_size=10,
        history_padding_side='left', root=dataset_root,
        delta_timestamps=delta_timestamps, streaming=True, buffer_size=1000,
        seed=42, shuffle=True, deferred_video_decode=True,
        async_decode=True, num_dataloader_workers=8,
        decode_device='cpu', decode_num_threads=n_threads,
    )
    loader2 = DataLoader(dataset2, batch_size=16, num_workers=8, collate_fn=lambda b: b, pin_memory=False)
    start = time.time()
    for i, decoded in enumerate(dataset2.decode_iter(loader2)):
        batch = collate_decoded(decoded)
        if i >= 49: break
    elapsed = time.time() - start
    print(f'  {50/elapsed:.1f} batch/s, {elapsed:.1f}s for 50 batches')
    dataset2.shutdown_decode_pipeline()
    del loader2, dataset2; gc.collect()