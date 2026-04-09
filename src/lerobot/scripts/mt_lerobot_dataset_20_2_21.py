#!/usr/bin/env python3
"""
Multi-process version of lerobot_dataset_20_2_21.py
Optimized for:
1. Faster progress updates
2. Lower memory usage
3. Higher processing speed

Key optimizations:
- Streaming writes: write stats incrementally instead of accumulating
- Numpy-based stats: faster computation without intermediate DataFrames
- Batch writing: write stats in batches to reduce I/O overhead
- Memory-efficient: minimize data copies and IPC overhead
"""

import argparse
import logging
import os
import json
import gc
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import jsonlines
from scipy.spatial.transform import Rotation
import warnings

import polars as pl

from oxe_configs import OXE_DATASET_CONFIGS, StateEncoding as OXEStateEncoding, ActionEncoding as OXEActionEncoding

# Suppress warnings globally
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)


# ==================== Constants ====================
V20 = "v2.0"
V21 = "v2.1"
V30 = "v3.0"
EPISODES_STATS_PATH = Path("meta/episodes_stats.jsonl")
EPISODES_PATH = Path("meta/episodes.jsonl")


# ==================== State/Action Encoding Definitions ====================
from enum import IntEnum

class StateEncoding(IntEnum):
    NONE = -1
    POS_EULER = 1
    POS_QUAT = 2
    JOINT = 3
    JOINT_BIMANUAL = 4


class ActionEncoding(IntEnum):
    EEF_POS = 1
    JOINT_POS = 2
    JOINT_POS_BIMANUAL = 3
    EEF_R6 = 4


# ==================== Rotation Conversion Functions ====================

def euler_to_ort6d_batch(euler_angles: np.ndarray, convention: str = 'xyz') -> np.ndarray:
    """Convert euler angles to ort6d in batch."""
    if euler_angles.size == 0:
        return np.zeros((0, 6))
    rot_mat = Rotation.from_euler(convention, euler_angles).as_matrix()
    return rot_mat[..., :2].reshape(*rot_mat.shape[:-2], 6)


def quat_to_ort6d_batch(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to ort6d in batch."""
    if quat.size == 0:
        return np.zeros((0, 6))
    rot_mat = Rotation.from_quat(quat).as_matrix()
    return rot_mat[..., :2].reshape(*rot_mat.shape[:-2], 6)


def convert_euler_data_batch(data: np.ndarray) -> np.ndarray:
    """Convert POS_EULER format to ort6d format in batch."""
    if data.shape[1] == 7:
        xyz, euler, gripper = data[:, :3], data[:, 3:6], data[:, 6:7]
    elif data.shape[1] == 8:
        xyz, euler, gripper = data[:, :3], data[:, 3:6], data[:, 7:8]
    else:
        return data
    ort6d = euler_to_ort6d_batch(euler)
    return np.concatenate([xyz, ort6d, gripper], axis=1)


def convert_quat_data_batch(data: np.ndarray) -> np.ndarray:
    """Convert POS_QUAT format to ort6d format in batch."""
    if data.shape[1] != 8:
        return data
    xyz, quat, gripper = data[:, :3], data[:, 3:7], data[:, 7:8]
    quat_wxyz = np.roll(quat, 1, axis=1)
    ort6d = quat_to_ort6d_batch(quat_wxyz)
    return np.concatenate([xyz, ort6d, gripper], axis=1)


# ==================== Optimized Stats Functions using NumPy ====================

def compute_feature_stats_numpy(data: np.ndarray, axis: int = 0) -> dict:
    """
    Compute all statistics using NumPy for maximum efficiency.
    Avoids creating intermediate DataFrames.
    
    Args:
        data: NumPy array of shape (n_samples, n_dims) or (n_samples,)
        axis: Axis along which to compute statistics
    
    Returns:
        Dictionary with min, max, mean, std, count, and quantiles
    """
    if data is None or data.size == 0:
        return {"min": [], "max": [], "mean": [], "std": [], "count": [], 
                "q01": [], "q10": [], "q50": [], "q90": [], "q99": []}
    
    # Ensure 2D array
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    
    n_samples, n_dims = data.shape
    
    # Compute all statistics in one pass using numpy
    return {
        "min": np.nanmin(data, axis=axis).tolist(),
        "max": np.nanmax(data, axis=axis).tolist(),
        "mean": np.nanmean(data, axis=axis).tolist(),
        "std": np.nanstd(data, axis=axis, ddof=1).tolist(),
        "count": [int(n_samples)],
        "q01": np.nanquantile(data, 0.01, axis=axis).tolist(),
        "q10": np.nanquantile(data, 0.10, axis=axis).tolist(),
        "q50": np.nanquantile(data, 0.50, axis=axis).tolist(),
        "q90": np.nanquantile(data, 0.90, axis=axis).tolist(),
        "q99": np.nanquantile(data, 0.99, axis=axis).tolist(),
    }


# ==================== Lightweight Dataset Class ====================

class LightweightLeRobotDataset:
    """Lightweight dataset class for loading v2.0/v2.1 LeRobot datasets."""
    
    def __init__(self, root: Path):
        self.root = Path(root)
        self._info = None
        self._features = None
    
    @property
    def info(self) -> dict:
        if self._info is None:
            info_path = self.root / "meta" / "info.json"
            with open(info_path, "r") as f:
                self._info = json.load(f)
            for ft in self._info.get("features", {}).values():
                if "shape" in ft and isinstance(ft["shape"], list):
                    ft["shape"] = tuple(ft["shape"])
        return self._info
    
    @property
    def features(self) -> dict:
        if self._features is None:
            self._features = self.info.get("features", {})
        return self._features
    
    @property
    def total_episodes(self) -> int:
        return self.info.get("total_episodes", 0)
    
    def update_info(self, updates: dict):
        info = self.info.copy()
        info.update(updates)
        info_path = self.root / "meta" / "info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        self._info = None


# ==================== Helper Functions ====================

def get_dataset_encoding(dataset_name: str) -> tuple:
    """Get state and action encoding for a dataset from oxe_configs.py."""
    if dataset_name in OXE_DATASET_CONFIGS:
        config = OXE_DATASET_CONFIGS[dataset_name]
        state_enc_map = {
            OXEStateEncoding.NONE: StateEncoding.NONE,
            OXEStateEncoding.POS_EULER: StateEncoding.POS_EULER,
            OXEStateEncoding.POS_QUAT: StateEncoding.POS_QUAT,
            OXEStateEncoding.JOINT: StateEncoding.JOINT,
            OXEStateEncoding.JOINT_BIMANUAL: StateEncoding.JOINT_BIMANUAL,
            OXEStateEncoding.DUAL_POS_QUAT: StateEncoding.POS_QUAT,
        }
        action_enc_map = {
            OXEActionEncoding.EEF_POS: ActionEncoding.EEF_POS,
            OXEActionEncoding.JOINT_POS: ActionEncoding.JOINT_POS,
            OXEActionEncoding.JOINT_POS_BIMANUAL: ActionEncoding.JOINT_POS_BIMANUAL,
            OXEActionEncoding.EEF_R6: ActionEncoding.EEF_R6,
            OXEActionEncoding.DUAL_EEF_POS: ActionEncoding.EEF_POS,
        }
        oxe_state_enc = config.get("state_encoding", OXEStateEncoding.POS_EULER)
        oxe_action_enc = config.get("action_encoding", OXEActionEncoding.EEF_POS)
        return (state_enc_map.get(oxe_state_enc, StateEncoding.POS_EULER),
                action_enc_map.get(oxe_action_enc, ActionEncoding.EEF_POS))
    return (StateEncoding.POS_EULER, ActionEncoding.EEF_POS)


def copy_dataset_light(src_root: Path, dst_root: Path) -> bool:
    """Copy only data and meta folders."""
    import shutil
    if dst_root.exists():
        print(f"[WARN] Destination {dst_root} already exists")
        return False
    try:
        print(f"[INFO] Copying data and meta from {src_root} -> {dst_root}")
        dst_root.mkdir(parents=True, exist_ok=True)
        for subdir in ["data", "meta"]:
            src_subdir, dst_subdir = src_root / subdir, dst_root / subdir
            if src_subdir.exists():
                shutil.copytree(src_subdir, dst_subdir)
                print(f"[INFO] Copied {subdir}/")
        for video_dir in src_root.iterdir():
            if video_dir.is_dir() and video_dir.name not in ["data", "meta"]:
                link_path = dst_root / video_dir.name
                link_path.symlink_to(video_dir)
                print(f"[INFO] Created symlink for {video_dir.name}/")
        return True
    except Exception as e:
        print(f"[ERROR] Copy failed: {e}")
        return False


# ==================== Multiprocessing Worker Functions ====================

def process_single_parquet_file_internal(
    pf_path_str: str,
    features_dict: dict,
    video_shapes_dict: dict,
    state_encoding: int,
    action_encoding: int
) -> Tuple[str, List[Tuple[int, dict]], bool, Optional[str]]:
    """Internal function to process a single parquet file."""
    pf_path = Path(pf_path_str)
    state_enc = StateEncoding(state_encoding)
    action_enc = ActionEncoding(action_encoding)
    
    try:
        # Step 1: Read parquet file
        df = pl.read_parquet(pf_path)
        
        # Step 2: Format conversion
        modified = False
        new_columns = {}
        
        need_convert_action = "action" in features_dict and action_enc == ActionEncoding.EEF_POS
        need_convert_state = "observation.state" in features_dict and state_enc in [StateEncoding.POS_EULER, StateEncoding.POS_QUAT]
        
        if need_convert_action and "action" in df.columns:
            action_data = np.stack(df["action"].to_list())
            if action_data.shape[1] == 7:
                converted_action = convert_euler_data_batch(action_data)
                new_columns["action"] = pl.Series("action", converted_action.tolist())
                modified = True
        
        if need_convert_state and "observation.state" in df.columns:
            state_data = np.stack(df["observation.state"].to_list())
            if state_enc == StateEncoding.POS_EULER and state_data.shape[1] in [7, 8]:
                converted_state = convert_euler_data_batch(state_data)
                new_columns["observation.state"] = pl.Series("observation.state", converted_state.tolist())
                modified = True
            elif state_enc == StateEncoding.POS_QUAT and state_data.shape[1] == 8:
                converted_state = convert_quat_data_batch(state_data)
                new_columns["observation.state"] = pl.Series("observation.state", converted_state.tolist())
                modified = True
        
        if new_columns:
            df = df.with_columns(list(new_columns.values()))
            df.write_parquet(pf_path)
        
        # Step 3: Compute stats using optimized numpy function
        episode_indices = df["episode_index"].to_numpy()
        unique_episodes = np.unique(episode_indices)
        
        stats = []
        for ep_idx in unique_episodes:
            ep_mask = np.where(episode_indices == ep_idx)[0]
            ep_stats = {}
            
            for key, ft in features_dict.items():
                if ft["dtype"] == "video":
                    c, h, w = video_shapes_dict.get(key, (3, 224, 224))
                    ep_ft_data = np.zeros((1, c, h, w), dtype=np.float32)
                else:
                    if key not in df.columns:
                        continue
                    col_data = df[key]
                    if col_data.dtype == pl.List:
                        # More efficient: convert only needed rows
                        ep_rows = [col_data[i] for i in ep_mask]
                        ep_ft_data = np.array(ep_rows, dtype=np.float32)
                    else:
                        ep_ft_data = col_data.to_numpy()[ep_mask]
                
                # Use optimized numpy stats
                ep_stats[key] = compute_feature_stats_numpy(ep_ft_data, axis=0)
            
            stats.append((int(ep_idx), ep_stats))
        
        # Explicit cleanup
        del df
        del episode_indices
        del unique_episodes
        
        return (pf_path_str, stats, modified, None)
    
    except Exception as e:
        import traceback
        return (pf_path_str, [], False, f"{str(e)}\n{traceback.format_exc()}")


def process_parquet_chunk(args: Tuple) -> List[Tuple[str, List[Tuple[int, dict]], bool, Optional[str]]]:
    """
    Process a chunk of parquet files in a single worker process.
    Each worker processes multiple files to reduce IPC overhead.
    """
    file_list, features_dict, video_shapes_dict, state_encoding_int, action_encoding_int = args
    
    # Suppress warnings in worker process
    warnings.filterwarnings('ignore')
    
    results = []
    for pf_path_str in file_list:
        result = process_single_parquet_file_internal(
            pf_path_str, features_dict, video_shapes_dict, 
            state_encoding_int, action_encoding_int
        )
        results.append(result)
    
    # Force garbage collection after processing chunk
    gc.collect()
    
    return results


class StreamingStatsWriter:
    """
    Context manager for streaming stats writes.
    Writes stats incrementally to avoid memory accumulation.
    """
    def __init__(self, output_path: Path, buffer_size: int = 100):
        self.output_path = output_path
        self.buffer_size = buffer_size
        self.buffer = []
        self.writer = None
    
    def __enter__(self):
        self.output_path.parent.mkdir(exist_ok=True, parents=True)
        self.writer = jsonlines.open(self.output_path, "w")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.buffer:
            self._write_buffer()
        if self.writer:
            self.writer.close()
        return False
    
    def _write_buffer(self):
        """Write buffered stats to disk."""
        for ep_idx, ep_stats in sorted(self.buffer, key=lambda x: x[0]):
            self.writer.write({"episode_index": ep_idx, "stats": ep_stats})
        self.buffer.clear()
    
    def add_stats(self, stats: List[Tuple[int, dict]]):
        """Add stats to buffer and flush if needed."""
        self.buffer.extend(stats)
        if len(self.buffer) >= self.buffer_size:
            self._write_buffer()


def process_parquet_files_mp(
    dataset: LightweightLeRobotDataset,
    state_encoding: StateEncoding,
    action_encoding: ActionEncoding,
    num_workers: int = 8,
    chunksize: int = 4
) -> Tuple[bool, int]:
    """
    Process parquet files using multiprocessing with streaming writes.
    
    Key optimizations:
    1. Streaming writes: Stats are written incrementally, not accumulated
    2. Buffered I/O: Batch writes to reduce I/O overhead
    3. Memory-efficient: Minimal data retention in memory
    
    Args:
        dataset: Dataset to process
        state_encoding: State encoding type
        action_encoding: Action encoding type
        num_workers: Number of worker processes
        chunksize: Number of files per worker task
    
    Returns:
        Tuple of (any_modified, total_episodes_processed)
    """
    # Get parquet files
    data_dir = dataset.root / "data"
    parquet_files = []
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        parquet_files.extend(sorted(chunk_dir.glob("episode_*.parquet")))
    if not parquet_files:
        parquet_files = sorted(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        print("[WARN] No parquet files found")
        return False, 0
    
    print(f"[INFO] Processing {len(parquet_files)} parquet files with {num_workers} worker processes")
    print(f"[INFO] Chunk size: {chunksize} files per task")
    
    # Prepare video shapes
    video_shapes = {}
    for key, ft in dataset.features.items():
        if ft["dtype"] == "video":
            video_shapes[key] = tuple(ft.get("shape", [3, 224, 224]))
    
    # Split files into chunks
    file_chunks = []
    for i in range(0, len(parquet_files), chunksize):
        chunk = [str(pf_path) for pf_path in parquet_files[i:i + chunksize]]
        file_chunks.append(chunk)
    
    # Prepare arguments for worker processes
    args_list = [
        (chunk, dataset.features, video_shapes, int(state_encoding), int(action_encoding))
        for chunk in file_chunks
    ]
    
    any_modified = False
    errors = []
    total_episodes = 0
    
    # Use Pool with streaming writes
    episodes_stats_path = dataset.root / EPISODES_STATS_PATH
    
    with StreamingStatsWriter(episodes_stats_path, buffer_size=200) as writer:
        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(parquet_files), desc="Processing parquet files", smoothing=0.1) as pbar:
                # Use imap for streaming results
                for chunk_results in pool.imap_unordered(process_parquet_chunk, args_list):
                    for pf_path, stats, modified, error in chunk_results:
                        if error:
                            errors.append((pf_path, error))
                        else:
                            total_episodes += len(stats)
                            if modified:
                                any_modified = True
                            # Stream stats to writer
                            writer.add_stats(stats)
                        
                        pbar.update(1)
    
    # Report errors
    if errors:
        print(f"\n[WARN] {len(errors)} files had errors:")
        for pf_path, error in errors[:5]:
            print(f"  - {pf_path}: {error[:100]}...")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    return any_modified, total_episodes


def convert_dataset(dataset_name: str, original_path: str, num_workers: int = 8, 
                    backup: bool = False, output_root: str = None, chunksize: int = 4):
    """Convert a single dataset from v2.0 to v2.1 using multiprocessing.
    
    Args:
        dataset_name: Name of the dataset
        original_path: Path to the source dataset directory
        num_workers: Number of worker processes
        backup: Whether to create a backup before conversion
        output_root: Optional output directory
        chunksize: Number of files per worker task
    """
    src_root = Path(original_path)
    
    if not src_root.exists():
        print(f"[ERROR] {src_root} does not exist")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"Source: {src_root}")
    
    root = Path(output_root) if output_root else src_root
    if output_root:
        print(f"Output: {root}")
        if not copy_dataset_light(src_root, root):
            return
    
    print(f"{'='*60}")
    
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"[ERROR] info.json not found at {info_path}")
        return
    
    with open(info_path, "r") as f:
        info_dict = json.load(f)
    
    current_version = info_dict.get("codebase_version", V20)
    if current_version == V30:
        print(f"[INFO] Found v3.0, treating as v2.0")
    
    if backup and output_root is None:
        backup_root = root.parent / f"{root.name}_backup"
        if not backup_root.exists():
            copy_dataset_light(root, backup_root)
    
    info_dict["codebase_version"] = V20
    with open(info_path, "w") as f:
        json.dump(info_dict, f, indent=4, ensure_ascii=False)
    
    try:
        dataset = LightweightLeRobotDataset(root)
        
        # Remove old stats file if exists
        episodes_stats_path = root / EPISODES_STATS_PATH
        if episodes_stats_path.exists():
            episodes_stats_path.unlink()
        
        state_encoding, action_encoding = get_dataset_encoding(dataset_name)
        print(f"[INFO] State: {state_encoding.name}, Action: {action_encoding.name}")
        
        # Process parquet files with multiprocessing
        any_modified, total_episodes = process_parquet_files_mp(
            dataset, state_encoding, action_encoding, num_workers, chunksize
        )
        
        # Update feature shapes if modified
        if any_modified:
            info = dataset.info
            if "action" in info.get("features", {}) and action_encoding == ActionEncoding.EEF_POS:
                info["features"]["action"]["shape"] = [10]
            if "observation.state" in info.get("features", {}) and state_encoding in [StateEncoding.POS_EULER, StateEncoding.POS_QUAT]:
                info["features"]["observation.state"]["shape"] = [10]
            info_path = root / "meta" / "info.json"
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4, ensure_ascii=False)
            print("[INFO] Updated feature shapes in info.json")
        
        print(f"[INFO] Processed {total_episodes} episodes")
        dataset.update_info({"codebase_version": V21})
        print(f"[DONE] Converted {dataset_name} to v2.1")
        
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        info_dict["codebase_version"] = current_version
        with open(info_path, "w") as f:
            json.dump(info_dict, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a single LeRobot dataset from v2.0 to v2.1 (Optimized Multiprocessing)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--original_path", type=str, required=True, help="Path to the source dataset directory")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--chunksize", type=int, default=4, help="Number of files per worker task (reduces IPC overhead)")
    parser.add_argument("--backup", action="store_true", help="Create a backup before conversion")
    parser.add_argument("--output_root", type=str, default=None, help="Optional output directory")
    
    args = parser.parse_args()
    
    # Determine number of workers
    available_cpus = cpu_count()
    if args.num_workers > available_cpus:
        print(f"[INFO] Limiting workers to {available_cpus} (available CPUs)")
        args.num_workers = available_cpus
    
    convert_dataset(
        dataset_name=args.dataset_name,
        original_path=args.original_path,
        num_workers=args.num_workers,
        backup=args.backup,
        output_root=args.output_root,
        chunksize=args.chunksize
    )