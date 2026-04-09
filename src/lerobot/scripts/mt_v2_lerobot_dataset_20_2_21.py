#!/usr/bin/env python3
"""
High-performance version of lerobot_dataset_20_2_21.py
Optimized for maximum throughput while maintaining low memory usage.

Key optimizations:
- Worker initialization: share data across workers via initializer (avoids repeated IPC)
- Single quantile call: compute all quantiles at once
- Pre-extract columns: convert all data once per file
- Batch writes: write stats in batches to balance memory and I/O
- Minimal IPC: return only essential data
"""

import argparse
import logging
import os
import time

os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import json

import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count, set_start_method
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import jsonlines
from scipy.spatial.transform import Rotation
import warnings
import polars as pl
import traceback

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

# Quantiles to compute - computed once
QUANTILES = np.array([0.01, 0.10, 0.50, 0.90, 0.99])


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


# ==================== Highly Optimized Stats Function ====================

def compute_stats_fast(data: np.ndarray) -> dict:
    """
    Compute all statistics in minimal operations.
    Uses single quantile call for all quantiles.
    """
    if data is None or data.size == 0:
        return {"min": [], "max": [], "mean": [], "std": [], "count": [], 
                "q01": [], "q10": [], "q50": [], "q90": [], "q99": []}
    
    # Flatten if needed
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    elif data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_samples = data.shape[0]
    
    # Compute all quantiles in one call (most expensive operation)
    quantile_results = np.nanquantile(data, QUANTILES, axis=0)
    
    return {
        "min": np.nanmin(data, axis=0).tolist(),
        "max": np.nanmax(data, axis=0).tolist(),
        "mean": np.nanmean(data, axis=0).tolist(),
        "std": np.nanstd(data, axis=0, ddof=1).tolist(),
        "count": [int(n_samples)],
        "q01": quantile_results[0].tolist(),
        "q10": quantile_results[1].tolist(),
        "q50": quantile_results[2].tolist(),
        "q90": quantile_results[3].tolist(),
        "q99": quantile_results[4].tolist(),
    }


# ==================== Lightweight Dataset Class ====================

class LightweightLeRobotDataset:
    """Lightweight dataset class for loading v2.0/v2.1 LeRobot datasets."""
    
    __slots__ = ['root', '_info', '_features']
    
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


# ==================== High Performance Worker Functions ====================

# Global variables for worker processes (set once at initialization)
_worker_features = None
_worker_video_shapes = None
_worker_state_enc = None
_worker_action_enc = None


def init_worker(features_dict, video_shapes_dict, state_encoding, action_encoding):
    """Initialize worker process with shared data to avoid repeated serialization."""
    global _worker_features, _worker_video_shapes, _worker_state_enc, _worker_action_enc
    _worker_features = features_dict
    _worker_video_shapes = video_shapes_dict
    _worker_state_enc = state_encoding
    _worker_action_enc = action_encoding


def process_parquet_file_fast(pf_path_str: str) -> Tuple[str, List, bool, Optional[str]]:
    """
    Process a single parquet file with maximum efficiency.
    Uses pre-initialized worker state to avoid repeated data transfer.
    """
    pf_path = Path(pf_path_str)
    
    try:
        # Read parquet
        df = pl.read_parquet(pf_path)
        
        # ========== Format conversion ==========
        modified = False
        need_convert_action = "action" in _worker_features and _worker_action_enc == ActionEncoding.EEF_POS
        need_convert_state = "observation.state" in _worker_features and _worker_state_enc in [StateEncoding.POS_EULER, StateEncoding.POS_QUAT]
        
        new_columns = []
        
        if need_convert_action and "action" in df.columns:
            action_data = np.stack(df["action"].to_list())
            if action_data.shape[1] == 7:
                converted_action = convert_euler_data_batch(action_data)
                new_columns.append(pl.Series("action", converted_action.tolist()))
                modified = True
        
        if need_convert_state and "observation.state" in df.columns:
            state_data = np.stack(df["observation.state"].to_list())
            if _worker_state_enc == StateEncoding.POS_EULER and state_data.shape[1] in [7, 8]:
                converted_state = convert_euler_data_batch(state_data)
                new_columns.append(pl.Series("observation.state", converted_state.tolist()))
                modified = True
            elif _worker_state_enc == StateEncoding.POS_QUAT and state_data.shape[1] == 8:
                converted_state = convert_quat_data_batch(state_data)
                new_columns.append(pl.Series("observation.state", converted_state.tolist()))
                modified = True
        
        
        if new_columns:
            df = df.with_columns(new_columns)
            df.write_parquet(pf_path)
        
        
        # ========== Fast stats computation ==========
        # Get episode indices and create groups in one pass
        episode_indices = df["episode_index"].to_numpy()
        unique_episodes, inverse_indices = np.unique(episode_indices, return_inverse=True)
        
        # Pre-extract all column data once
        col_data_cache = {}
        for key, ft in _worker_features.items():
            if ft["dtype"] != "video" and key in df.columns:
                col = df[key]
                if col.dtype == pl.List:
                    col_data_cache[key] = np.array(col.to_list(), dtype=np.float32)
                else:
                    col_data_cache[key] = col.to_numpy()
        
        stats = []
        for i, ep_idx in enumerate(unique_episodes):
            ep_mask = (inverse_indices == i)
            ep_stats = {}
            
            for key, ft in _worker_features.items():
                if ft["dtype"] == "video":
                    # c, h, w = _worker_video_shapes.get(key, (3, 224, 224))
                    ep_ft_data = np.zeros((1, 1), dtype=np.float32)
                    continue
                else:
                    if key not in col_data_cache:
                        continue
                    ep_ft_data = col_data_cache[key][ep_mask]
                
                ep_stats[key] = compute_stats_fast(ep_ft_data)
            
            stats.append((int(ep_idx), ep_stats))
        

        return (pf_path_str, stats, modified, None)
    
    except Exception as e:
        
        return (pf_path_str, [], False, f"{str(e)}\n{traceback.format_exc()}")


def _write_stats_batch(output_path: Path, stats: List[Tuple[int, dict]], append: bool = False):
    """Write a batch of stats to disk."""
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    mode = "a" if append else "w"
    with jsonlines.open(output_path, mode) as writer:
        for ep_idx, ep_stats in stats:
            writer.write({"episode_index": ep_idx, "stats": ep_stats})


def process_parquet_files_mp(
    dataset: LightweightLeRobotDataset,
    state_encoding: StateEncoding,
    action_encoding: ActionEncoding,
    num_workers: int = 8,
    chunksize: int = 4
) -> Tuple[bool, int]:
    """
    Process parquet files with maximum throughput.
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
    
    print(f"[INFO] Processing {len(parquet_files)} parquet files with {num_workers} workers")
    
    # Prepare video shapes
    video_shapes = {}
    for key, ft in dataset.features.items():
        if ft["dtype"] == "video":
            # video_shapes[key] = tuple(ft.get("shape", [3, 224, 224]))
            continue
    
    # Prepare init data for workers
    init_args = (dataset.features, video_shapes, int(state_encoding), int(action_encoding))
    
    any_modified = False
    errors = []
    total_episodes = 0
    all_stats = []
    episodes_stats_path = dataset.root / EPISODES_STATS_PATH
    first_write = True
    
    # Use Pool with initializer to share data across workers
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=init_args,
        maxtasksperchild=64
    ) as pool:
        # Process files with imap for streaming
        file_paths = [str(f) for f in parquet_files]
        results = pool.imap_unordered(process_parquet_file_fast, file_paths, chunksize=chunksize)
        
        with tqdm(total=len(parquet_files), desc="Processing", smoothing=0.05) as pbar:
            for pf_path, stats, modified, error in results:
                pbar.update(1)
                
                if error:
                    errors.append((pf_path, error))
                else:
                    total_episodes += len(stats)
                    if modified:
                        any_modified = True
                    all_stats.extend(stats)
                    
                    # Write periodically to avoid memory buildup
                    if len(all_stats) >= 500:
                        all_stats.sort(key=lambda x: x[0])
                        _write_stats_batch(episodes_stats_path, all_stats, append=not first_write)
                        first_write = False
                        all_stats = []
    
    # Write remaining stats
    if all_stats:
        all_stats.sort(key=lambda x: x[0])
        _write_stats_batch(episodes_stats_path, all_stats, append=not first_write)
    
    if errors:
        print(f"\n[WARN] {len(errors)} errors:")
        for pf_path, error in errors[:3]:
            print(f"  - {pf_path}: {error[:80]}...")
    
    return any_modified, total_episodes


def convert_dataset(dataset_name: str, original_path: str, num_workers: int = 8, 
                    backup: bool = False, output_root: str = None, chunksize: int = 4):
    """Convert a single dataset from v2.0 to v2.1."""
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
        traceback.print_exc()
        info_dict["codebase_version"] = current_version
        with open(info_path, "w") as f:
            json.dump(info_dict, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Convert LeRobot dataset v2.0 to v2.1 (High Performance)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--original_path", type=str, required=True, help="Source path")
    parser.add_argument("--num-workers", type=int, default=16, help="Worker count")
    parser.add_argument("--chunksize", type=int, default=32, help="Chunk size per task")
    parser.add_argument("--backup", action="store_true", help="Create backup")
    parser.add_argument("--output_root", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    available_cpus = cpu_count()
    if args.num_workers > available_cpus:
        print(f"[INFO] Limiting workers to {available_cpus}")
        args.num_workers = available_cpus
    
    convert_dataset(
        dataset_name=args.dataset_name,
        original_path=args.original_path,
        num_workers=args.num_workers,
        backup=args.backup,
        output_root=args.output_root,
        chunksize=args.chunksize
    )