import argparse
import logging
import os
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import jsonlines
from scipy.spatial.transform import Rotation
import warnings

import polars as pl
import torch

from oxe_configs import OXE_DATASET_CONFIGS, StateEncoding as OXEStateEncoding, ActionEncoding as OXEActionEncoding

# Suppress warnings globally
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)


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


V20 = "v2.0"
V21 = "v2.1"
V30 = "v3.0"
EPISODES_STATS_PATH = Path("meta/episodes_stats.jsonl")
EPISODES_PATH = Path("meta/episodes.jsonl")


# ==================== Lightweight Dataset Class using Polars ====================

class LightweightLeRobotDataset:
    """
    A lightweight class to load v2.0/v2.1 LeRobot dataset information directly from files
    without instantiating the full LeRobotDataset from v3.0.
    Uses Polars for efficient data processing.
    """
    
    def __init__(self, root: Path):
        self.root = Path(root)
        self._info = None
        self._features = None
        self._episodes = None
        self._pl_dataframe = None
        self._episode_data_index = None
    
    @property
    def info(self) -> dict:
        """Load info.json if not already loaded."""
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
    
    @property
    def total_frames(self) -> int:
        return self.info.get("total_frames", 0)
    
    def _load_episodes_jsonl(self) -> dict:
        episodes_path = self.root / EPISODES_PATH
        episodes = {}
        if episodes_path.exists():
            with jsonlines.open(episodes_path, "r") as reader:
                for item in reader:
                    ep_idx = item["episode_index"]
                    episodes[ep_idx] = item
        return episodes
    
    @property
    def episodes(self) -> dict:
        if self._episodes is None:
            self._episodes = self._load_episodes_jsonl()
        return self._episodes
    
    def load_pl_dataframe(self) -> pl.DataFrame:
        """Load the data as a Polars DataFrame from parquet files."""
        if self._pl_dataframe is None:
            data_dir = self.root / "data"
            parquet_files = []
            for chunk_dir in sorted(data_dir.glob("chunk-*")):
                parquet_files.extend(sorted(chunk_dir.glob("episode_*.parquet")))
            if not parquet_files:
                parquet_files = sorted(data_dir.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {data_dir}")
            paths = [str(p) for p in parquet_files]
            self._pl_dataframe = pl.read_parquet(paths)
        return self._pl_dataframe
    
    @property
    def pl_dataframe(self) -> pl.DataFrame:
        if self._pl_dataframe is None:
            self._pl_dataframe = self.load_pl_dataframe()
        return self._pl_dataframe
    
    @property
    def episode_data_index(self) -> dict:
        if self._episode_data_index is None:
            df = self.pl_dataframe
            if "episode_index" in df.columns:
                ep_col = df.select("episode_index").to_series()
                from_indices, to_indices = [], []
                current_ep = None
                for idx in range(len(ep_col)):
                    ep_idx = ep_col[idx]
                    if current_ep is None:
                        current_ep = ep_idx
                        from_indices.append(idx)
                    elif ep_idx != current_ep:
                        to_indices.append(idx)
                        from_indices.append(idx)
                        current_ep = ep_idx
                if len(from_indices) > 0:
                    to_indices.append(len(df))
                self._episode_data_index = {"from": torch.tensor(from_indices), "to": torch.tensor(to_indices)}
            else:
                self._episode_data_index = {"from": torch.tensor([]), "to": torch.tensor([])}
        return self._episode_data_index
    
    def update_info(self, updates: dict):
        info = self.info.copy()
        info.update(updates)
        info_path = self.root / "meta" / "info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        self._info = None


# ==================== Stats Functions using Polars ====================

def compute_feature_stats_polars(data_list: list) -> dict:
    """
    Compute all statistics using Polars expressions for efficiency.
    
    Args:
        data_list: List of values (can be scalars or vectors)
    
    Returns:
        Dictionary with min, max, mean, std, count, and quantiles
    """
    if len(data_list) == 0 or data_list[0] is None:
        return {"min": [], "max": [], "mean": [], "std": [], "count": [], "q01": [], "q10": [], "q50": [], "q90": [], "q99": []}
    
    # Convert to numpy array
    data_arr = np.array(data_list)
    
    # Flatten for multi-dimensional data
    if data_arr.ndim > 2:
        data_flat = data_arr.reshape(data_arr.shape[0], -1)
    elif data_arr.ndim == 1:
        data_flat = data_arr.reshape(-1, 1)
    else:
        data_flat = data_arr
    
    n_dims = data_flat.shape[1]
    
    # Build polars DataFrame with each dimension as column
    columns = {f"dim_{i}": data_flat[:, i] for i in range(n_dims)}
    stats_df = pl.DataFrame(columns)
    
    # Compute statistics using polars with alias (compatible with older versions)
    min_vals = stats_df.select([pl.col(f"dim_{i}").min().alias(f"min_{i}") for i in range(n_dims)])
    max_vals = stats_df.select([pl.col(f"dim_{i}").max().alias(f"max_{i}") for i in range(n_dims)])
    mean_vals = stats_df.select([pl.col(f"dim_{i}").mean().alias(f"mean_{i}") for i in range(n_dims)])
    std_vals = stats_df.select([pl.col(f"dim_{i}").std().alias(f"std_{i}") for i in range(n_dims)])
    q01_vals = stats_df.select([pl.col(f"dim_{i}").quantile(0.01).alias(f"q01_{i}") for i in range(n_dims)])
    q10_vals = stats_df.select([pl.col(f"dim_{i}").quantile(0.10).alias(f"q10_{i}") for i in range(n_dims)])
    q50_vals = stats_df.select([pl.col(f"dim_{i}").quantile(0.50).alias(f"q50_{i}") for i in range(n_dims)])
    q90_vals = stats_df.select([pl.col(f"dim_{i}").quantile(0.90).alias(f"q90_{i}") for i in range(n_dims)])
    q99_vals = stats_df.select([pl.col(f"dim_{i}").quantile(0.99).alias(f"q99_{i}") for i in range(n_dims)])
    
    return {
        "min": [min_vals[f"min_{i}"][0] for i in range(n_dims)],
        "max": [max_vals[f"max_{i}"][0] for i in range(n_dims)],
        "mean": [mean_vals[f"mean_{i}"][0] for i in range(n_dims)],
        "std": [std_vals[f"std_{i}"][0] for i in range(n_dims)],
        "count": [int(data_flat.shape[0])],
        "q01": [q01_vals[f"q01_{i}"][0] for i in range(n_dims)],
        "q10": [q10_vals[f"q10_{i}"][0] for i in range(n_dims)],
        "q50": [q50_vals[f"q50_{i}"][0] for i in range(n_dims)],
        "q90": [q90_vals[f"q90_{i}"][0] for i in range(n_dims)],
        "q99": [q99_vals[f"q99_{i}"][0] for i in range(n_dims)],
    }


def compute_feature_stats_fast(data, axis: int = 0) -> dict:
    """Compute statistics - accepts polars Series, numpy array, or list."""
    if data is None or (hasattr(data, '__len__') and len(data) == 0):
        return {"min": [], "max": [], "mean": [], "std": [], "count": [], "q01": [], "q10": [], "q50": [], "q90": [], "q99": []}
    
    # Polars Series
    if isinstance(data, pl.Series):
        if data.dtype == pl.List:
            return compute_feature_stats_polars(data.to_list())
        else:
            data_list = data.to_list()
            return compute_feature_stats_polars([[x] if not isinstance(x, list) else x for x in data_list])
    
    # Numpy array
    if isinstance(data, np.ndarray):
        if data.size == 0:
            return {"min": [], "max": [], "mean": [], "std": [], "count": [], "q01": [], "q10": [], "q50": [], "q90": [], "q99": []}
        return compute_feature_stats_polars(data.tolist() if data.ndim == 1 else data.tolist())
    
    # List
    if isinstance(data, list):
        return compute_feature_stats_polars(data)
    
    return {"min": [], "max": [], "mean": [], "std": [], "count": [], "q01": [], "q10": [], "q50": [], "q90": [], "q99": []}


def process_single_episode(args: Tuple) -> Tuple[int, dict]:
    """Process a single episode for multiprocessing."""
    ep_idx, ep_data_dict, features, video_shapes = args
    ep_stats = {}
    
    for key, ft in features.items():
        if ft["dtype"] == "video":
            c, h, w = video_shapes.get(key, (3, 224, 224))
            ep_ft_data = np.zeros((1, c, h, w), dtype=np.float32)
        else:
            ep_ft_data = ep_data_dict[key]
        ep_stats[key] = compute_feature_stats_fast(ep_ft_data, axis=0)
    
    return (ep_idx, ep_stats)


# ==================== Data Conversion Functions ====================

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


def convert_parquet_data_fast(dataset: LightweightLeRobotDataset, state_encoding: StateEncoding, action_encoding: ActionEncoding):
    """Optimized parquet data conversion using Polars."""
    state_key, action_key = "observation.state", "action"
    has_state = state_key in dataset.features
    has_action = action_key in dataset.features
    
    if not has_state and not has_action:
        print("[INFO] No action or state to convert")
        return False
    
    data_dir = dataset.root / "data"
    parquet_files = []
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        parquet_files.extend(sorted(chunk_dir.glob("episode_*.parquet")))
    if not parquet_files:
        parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print("[WARN] No parquet files found")
        return False
    
    print(f"[INFO] Converting {len(parquet_files)} parquet files...")
    
    need_convert_action = has_action and action_encoding == ActionEncoding.EEF_POS
    need_convert_state = has_state and state_encoding in [StateEncoding.POS_EULER, StateEncoding.POS_QUAT]
    
    if not need_convert_action and not need_convert_state:
        print("[INFO] No conversion needed")
        return False
    
    modified = False
    
    for pf_path in tqdm(parquet_files, desc="Converting parquet"):
        df = pl.read_parquet(pf_path)
        new_columns = {}
        
        if need_convert_action and action_key in df.columns:
            action_data = np.stack(df[action_key].to_list())
            if action_data.shape[1] == 7:
                converted_action = convert_euler_data_batch(action_data)
                new_columns[action_key] = pl.Series(action_key, converted_action.tolist())
                modified = True
        
        if need_convert_state and state_key in df.columns:
            state_data = np.stack(df[state_key].to_list())
            if state_encoding == StateEncoding.POS_EULER and state_data.shape[1] in [7, 8]:
                converted_state = convert_euler_data_batch(state_data)
                new_columns[state_key] = pl.Series(state_key, converted_state.tolist())
                modified = True
            elif state_encoding == StateEncoding.POS_QUAT and state_data.shape[1] == 8:
                converted_state = convert_quat_data_batch(state_data)
                new_columns[state_key] = pl.Series(state_key, converted_state.tolist())
                modified = True
        
        if new_columns:
            df = df.with_columns(list(new_columns.values()))
            df.write_parquet(pf_path)
    
    if modified:
        info = dataset.info
        if has_action and action_key in info.get("features", {}):
            if action_encoding == ActionEncoding.EEF_POS:
                info["features"][action_key]["shape"] = [10]
        if has_state and state_key in info.get("features", {}):
            if state_encoding in [StateEncoding.POS_EULER, StateEncoding.POS_QUAT]:
                info["features"][state_key]["shape"] = [10]
        info_path = dataset.root / "meta" / "info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        print("[INFO] Updated feature shapes in info.json")
    
    return modified


def compute_stats_sequential(dataset: LightweightLeRobotDataset):
    """Compute episode statistics sequentially (avoids multiprocessing data transfer overhead)."""
    print("Computing episodes stats...")
    total_episodes = dataset.total_episodes
    
    video_shapes = {}
    for key, ft in dataset.features.items():
        if ft["dtype"] == "video":
            video_shapes[key] = tuple(ft.get("shape", [3, 224, 224]))
    
    episode_indices = []
    for ep_idx in range(total_episodes):
        ep_start = dataset.episode_data_index["from"][ep_idx]
        ep_end = dataset.episode_data_index["to"][ep_idx]
        episode_indices.append((ep_start.item() if hasattr(ep_start, 'item') else ep_start,
                                ep_end.item() if hasattr(ep_end, 'item') else ep_end))
    
    print("Loading episode data...")
    df = dataset.pl_dataframe
    
    results = []
    for ep_idx in tqdm(range(total_episodes), desc="Computing stats"):
        ep_start, ep_end = episode_indices[ep_idx]
        ep_df = df.slice(ep_start, ep_end - ep_start)
        
        ep_data_dict = {}
        for key, ft in dataset.features.items():
            if ft["dtype"] != "video":
                col_data = ep_df[key]
                if col_data.dtype == pl.List:
                    ep_data_dict[key] = np.array(col_data.to_list())
                else:
                    ep_data_dict[key] = col_data.to_numpy()
        
        # Compute stats for this episode
        ep_stats = {}
        for key, ft in dataset.features.items():
            if ft["dtype"] == "video":
                c, h, w = video_shapes.get(key, (3, 224, 224))
                ep_ft_data = np.zeros((1, c, h, w), dtype=np.float32)
            else:
                ep_ft_data = ep_data_dict[key]
            ep_stats[key] = compute_feature_stats_fast(ep_ft_data, axis=0)
        
        results.append((ep_idx, ep_stats))
    
    print("Writing episode stats...")
    episodes_stats_path = dataset.root / EPISODES_STATS_PATH
    episodes_stats_path.parent.mkdir(exist_ok=True, parents=True)
    
    results.sort(key=lambda x: x[0])
    with jsonlines.open(episodes_stats_path, "w") as writer:
        for ep_idx, ep_stats in results:
            writer.write({"episode_index": ep_idx, "stats": ep_stats})
    
    print(f"[INFO] Wrote stats for {len(results)} episodes")


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


def convert_dataset(dataset_name: str, original_path: str, num_workers: int = 8, backup: bool = False, output_root: str = None):
    """Convert a single dataset from v2.0 to v2.1.
    
    Args:
        dataset_name: Name of the dataset (used to determine encoding from oxe_configs)
        original_path: Path to the source dataset directory
        num_workers: Number of multiprocessing workers for stats computation
        backup: Whether to create a backup before conversion
        output_root: Optional output directory. If specified, data will be copied here.
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
        
        episodes_stats_path = root / EPISODES_STATS_PATH
        if episodes_stats_path.exists():
            episodes_stats_path.unlink()
        
        state_encoding, action_encoding = get_dataset_encoding(dataset_name)
        print(f"[INFO] State: {state_encoding.name}, Action: {action_encoding.name}")
        
        convert_parquet_data_fast(dataset, state_encoding, action_encoding)
        compute_stats_sequential(dataset)
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
    parser = argparse.ArgumentParser(description="Convert a single LeRobot dataset from v2.0 to v2.1 (Polars)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (used to determine encoding from oxe_configs)")
    parser.add_argument("--original_path", type=str, required=True, help="Path to the source dataset directory")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of multiprocessing workers")
    parser.add_argument("--backup", action="store_true", help="Create a backup before conversion")
    parser.add_argument("--output_root", type=str, default=None, help="Optional output directory. If specified, data will be copied here")
    
    args = parser.parse_args()
    available_cpus = cpu_count()
    if args.num_workers > available_cpus:
        print(f"[INFO] Limiting workers to {available_cpus} (available CPUs)")
        args.num_workers = available_cpus
    
    # os.makedirs(args.output_root, exist_ok=True)
    convert_dataset(
        dataset_name=args.dataset_name,
        original_path=args.original_path,
        num_workers=args.num_workers,
        backup=args.backup,
        output_root=args.output_root
    )
