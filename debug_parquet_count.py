"""Diagnostic script: verify episode count in parquet files vs info.json.

Only reads parquet metadata + episode columns. No video decode, no OOM risk.
"""
import json
import polars as pl
from pathlib import Path

ROOT = Path("/data_6t_1/lerobot-v30/merged_0412_v1")

# 1. info.json
info = json.load(open(ROOT / "meta" / "info.json"))
print(f"[info.json] total_episodes = {info['total_episodes']}")
print(f"[info.json] total_frames   = {info.get('total_frames', 'N/A')}")

# 2. Scan episode parquets
episodes_dir = ROOT / "meta" / "episodes"
parquet_files = sorted(episodes_dir.glob("*/*.parquet"))
print(f"\n[episodes dir] parquet file count = {len(parquet_files)}")

# 2a. Per-file breakdown
total_rows_all = 0
total_rows_non_stats = 0
ep_idx_set = set()
for pf in parquet_files:
    df = pl.scan_parquet(str(pf)).collect()
    total_rows_all += df.height
    non_stats = [c for c in df.columns if not c.startswith("stats/")]
    if non_stats:
        total_rows_non_stats += df.height
        if "episode_index" in non_stats:
            ep_idx_set.update(df["episode_index"].to_list())
    print(f"  {pf.name}: {df.height} rows, non_stats_cols={len(non_stats)}")

print(f"\nTotal rows (all columns):  {total_rows_all}")
print(f"Total rows (non-stats):    {total_rows_non_stats}")
print(f"Unique episode_index:      {len(ep_idx_set)}")

# 3. Data parquet verification (scan only episode_index column)
data_dir = ROOT / "data"
data_files = sorted(data_dir.glob("*/*.parquet"))
print(f"\n[data dir] parquet file count = {len(data_files)}")

total_data_rows = 0
data_ep_idx_set = set()
for df in pl.scan_parquet([str(f) for f in data_files]).select("episode_index").collect().partition_by("episode_index", as_dict=True):
    pass

# More efficient: just count unique episode_index
lazy = pl.scan_parquet([str(f) for f in data_files]).select("episode_index")
result = lazy.collect()
total_data_rows = result.height
data_ep_idx_set = set(result["episode_index"].to_list())
print(f"Total data rows:           {total_data_rows}")
print(f"Unique episode_index:      {len(data_ep_idx_set)}")
print(f"Episode index range:       [{min(data_ep_idx_set)} - {max(data_ep_idx_set)}]")

# 4. Summary
print(f"\n{'='*50}")
print(f"BUG ANALYSIS:")
print(f"  info.json total_episodes:    {info['total_episodes']}")
print(f"  Episodes in parquet (non-stats only): {total_rows_non_stats}")
print(f"  Episodes in parquet (all rows):       {total_rows_all}")
print(f"  Unique episode_index in data:         {len(data_ep_idx_set)}")
print(f"  MISSING episodes: {info['total_episodes'] - total_rows_non_stats}")
print(f"  Root cause: Files 000-045 only have stats/* columns.")
print(f"  The non_stats filter silently drops them all.")
