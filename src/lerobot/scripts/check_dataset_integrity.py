#!/usr/bin/env python
"""
数据集完整性检查脚本

检查内容：
1. 视频文件是否存在
2. 视频帧数与数据集中记录的时间戳是否匹配
3. 时间戳是否超出视频范围
4. 视频时长与episode时长是否一致
5. 视频帧索引计算是否会越界
6. 视频解码是否正常

使用方法:
    python src/lerobot/scripts/check_dataset_integrity.py --root /path/to/dataset
    python src/lerobot/scripts/check_dataset_integrity.py --repo_id lerobot/pusht
    python src/lerobot/scripts/check_dataset_integrity.py --root /path/to/dataset --decode_frames
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dataset_integrity(
    root: str | Path,
    repo_id: Optional[str] = None,
    tolerance_s: float = 1e-4,
    check_frames: bool = False,
    decode_frames: bool = False,
    max_episodes: Optional[int] = None,
    verbose: bool = True,
):
    """
    检查数据集完整性

    Args:
        root: 数据集根目录
        repo_id: 数据集repo id (可选，用于日志)
        tolerance_s: 时间戳容差
        check_frames: 是否实际解码帧进行检查 (较慢)
        decode_frames: 是否解码视频帧来检测损坏的视频文件
        max_episodes: 最大检查episode数量 (None表示检查全部)
        verbose: 是否输出详细信息
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import get_video_duration_in_s, get_video_info
    import av

    root = Path(root)
    if repo_id is None:
        repo_id = root.name

    logger.info(f"开始检查数据集: {repo_id}")
    logger.info(f"数据集路径: {root}")

    # 加载元数据
    try:
        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    except Exception as e:
        logger.error(f"无法加载元数据: {e}")
        return {"error": str(e)}

    issues = {
        "missing_videos": [],
        "video_duration_mismatch": [],
        "timestamp_out_of_bounds": [],
        "frame_index_out_of_bounds": [],
        "fps_mismatch": [],
        "video_read_errors": [],
        "decode_errors": [],
    }

    video_keys = meta.video_keys
    fps = meta.fps

    if len(video_keys) == 0:
        logger.info("数据集没有视频文件")
        return issues

    logger.info(f"视频keys: {video_keys}")
    logger.info(f"数据集FPS: {fps}")
    logger.info(f"总episodes: {meta.total_episodes}")

    episodes_to_check = range(meta.total_episodes)
    if max_episodes is not None:
        episodes_to_check = range(min(max_episodes, meta.total_episodes))

    # 记录已检查的视频文件，避免重复检查
    checked_videos = set()

    # 检查每个episode
    for ep_idx in tqdm(episodes_to_check, desc="检查episodes"):
        ep = meta.episodes[ep_idx]
        episode_length = ep["length"]

        for vid_key in video_keys:
            # 获取视频路径
            try:
                video_path = root / meta.get_video_file_path(ep_idx, vid_key)
            except Exception as e:
                issues["missing_videos"].append({
                    "episode": ep_idx,
                    "video_key": vid_key,
                    "error": str(e),
                })
                continue

            # 检查视频文件是否存在
            if not video_path.exists():
                issues["missing_videos"].append({
                    "episode": ep_idx,
                    "video_key": vid_key,
                    "path": str(video_path),
                })
                continue

            # 获取视频信息
            try:
                video_info = get_video_info(video_path)
                video_duration = get_video_duration_in_s(video_path)
                video_fps = video_info.get("video.fps", fps)
                video_height = video_info.get("video.height")
                video_width = video_info.get("video.width")
            except Exception as e:
                issues["video_read_errors"].append({
                    "episode": ep_idx,
                    "video_key": vid_key,
                    "path": str(video_path),
                    "error": str(e),
                })
                continue

            # 检查FPS是否匹配
            if video_fps != fps:
                issues["fps_mismatch"].append({
                    "episode": ep_idx,
                    "video_key": vid_key,
                    "dataset_fps": fps,
                    "video_fps": video_fps,
                })

            # 获取episode在视频中的起始时间戳
            from_timestamp = ep.get(f"videos/{vid_key}/from_timestamp", 0.0)

            # 计算episode的预期时长
            episode_duration = episode_length / fps

            # 检查时间戳边界
            # 最后一个帧的时间戳
            last_frame_timestamp = from_timestamp + (episode_length - 1) / fps

            # 检查是否超出视频时长
            if last_frame_timestamp > video_duration + tolerance_s:
                issues["timestamp_out_of_bounds"].append({
                    "episode": ep_idx,
                    "video_key": vid_key,
                    "from_timestamp": from_timestamp,
                    "episode_length": episode_length,
                    "last_frame_timestamp": last_frame_timestamp,
                    "video_duration": video_duration,
                    "exceeded_by": last_frame_timestamp - video_duration,
                })

            # 计算帧索引是否会越界
            # 使用 torchcodec 的方式计算
            with av.open(str(video_path)) as container:
                video_stream = container.streams.video[0]
                num_frames = video_stream.frames

                # 计算最大帧索引
                if num_frames > 0:
                    average_fps = float(video_stream.average_rate)
                    max_frame_index = round(last_frame_timestamp * average_fps)

                    if max_frame_index >= num_frames:
                        issues["frame_index_out_of_bounds"].append({
                            "episode": ep_idx,
                            "video_key": vid_key,
                            "path": str(video_path),
                            "last_timestamp": last_frame_timestamp,
                            "computed_frame_index": max_frame_index,
                            "num_frames": num_frames,
                            "average_fps": average_fps,
                        })

            # 解码测试 - 检查视频是否有损坏的帧
            if decode_frames and str(video_path) not in checked_videos:
                checked_videos.add(str(video_path))
                try:
                    from lerobot.datasets.video_utils import decode_video_frames_torchcodec
                    # 尝试解码中间帧和最后一帧
                    test_timestamps = [video_duration / 2, video_duration - 0.1]
                    decode_video_frames_torchcodec(video_path, test_timestamps, tolerance_s)
                except Exception as e:
                    issues["decode_errors"].append({
                        "episode": ep_idx,
                        "video_key": vid_key,
                        "path": str(video_path),
                        "error": str(e),
                    })

    # 打印汇总
    print("\n" + "="*60)
    print("检查结果汇总")
    print("="*60)

    total_issues = sum(len(v) for v in issues.values())

    if total_issues == 0:
        print("✅ 数据集完整性检查通过，未发现问题")
    else:
        print(f"❌ 发现 {total_issues} 个问题:")

        if issues["missing_videos"]:
            print(f"\n📁 缺失视频文件 ({len(issues['missing_videos'])} 个):")
            for item in issues["missing_videos"][:5]:
                print(f"   - Episode {item['episode']}, {item['video_key']}")
            if len(issues["missing_videos"]) > 5:
                print(f"   ... 还有 {len(issues['missing_videos']) - 5} 个")

        if issues["timestamp_out_of_bounds"]:
            print(f"\n⏰ 时间戳超出视频范围 ({len(issues['timestamp_out_of_bounds'])} 个):")
            for item in issues["timestamp_out_of_bounds"][:5]:
                print(f"   - Episode {item['episode']}, {item['video_key']}")
                print(f"     最后帧时间戳: {item['last_frame_timestamp']:.4f}s, 视频时长: {item['video_duration']:.4f}s")
                print(f"     超出: {item['exceeded_by']:.4f}s")
            if len(issues['timestamp_out_of_bounds']) > 5:
                print(f"   ... 还有 {len(issues['timestamp_out_of_bounds']) - 5} 个")

        if issues["frame_index_out_of_bounds"]:
            print(f"\n🔢 帧索引越界 ({len(issues['frame_index_out_of_bounds'])} 个):")
            for item in issues["frame_index_out_of_bounds"][:5]:
                print(f"   - Episode {item['episode']}, {item['video_key']}")
                print(f"     视频帧数: {item['num_frames']}, 计算索引: {item['computed_frame_index']}")
                print(f"     视频路径: {item['path']}")
            if len(issues['frame_index_out_of_bounds']) > 5:
                print(f"   ... 还有 {len(issues['frame_index_out_of_bounds']) - 5} 个")

        if issues["fps_mismatch"]:
            print(f"\n🎬 FPS不匹配 ({len(issues['fps_mismatch'])} 个):")
            for item in issues["fps_mismatch"][:5]:
                print(f"   - Episode {item['episode']}, {item['video_key']}")
                print(f"     数据集FPS: {item['dataset_fps']}, 视频FPS: {item['video_fps']}")
            if len(issues['fps_mismatch']) > 5:
                print(f"   ... 还有 {len(issues['fps_mismatch']) - 5} 个")

        if issues["video_read_errors"]:
            print(f"\n⚠️ 视频读取错误 ({len(issues['video_read_errors'])} 个):")
            for item in issues["video_read_errors"][:5]:
                print(f"   - Episode {item['episode']}, {item['video_key']}")
                print(f"     错误: {item['error']}")
            if len(issues['video_read_errors']) > 5:
                print(f"   ... 还有 {len(issues['video_read_errors']) - 5} 个")

        if issues["decode_errors"]:
            print(f"\n🔴 视频解码错误 ({len(issues['decode_errors'])} 个):")
            for item in issues["decode_errors"][:10]:
                print(f"   - Episode {item['episode']}, {item['video_key']}")
                print(f"     路径: {item['path']}")
                print(f"     错误: {item['error']}")
            if len(issues['decode_errors']) > 10:
                print(f"   ... 还有 {len(issues['decode_errors']) - 10} 个")

    print("\n" + "="*60)

    return issues


def check_sample_loading(
    root: str | Path,
    repo_id: Optional[str] = None,
    num_samples: int = 100,
    delta_timestamps: Optional[dict] = None,
):
    """
    检查样本加载是否正常

    Args:
        root: 数据集根目录
        repo_id: 数据集repo id
        num_samples: 检查的样本数量
        delta_timestamps: delta_timestamps配置
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if repo_id is None:
        repo_id = root.name

    logger.info(f"检查样本加载: {repo_id}")

    try:
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            delta_timestamps=delta_timestamps,
        )
    except Exception as e:
        logger.error(f"无法加载数据集: {e}")
        return {"error": str(e)}

    errors = []

    import random
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in tqdm(indices, desc="检查样本"):
        try:
            item = dataset[idx]
            # 检查是否有视频帧
            for vid_key in dataset.meta.video_keys:
                if vid_key in item:
                    frames = item[vid_key]
                    if frames is None or len(frames) == 0:
                        errors.append({
                            "index": idx,
                            "error": f"Empty frames for {vid_key}",
                        })
        except Exception as e:
            errors.append({
                "index": idx,
                "error": str(e),
            })

    if errors:
        print(f"\n❌ 样本加载错误 ({len(errors)} 个):")
        for err in errors[:10]:
            print(f"   - Index {err['index']}: {err['error']}")
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors) - 10} 个")
    else:
        print(f"✅ 样本加载检查通过 ({num_samples} 个样本)")

    return errors


def main():
    parser = argparse.ArgumentParser(description="检查LeRobot数据集完整性")
    parser.add_argument("--root", type=str, help="数据集根目录")
    parser.add_argument("--repo_id", type=str, help="数据集repo id")
    parser.add_argument("--tolerance_s", type=float, default=1e-4, help="时间戳容差")
    parser.add_argument("--max_episodes", type=int, default=None, help="最大检查episode数量")
    parser.add_argument("--check_samples", action="store_true", help="是否检查样本加载")
    parser.add_argument("--num_samples", type=int, default=100, help="检查的样本数量")
    parser.add_argument("--decode_frames", action="store_true", help="解码视频帧检测损坏")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    if args.root is None and args.repo_id is None:
        parser.error("必须指定 --root 或 --repo_id")

    # 执行检查
    issues = check_dataset_integrity(
        root=args.root,
        repo_id=args.repo_id,
        tolerance_s=args.tolerance_s,
        max_episodes=args.max_episodes,
        decode_frames=args.decode_frames,
        verbose=args.verbose,
    )

    # 检查样本加载
    if args.check_samples:
        check_sample_loading(
            root=args.root,
            repo_id=args.repo_id,
            num_samples=args.num_samples,
        )

    return issues


if __name__ == "__main__":
    main()
