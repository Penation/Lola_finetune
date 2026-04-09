#!/usr/bin/env python
"""
测试视频解码在多进程环境下的稳定性
"""

import argparse
import multiprocessing as mp
from pathlib import Path


def decode_video_worker(video_path: str, frame_indices: list[int], worker_id: int):
    """Worker function to decode video frames."""
    import fsspec
    from torchcodec.decoders import VideoDecoder

    try:
        # 每个worker创建自己的decoder
        file_handle = fsspec.open(video_path).__enter__()
        decoder = VideoDecoder(file_handle, seek_mode="approximate")

        # 解码指定的帧
        frames = decoder.get_frames_at(indices=frame_indices)

        file_handle.close()
        return {"worker_id": worker_id, "status": "success", "num_frames": len(frames.data)}
    except Exception as e:
        return {"worker_id": worker_id, "status": "error", "error": str(e)}


def test_multiprocess_decode(video_path: str, num_workers: int = 4):
    """Test video decoding with multiple processes."""
    import fsspec
    from torchcodec.decoders import VideoDecoder

    # 首先获取视频信息
    file_handle = fsspec.open(video_path).__enter__()
    decoder = VideoDecoder(file_handle, seek_mode="approximate")
    num_frames = decoder.metadata.num_frames
    file_handle.close()

    print(f"Video: {video_path}")
    print(f"Total frames: {num_frames}")
    print(f"Testing with {num_workers} workers...")

    # 分配帧给不同的worker
    frames_per_worker = num_frames // num_workers
    tasks = []
    for i in range(num_workers):
        start = i * frames_per_worker
        end = (i + 1) * frames_per_worker if i < num_workers - 1 else num_frames
        # 测试每个worker解码几个帧
        frame_indices = [start, (start + end) // 2, end - 1]
        tasks.append((video_path, frame_indices, i))

    # 使用multiprocessing
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(decode_video_worker, tasks)

    # 打印结果
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for err in errors:
            print(f"   Worker {err['worker_id']}: {err['error']}")
    else:
        print(f"\n✅ All {num_workers} workers decoded successfully")

    return errors


def test_sequential_decode(video_path: str, sample_rate: int = 100):
    """Test sequential decoding of video frames."""
    import fsspec
    from torchcodec.decoders import VideoDecoder

    print(f"\nTesting sequential decode for: {video_path}")

    # 获取视频信息
    file_handle = fsspec.open(video_path).__enter__()
    decoder = VideoDecoder(file_handle, seek_mode="approximate")
    num_frames = decoder.metadata.num_frames
    fps = decoder.metadata.average_fps

    print(f"Frames: {num_frames}, FPS: {fps}")

    errors = []

    # 测试解码每隔sample_rate帧
    for i in range(0, num_frames, sample_rate):
        try:
            frames = decoder.get_frames_at(indices=[i])
        except Exception as e:
            errors.append({"frame_index": i, "error": str(e)})
            if len(errors) >= 10:  # 只记录前10个错误
                break

    file_handle.close()

    if errors:
        print(f"\n❌ Found {len(errors)} decode errors:")
        for err in errors[:10]:
            print(f"   Frame {err['frame_index']}: {err['error'][:100]}")
    else:
        print(f"✅ Successfully decoded {num_frames // sample_rate} sample frames")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Test video decoding")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--sample_rate", type=int, default=100, help="Sample rate for sequential test")

    args = parser.parse_args()

    # 测试顺序解码
    seq_errors = test_sequential_decode(args.video_path, args.sample_rate)

    # 测试多进程解码
    mp_errors = test_multiprocess_decode(args.video_path, args.num_workers)

    return seq_errors + mp_errors


if __name__ == "__main__":
    main()
