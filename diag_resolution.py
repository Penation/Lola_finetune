#!/usr/bin/env python3
"""Diagnostic: decode ALL frames from problematic video files using both
torchcodec and PyAV, recording per-frame resolution to investigate
the 128→256 upscale mystery and AV1 dynamic resolution.
cv2 skipped — no AV1 software decoder on this platform."""

import subprocess
import sys
from collections import Counter

VIDEO_FILES = [
    "/data_6t_1/lerobot-v30/merged_0419_mini_v2/videos/observation.images.wrist/chunk-000/file-040.mp4",
    "/data_6t_1/lerobot-v30/merged_0419_mini_v2/videos/observation.images.primary/chunk-000/file-059.mp4",
]


def ffprobe_info(path):
    """Print ffprobe stream info for cross-reference."""
    print(f"\n{'='*60}")
    print(f"ffprobe info: {path}")
    print(f"{'='*60}")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,sample_aspect_ratio,display_aspect_ratio",
        "-of", "default=noprint_wrappers=1",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(result.stdout)
    except Exception as e:
        print(f"ffprobe failed: {e}")


def decode_torchcodec_all_frames(path):
    """Decode all frames with torchcodec, record per-frame resolution."""
    print(f"\n--- torchcodec: {path} ---")
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError:
        print("torchcodec.decoders.VideoDecoder not available, skipping")
        return

    try:
        decoder = VideoDecoder(path)
        num_frames = decoder.metadata.num_frames
        print(f"Total frames (metadata): {num_frames}")

        resolutions = []
        for i in range(num_frames):
            try:
                frame = decoder[i]
                # frame.data is (C, H, W) tensor
                h, w = frame.data.shape[1], frame.data.shape[2]
                resolutions.append((h, w))
                if (i < 5) or (i >= num_frames - 2) or (h, w) != resolutions[0]:
                    print(f"  frame {i}: {h}x{w}")
            except RuntimeError as e:
                print(f"  frame {i}: DECODE ERROR: {e}")
                resolutions.append(None)

        # Summary
        valid = [r for r in resolutions if r is not None]
        counter = Counter(valid)
        print(f"\n  Resolution distribution:")
        for res, count in counter.most_common():
            print(f"    {res[0]}x{res[1]}: {count} frames")
        if len(counter) > 1:
            print(f"  *** DYNAMIC RESOLUTION DETECTED: {len(counter)} distinct resolutions ***")
        else:
            print(f"  All frames have same resolution")

    except Exception as e:
        print(f"torchcodec failed: {type(e).__name__}: {e}")


def decode_pyav_all_frames(path):
    """Decode all frames with PyAV, record per-frame resolution."""
    print(f"\n--- PyAV: {path} ---")
    try:
        import av
    except ImportError:
        print("PyAV not available, skipping")
        return

    try:
        container = av.open(path)
        stream = container.streams.video[0]
        print(f"Stream codec: {stream.codec_context.name}")
        print(f"Stream width x height: {stream.width}x{stream.height}")
        print(f"Stream frames: {stream.frames}")

        resolutions = []
        idx = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                h, w = frame.height, frame.width
                resolutions.append((h, w))
                if (idx < 5) or (h, w) != resolutions[0]:
                    print(f"  frame {idx}: {w}x{h}")
                idx += 1

        container.close()

        # Summary
        valid = [r for r in resolutions if r is not None]
        counter = Counter(valid)
        print(f"\n  Actually decoded {idx} frames")
        print(f"  Resolution distribution:")
        for res, count in counter.most_common():
            print(f"    {res[0]}x{res[1]}: {count} frames")
        if len(counter) > 1:
            print(f"  *** DYNAMIC RESOLUTION DETECTED: {len(counter)} distinct resolutions ***")
        else:
            print(f"  All frames have same resolution")

    except Exception as e:
        print(f"PyAV failed: {type(e).__name__}: {e}")


def main():
    for vf in VIDEO_FILES:
        print(f"\n{'#'*60}")
        print(f"# FILE: {vf}")
        print(f"{'#'*60}")

        ffprobe_info(vf)
        decode_torchcodec_all_frames(vf)
        decode_pyav_all_frames(vf)

    print(f"\n{'='*60}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
