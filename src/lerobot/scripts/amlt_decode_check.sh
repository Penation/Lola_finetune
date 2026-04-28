#!/bin/bash

set -euo pipefail

DATASET_ROOT="${1:?dataset root required}"
BACKEND="${2:?backend required}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    TORCH_LIB_DIR="$(python - <<'PY'
import os
import torch

print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
    if [[ -d "${TORCH_LIB_DIR}" ]]; then
        EXTRA_LD_PATH="${TORCH_LIB_DIR}"
    fi
    if [[ -d "${CONDA_PREFIX}/lib" ]]; then
        EXTRA_LD_PATH="${EXTRA_LD_PATH:+${EXTRA_LD_PATH}:}${CONDA_PREFIX}/lib"
    fi
    export LD_LIBRARY_PATH="${EXTRA_LD_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LIBRARY_PATH="${EXTRA_LD_PATH}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

python - "$DATASET_ROOT" "$BACKEND" <<'PY'
from pathlib import Path
import sys

from lerobot.datasets.video_utils import decode_video_frames

dataset_root = Path(sys.argv[1])
backend = sys.argv[2]
video = sorted(dataset_root.glob("videos/chunk-*/observation.images.primary/*.mp4"))[0]
frames = decode_video_frames(video, [0.0, 1.0 / 30.0, 2.0 / 30.0], tolerance_frames=1, backend=backend)
print("decode_backend", backend)
print("decode_video", video)
print("decoded_shape", tuple(frames.shape))
PY
