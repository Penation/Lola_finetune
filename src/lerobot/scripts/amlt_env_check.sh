#!/bin/bash

set -euo pipefail

BACKEND="${1:-}"

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

python - <<'PY'
import glob
import importlib.util
import os
import sys

import torch
import torchvision

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torch_file", torch.__file__)
print("CONDA_PREFIX", os.environ.get("CONDA_PREFIX"))
print("TORCH_LIB_DIR", os.path.join(os.path.dirname(torch.__file__), "lib"))
print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH"))
print("libavutil", glob.glob(os.path.join(sys.prefix, "lib", "libavutil.so*")))
spec = importlib.util.find_spec("torchcodec")
print("torchcodec_spec", getattr(spec, "origin", None))
PY

if [[ "${BACKEND}" == "torchcodec" ]]; then
    python - <<'PY'
import torchcodec
from torchcodec.decoders import VideoDecoder

print("torchcodec", torchcodec.__version__)
print("decoder_import_ok", VideoDecoder.__name__)
PY
fi
