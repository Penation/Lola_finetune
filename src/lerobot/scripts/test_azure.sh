#!/bin/bash
# LoLA Azure 分布式训练脚本
#
# 此脚本用于在 Azure ML 上运行分布式训练。
# Azure ML 会为每个节点运行一次此脚本，并自动传入以下参数：
#   --nnodes: 节点数量
#   --nproc_per_node: 每个节点的 GPU 数量
#   --node_rank: 当前节点的 rank
#   --master_addr: 主节点 IP
#   --master_port: 主节点端口
#
# 使用方法:
#   bash test_azure.sh --nnodes $NODES --nproc_per_node $GPUS \
#       --node_rank $AZUREML_CR_NODE_RANK \
#       --master_addr $AZ_BATCHAI_JOB_MASTER_NODE_IP \
#       --master_port 9901

set -e

# 环境变量设置
export OPENSSL_FIPS=0  # 禁用 FIPS 避免自检失败
export TOKENIZERS_PARALLELISM=false

# ----------------------------------------------------------------------
# 默认参数（可被命令行参数覆盖）
# ----------------------------------------------------------------------
NNODES=1
NPROC_PER_NODE=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"  # 使用 IP 而非 localhost，避免 IPv6 问题
MASTER_PORT=29500

# 训练参数
STRATEGY="fsdp"
BATCH_SIZE=4
MAX_STEPS=10000
MAX_EPOCHS=""
LEARNING_RATE=2.5e-5
LOG_EVERY_N_STEPS=10
SAVE_INTERVAL=5000
GRADIENT_CLIP_VAL=1.0
NUM_WORKERS=4
ENTRY_SCRIPT="src/lerobot/scripts/train_lola_azure.py"

# 数据集参数
DATASET_REPO_ID=""
DATASET_ROOT="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/simpler_bridge_v3"
VIDEO_BACKEND=""

# 模型参数
VLM_PATH="/mnt/wangxiaofa/qwen3_5/Qwen3.5-4B/"
CKPT_DIR="/mnt/wangxiaofa/checkpoints/lola-simpler"
TRAIN_VLM=false
STAGE_TRAIN_VLM_AFTER_EPOCH=0
SAVE_CHECKPOINT_ON_VLM_UNFREEZE=false
SAVE_CHECKPOINT_EVERY_EPOCH=false

# 历史action加载参数
LOAD_FULL_HISTORY=true
MAX_HISTORY_LENGTH=1024
HISTORY_PADDING_SIDE="left"
CONVERT_CALVIN_RPY_TO_ORTHO6D=false
CALVIN_XYZ_ONLY_NORMALIZE=false

# Wandb 参数
WANDB_PROJECT="lola-azure"
WANDB_NAME=""
WANDB_ENTITY=""
DISABLE_WANDB=false

# Resume 参数
RESUME=""

# ----------------------------------------------------------------------
# 解析命令行参数
# ----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        # Azure 分布式参数
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;

        # 训练参数
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --log_every_n_steps)
            LOG_EVERY_N_STEPS="$2"
            shift 2
            ;;
        --save_every_n_steps)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --gradient_clip_val)
            GRADIENT_CLIP_VAL="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --entry_script)
            ENTRY_SCRIPT="$2"
            shift 2
            ;;

        # 数据集参数
        --dataset_repo_id)
            DATASET_REPO_ID="$2"
            shift 2
            ;;
        --dataset_root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --video_backend)
            VIDEO_BACKEND="$2"
            shift 2
            ;;

        # 模型参数
        --vlm_path)
            VLM_PATH="$2"
            shift 2
            ;;
        --ckpt_dir)
            CKPT_DIR="$2"
            shift 2
            ;;
        --train_vlm)
            TRAIN_VLM=true
            shift
            ;;
        --stage_train_vlm_after_epoch)
            STAGE_TRAIN_VLM_AFTER_EPOCH="$2"
            shift 2
            ;;
        --save_checkpoint_on_vlm_unfreeze)
            SAVE_CHECKPOINT_ON_VLM_UNFREEZE=true
            shift
            ;;
        --save_checkpoint_every_epoch)
            SAVE_CHECKPOINT_EVERY_EPOCH=true
            shift
            ;;

        # 历史action参数
        --load_full_history)
            LOAD_FULL_HISTORY=true
            shift
            ;;
        --no_load_full_history)
            LOAD_FULL_HISTORY=false
            shift
            ;;
        --max_history_length)
            MAX_HISTORY_LENGTH="$2"
            shift 2
            ;;
        --history_padding_side)
            HISTORY_PADDING_SIDE="$2"
            shift 2
            ;;
        --convert_calvin_rpy_to_ortho6d)
            CONVERT_CALVIN_RPY_TO_ORTHO6D=true
            shift
            ;;
        --calvin_xyz_only_normalize)
            CALVIN_XYZ_ONLY_NORMALIZE=true
            shift
            ;;

        # Wandb 参数
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_name)
            WANDB_NAME="$2"
            shift 2
            ;;
        --wandb_entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --disable_wandb)
            DISABLE_WANDB=true
            shift
            ;;

        # Resume
        --resume)
            RESUME="$2"
            shift 2
            ;;

        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# TorchCodec 依赖 conda 环境中的 FFmpeg 动态库。`conda run` 不保证所有后续
# 子进程都能继承到正确的 loader 路径，因此这里显式补齐。
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

if [[ "${VIDEO_BACKEND}" == "torchcodec" ]]; then
    echo "TorchCodec runtime:"
    echo "  - CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}"
    echo "  - TORCH_LIB_DIR: ${TORCH_LIB_DIR:-<unset>}"
    echo "  - LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"
    ls -1 "${CONDA_PREFIX}/lib"/libavutil.so* 2>/dev/null || true
    python - <<'PY'
import importlib.util
import torch

print("torch file:", torch.__file__)
print("torchcodec spec:", getattr(importlib.util.find_spec("torchcodec"), "origin", None))
from torchcodec.decoders import VideoDecoder
print("torchcodec decoder import OK:", VideoDecoder.__name__)
PY
fi


# 打印配置信息
echo "========================================"
echo "LoLA Azure Distributed Training"
echo "========================================"
echo "Distributed Config:"
echo "  - Nodes: ${NNODES}"
echo "  - GPUs per node: ${NPROC_PER_NODE}"
echo "  - World size: ${WORLD_SIZE}"
echo "  - Node rank: ${NODE_RANK}"
echo "  - Master addr: ${MASTER_ADDR}"
echo "  - Master port: ${MASTER_PORT}"
echo ""
echo "Training Config:"
echo "  - Strategy: ${STRATEGY}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Max steps: ${MAX_STEPS}"
echo "  - Max epochs: ${MAX_EPOCHS}"
echo "  - Learning rate: ${LEARNING_RATE}"
echo "  - Gradient clip: ${GRADIENT_CLIP_VAL}"
echo "  - Num workers: ${NUM_WORKERS}"
echo "  - Dataset: ${DATASET_REPO_ID:-$DATASET_ROOT}"
echo "  - Video backend: ${VIDEO_BACKEND:-auto}"
echo "  - VLM path: ${VLM_PATH}"
echo "  - Stage train VLM after epoch: ${STAGE_TRAIN_VLM_AFTER_EPOCH}"
echo "  - Resume: ${RESUME:-<none>}"
echo "  - Entry script: ${ENTRY_SCRIPT}"
echo "========================================"

# ----------------------------------------------------------------------
# 启动训练
# 使用 torchrun 来管理多 GPU，每个节点运行一次
# 单节点时使用简化命令，多节点时使用完整参数
# ----------------------------------------------------------------------
if [ "$NNODES" -eq 1 ]; then
    # 单节点：使用简化的 torchrun 命令
    cmd="torchrun --nproc_per_node=${NPROC_PER_NODE} \
        ${ENTRY_SCRIPT} \
        --strategy ${STRATEGY} \
        --batch_size ${BATCH_SIZE} \
        --max_steps ${MAX_STEPS} \
        --learning_rate ${LEARNING_RATE} \
        --log_every_n_steps ${LOG_EVERY_N_STEPS} \
        --save_every_n_steps ${SAVE_INTERVAL} \
        --gradient_clip_val ${GRADIENT_CLIP_VAL} \
        --num_workers ${NUM_WORKERS} \
        --vlm_path ${VLM_PATH} \
        --ckpt_dir ${CKPT_DIR} \
        --wandb_project ${WANDB_PROJECT}"
else
    # 多节点：使用完整的分布式参数
    cmd="torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        ${ENTRY_SCRIPT} \
        --strategy ${STRATEGY} \
        --batch_size ${BATCH_SIZE} \
        --max_steps ${MAX_STEPS} \
        --learning_rate ${LEARNING_RATE} \
        --log_every_n_steps ${LOG_EVERY_N_STEPS} \
        --save_every_n_steps ${SAVE_INTERVAL} \
        --gradient_clip_val ${GRADIENT_CLIP_VAL} \
        --num_workers ${NUM_WORKERS} \
        --vlm_path ${VLM_PATH} \
        --ckpt_dir ${CKPT_DIR} \
        --wandb_project ${WANDB_PROJECT}"
fi

# 数据集参数
if [ -n "$DATASET_REPO_ID" ]; then
    cmd="${cmd} --dataset_repo_id ${DATASET_REPO_ID}"
else
    cmd="${cmd} --dataset_root ${DATASET_ROOT}"
fi

if [ -n "$VIDEO_BACKEND" ]; then
    cmd="${cmd} --video_backend ${VIDEO_BACKEND}"
fi

# 历史action参数
if [ "$LOAD_FULL_HISTORY" = true ]; then
    cmd="${cmd} --load_full_history --max_history_length ${MAX_HISTORY_LENGTH} --history_padding_side ${HISTORY_PADDING_SIDE}"
fi

if [ "$CONVERT_CALVIN_RPY_TO_ORTHO6D" = true ]; then
    cmd="${cmd} --convert_calvin_rpy_to_ortho6d"
fi

if [ "$CALVIN_XYZ_ONLY_NORMALIZE" = true ]; then
    cmd="${cmd} --calvin_xyz_only_normalize"
fi

if [ -n "$MAX_EPOCHS" ]; then
    cmd="${cmd} --max_epochs ${MAX_EPOCHS}"
fi

# 训练 VLM 参数
if [ "$TRAIN_VLM" = true ]; then
    cmd="${cmd} --train_vlm"
fi

if [ "$STAGE_TRAIN_VLM_AFTER_EPOCH" -gt 0 ]; then
    cmd="${cmd} --stage_train_vlm_after_epoch ${STAGE_TRAIN_VLM_AFTER_EPOCH}"
fi

if [ "$SAVE_CHECKPOINT_ON_VLM_UNFREEZE" = true ]; then
    cmd="${cmd} --save_checkpoint_on_vlm_unfreeze"
fi

if [ "$SAVE_CHECKPOINT_EVERY_EPOCH" = true ]; then
    cmd="${cmd} --save_checkpoint_every_epoch"
fi

# Wandb 参数
if [ -n "$WANDB_NAME" ]; then
    cmd="${cmd} --wandb_name ${WANDB_NAME}"
fi
if [ -n "$WANDB_ENTITY" ]; then
    cmd="${cmd} --wandb_entity ${WANDB_ENTITY}"
fi
if [ "$DISABLE_WANDB" = true ]; then
    cmd="${cmd} --disable_wandb"
fi

# Resume 参数
if [ -n "$RESUME" ]; then
    cmd="${cmd} --resume ${RESUME}"
fi

echo "Running: $cmd"
eval $cmd

echo "Training completed!"
