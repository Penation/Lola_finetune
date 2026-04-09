#!/bin/bash
# LoLA 多卡分布式训练测试脚本
# 使用 LeRobotDataset 进行训练

# 基础训练参数
STRATEGY="fsdp"
DEVICES=2
NUM_NODES=1
BATCH_SIZE=16
MAX_STEPS=10000
LEARNING_RATE=2.5e-5
PRECISION="bf16-mixed"
LOG_EVERY_N_STEPS=100
SAVE_INTERVAL=5000

# 数据集参数
DATASET_REPO_ID=""
DATASET_ROOT="/mnt/wangxiaofa/robot_dataset/lerobot-format-v30/simpler_bridge_v3"

# 模型参数
VLM_PATH="/mnt/wangxiaofa/qwen3_5/Qwen3.5-4B/"
CKPT_DIR="/mnt/wangxiaofa/policy_ckpts/lola_ckpts/"

# 历史action加载参数（可选）
LOAD_FULL_HISTORY=true  # 设置为true启用完整历史action加载
MAX_HISTORY_LENGTH=1024
HISTORY_PADDING_SIDE="left"

# 运行训练
cmd="torchrun --nproc_per_node=${DEVICES} src/lerobot/scripts/train_lola_multigpu.py \
    --dataset_root ${DATASET_ROOT} \
    --strategy ${STRATEGY} \
    --devices ${DEVICES} \
    --num_nodes ${NUM_NODES} \
    --batch_size ${BATCH_SIZE} \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --precision ${PRECISION} \
    --log_every_n_steps ${LOG_EVERY_N_STEPS} \
    --save_every_n_steps ${SAVE_INTERVAL} \
    --ckpt_dir \
    --vlm_path ${VLM_PATH}"

# 如果启用完整历史action加载
if [ "$LOAD_FULL_HISTORY" = true ]; then
    cmd="${cmd} --load_full_history --max_history_length ${MAX_HISTORY_LENGTH} --history_padding_side ${HISTORY_PADDING_SIDE}"
fi

echo "Running: $cmd"
eval $cmd
