#!/bin/bash
# 生成 Kolmogorov Re=1000 训练/评估数据集
# 用法: bash scripts/run_generate_data.sh [GPU_ID] [OUTPUT_DIR]
#
# 默认: GPU 4, 输出到 content/my_kolmogorov_re1000/
# 示例: bash scripts/run_generate_data.sh 3 content/my_data_re1000

set -e

GPU_ID="${1:-4}"
OUTPUT_DIR="${2:-/jumbo/yaoqingyang/yuxin/JAX-CFD/content/my_kolmogorov_re1000}"
DNS_SIZE=2048     # DNS 网格: 2048×2048 → 64×64 保存; 复刻原始数据精度
SAVE_SIZE=64

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
SCRIPT="${ROOT}/scripts/generate_kolmogorov_data.py"
LOG_DIR="${ROOT}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================="
echo "  GPU: ${GPU_ID}  DNS: ${DNS_SIZE}x${DNS_SIZE}"
echo "  OUTPUT: ${OUTPUT_DIR}"
echo "========================================="

# --- 训练集 (32 samples, 30s) ---
TRAIN_NC="${OUTPUT_DIR}/train_${DNS_SIZE}x${DNS_SIZE}_${SAVE_SIZE}x${SAVE_SIZE}.nc"
echo ""
echo "[TRAIN] 开始生成 train.nc ..."
echo "  预计时间: ~几天 (2048×2048 DNS，原始精度)"
nohup ${PYTHON} -u "${SCRIPT}" \
    --output "${TRAIN_NC}" \
    --num_samples 32 \
    --dns_size "${DNS_SIZE}" \
    --save_size "${SAVE_SIZE}" \
    --warmup_time 40.0 \
    --simulation_time 30.0 \
    --seed 0 \
    --chunk_steps 200 \
    > "${LOG_DIR}/generate_train_gpu${GPU_ID}.log" 2>&1

echo "[TRAIN] 完成: ${TRAIN_NC}"

# --- 评估集 (16 samples, 240s) ---
EVAL_NC="${OUTPUT_DIR}/long_eval_${DNS_SIZE}x${DNS_SIZE}_${SAVE_SIZE}x${SAVE_SIZE}.nc"
echo ""
echo "[EVAL] 开始生成 long_eval.nc ..."
echo "  预计时间: ~更长 (2048×2048 DNS)"
nohup ${PYTHON} -u "${SCRIPT}" \
    --output "${EVAL_NC}" \
    --num_samples 16 \
    --dns_size "${DNS_SIZE}" \
    --save_size "${SAVE_SIZE}" \
    --warmup_time 40.0 \
    --simulation_time 240.0 \
    --seed 2 \
    --chunk_steps 200 \
    > "${LOG_DIR}/generate_eval_gpu${GPU_ID}.log" 2>&1

echo "[EVAL] 完成: ${EVAL_NC}"
echo ""
echo "所有数据已生成到 ${OUTPUT_DIR}"
