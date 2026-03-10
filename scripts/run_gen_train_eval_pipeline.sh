#!/usr/bin/env bash
# ============================================================
# 完整 pipeline：生成数据 → 训练 1 epoch → 评估
#
# 用法：
#   bash scripts/run_gen_train_eval_pipeline.sh [GPU_ID]
#   bash scripts/run_gen_train_eval_pipeline.sh 4
#
# 流程：
#   Step 1  生成 train.nc      (~1.3h, 2048×2048 DNS, 32 samples, 30s)
#   Step 2  生成 eval.nc       (~20min, 2048×2048 DNS, 16 samples, 20s)
#   Step 3  训练 1 epoch       (~1.5h)
#   Step 4  notebook 风格评估  (~10min, 包含 L2 error + vorticity + 能谱)
# ============================================================
set -eo pipefail

GPU_ID="${1:-4}"
ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"

# ---------- 环境 ----------
CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${ROOT}/jax-cfd:${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

# ---------- 路径 ----------
DATA_DIR="${ROOT}/content/my_kolmogorov_re1000"
TRAIN_NC="${DATA_DIR}/train_2048x2048_64x64.nc"
EVAL_NC="${DATA_DIR}/long_eval_2048x2048_64x64.nc"
MODEL_OUT="${ROOT}/model_mydata_1ep"
LOG_DIR="${ROOT}/logs"
mkdir -p "${DATA_DIR}" "${MODEL_OUT}" "${LOG_DIR}"

# 原始 baseline 目录（不同分辨率 DNS，用作对比参考）
ORIG_BASE="${ROOT}/content/kolmogorov_re_1000"

PIPELINE_LOG="${LOG_DIR}/pipeline_gpu${GPU_ID}.log"
TOTAL_START=$(date +%s)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${PIPELINE_LOG}"; }

log "========================================="
log "  Pipeline start  GPU=${GPU_ID}"
log "  DATA_DIR  = ${DATA_DIR}"
log "  MODEL_OUT = ${MODEL_OUT}"
log "========================================="

# ============================================================
# Step 1: 生成 train.nc
# ============================================================
if [ -f "${TRAIN_NC}" ]; then
    log "[Step 1] train.nc 已存在，跳过生成"
else
    log "[Step 1] 生成 train.nc (2048×2048 DNS, 32 samples, 30s) ..."
    T0=$(date +%s)
    ${PYTHON} -u "${ROOT}/scripts/generate_kolmogorov_data.py" \
        --output   "${TRAIN_NC}" \
        --num_samples 32 \
        --dns_size  2048 \
        --save_size 64 \
        --warmup_time 40.0 \
        --simulation_time 30.0 \
        --seed 0 \
        --chunk_steps 100 \
        2>&1 | tee -a "${LOG_DIR}/gen_train_gpu${GPU_ID}.log"
    log "[Step 1] Done in $(( ($(date +%s)-T0)/60 ))min"
fi

# ============================================================
# Step 2: 生成 eval.nc（20s 足够评估用）
# ============================================================
if [ -f "${EVAL_NC}" ]; then
    log "[Step 2] eval.nc 已存在，跳过生成"
else
    log "[Step 2] 生成 eval.nc (2048×2048 DNS, 16 samples, 20s) ..."
    T0=$(date +%s)
    ${PYTHON} -u "${ROOT}/scripts/generate_kolmogorov_data.py" \
        --output   "${EVAL_NC}" \
        --num_samples 16 \
        --dns_size  2048 \
        --save_size 64 \
        --warmup_time 40.0 \
        --simulation_time 20.0 \
        --seed 2 \
        --chunk_steps 100 \
        2>&1 | tee -a "${LOG_DIR}/gen_eval_gpu${GPU_ID}.log"
    log "[Step 2] Done in $(( ($(date +%s)-T0)/60 ))min"
fi

# ============================================================
# Step 3: 训练 1 epoch（使用自己生成的数据）
# ============================================================
log "[Step 3] 训练 1 epoch ..."
T0=$(date +%s)
cd "${ROOT}"

${PYTHON} -u models/train.py \
    --model_encode_steps=16 \
    --model_decode_steps=32 \
    --model_predict_steps=16 \
    --train_device_batch_size=4 \
    --delta_time=0.007012483601762931 \
    --train_split="${TRAIN_NC}" \
    --eval_split="${EVAL_NC}" \
    --eval_batch_size=32 \
    --train_weight_decay=0.0 \
    --train_lr_init=0.001 \
    --train_lr_warmup_epochs=0.0 \
    --mp_scale_value=1.0 \
    --train_epochs=1 \
    --train_log_every=100 \
    --decoding_warmup_steps=0 \
    --mp_skip_nonfinite \
    --do_eval \
    --output_dir="${MODEL_OUT}" \
    --gin_file="models/configs/official_li_config.gin" \
    --gin_file="models/configs/kolmogorov_forcing.gin" \
    --gin_param="fixed_scale.rescaled_one = 0.2" \
    --gin_param="my_forward_tower_factory.num_hidden_channels = 128" \
    --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
    --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
    --dataset_num_workers=0 \
    2>&1 | tee "${LOG_DIR}/train_mydata_1ep.log"

log "[Step 3] Done in $(( ($(date +%s)-T0)/60 ))min"

# ============================================================
# Step 4: 评估（notebook 风格，含 L2 error）
# ============================================================
log "[Step 4] 评估 ..."
T0=$(date +%s)

${PYTHON} -u "${ROOT}/scripts/run_inference_eval.py" \
    --model_dir    "${MODEL_OUT}" \
    --output_dir   "${MODEL_OUT}" \
    --eval_nc      "${EVAL_NC}" \
    --train_nc     "${TRAIN_NC}" \
    --baselines_dir "${ORIG_BASE}" \
    --length       200 \
    --inner_steps  10 \
    --model_label  "LI_mydata_1ep" \
    2>&1 | tee "${LOG_DIR}/eval_mydata_1ep.log"

log "[Step 4] Done in $(( ($(date +%s)-T0)/60 ))min"

# ============================================================
ELAPSED=$(( $(date +%s) - TOTAL_START ))
log "========================================="
log "  Pipeline DONE"
log "  Total: $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))min"
log "  Plots: ${MODEL_OUT}/"
log "    notebook_eval_corr_spectrum.png"
log "    notebook_eval_l2_error.png"
log "    notebook_eval_vorticity.png"
log "========================================="
