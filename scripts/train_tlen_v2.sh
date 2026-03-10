#!/usr/bin/env bash
# tlen v2: Re={500,1000,2000,3000,4000} x tlen={50,100,500,1000,2000,4000}
# 3 data_seeds x 3 model_seeds = 9 models per (Re, tlen)  = 270 total
#
# 调度策略：
#   Phase-A  先跑 data_seed=0 的全部 90 个模型（现有数据，无需等待生成）
#   Phase-B  Phase-A 跑完后，生成 data_seed=1,2 数据（每 Re 独立，5 Re 并行 1 GPU/each）
#   Phase-C  数据生成完再跑剩余 180 个模型
#
# 每个 data_seed 对应的 3 个 model_seeds 全部跑完 → 触发 eval + phase plot
#
# GPU 调度：greedy 1进程/GPU，GPUs 4-7
# 运行：nohup bash scripts/train_tlen_v2.sh > logs/train_tlen_v2_master.log 2>&1 &

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export OMP_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=4
export TF_NUM_INTRAOP_THREADS=4

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
TRAIN_STEPS=50000
LR=0.001
EARLY_STOP_DELTA=1e-6
EARLY_STOP_PATIENCE=10
ENCODE_STEPS=16
DECODE_STEPS=32
BATCH_SIZE=16
DELTA_TIME=0.007012483601762931
NTRAJ=32
INNER_STEPS=10
DNS_SIZE=2048
SAVE_SIZE=64
GEN_SIMULATION_TIME=30.0
GEN_WARMUP_TIME=40.0

RE_LIST=(500 1000 2000 3000 4000)
TLEN_LIST=(50 100 500 1000 2000 4000)
DATA_SEEDS=(0 1 2)
MODEL_SEEDS=(0 1 2)
GPUS=(4 5 6 7)

QUEUE_FILE="/tmp/tlen_v2_queue.txt"
LOCK_FILE="/tmp/tlen_v2_queue.lock"
PLOT_LOCK="/tmp/tlen_v2_plot.lock"

mkdir -p "${ROOT}/logs/train_tlen_v2" "${ROOT}/results/tlen_v2"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── data_seed → 训练数据文件路径 ─────────────────────────────────────
data_nc() {
  local RE=$1 DSEED=$2
  if [[ $DSEED -eq 0 ]]; then
    echo "${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
  else
    echo "${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64_dseed${DSEED}.nc"
  fi
}

# ── 生成数据（data_seed >= 1） ─────────────────────────────────────
generate_data() {
  local RE=$1 DSEED=$2 GPU=$3
  local OUT_NC
  OUT_NC=$(data_nc "$RE" "$DSEED")
  if [[ -f "$OUT_NC" ]]; then
    log "[GPU${GPU}] DATA SKIP re${RE} dseed${DSEED}"
    return
  fi
  log "[GPU${GPU}] DATA GEN re${RE} dseed${DSEED} → $OUT_NC"
  mkdir -p "$(dirname "$OUT_NC")"
  CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u "${ROOT}/scripts/generate_kolmogorov_data.py" \
    --re "${RE}" \
    --output "${OUT_NC}" \
    --num_samples ${NTRAJ} \
    --dns_size ${DNS_SIZE} \
    --save_size ${SAVE_SIZE} \
    --warmup_time ${GEN_WARMUP_TIME} \
    --simulation_time ${GEN_SIMULATION_TIME} \
    --seed "${DSEED}" \
    > "${ROOT}/logs/train_tlen_v2/gen_re${RE}_dseed${DSEED}.log" 2>&1
  log "[GPU${GPU}] DATA DONE re${RE} dseed${DSEED}"
}

# ── 从队列弹出任务（原子） ─────────────────────────────────────────
pop_job() {
  (
    flock -x 9
    local job
    job=$(head -1 "${QUEUE_FILE}" 2>/dev/null)
    if [[ -n "$job" ]]; then
      tail -n +2 "${QUEUE_FILE}" > "${QUEUE_FILE}.tmp" \
        && mv "${QUEUE_FILE}.tmp" "${QUEUE_FILE}"
    fi
    echo "$job"
  ) 9>"${LOCK_FILE}"
}

# ── 训练单个模型 ──────────────────────────────────────────────────
train_one() {
  local RE=$1 TLEN=$2 DSEED=$3 MSEED=$4 GPU=$5
  local MODEL_DIR="${ROOT}/models_tlen_v2/re${RE}_tlen${TLEN}_dseed${DSEED}_mseed${MSEED}"
  local LOG="${ROOT}/logs/train_tlen_v2/re${RE}_tlen${TLEN}_dseed${DSEED}_mseed${MSEED}.log"
  local TRAIN_NC
  TRAIN_NC=$(data_nc "$RE" "$DSEED")

  if ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp"; then
    log "[SKIP] re${RE}_tlen${TLEN}_d${DSEED}_m${MSEED}"
    return
  fi
  if [[ ! -f "$TRAIN_NC" ]]; then
    log "[WARN] data not found: $TRAIN_NC — skipping"
    return
  fi

  mkdir -p "${MODEL_DIR}"
  log "[GPU${GPU}] TRAIN re${RE} tlen=${TLEN} d${DSEED} m${MSEED}"

  CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u "${ROOT}/models/train.py" \
    --gin_file="${ROOT}/models/configs/official_li_config.gin" \
    --gin_file="${ROOT}/models/configs/kolmogorov_forcing.gin" \
    --gin_param="fixed_scale.rescaled_one = 0.2" \
    --gin_param="my_forward_tower_factory.num_hidden_channels = 128" \
    --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
    --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
    "--gin_param=physics_specifications.NavierStokesPhysicsSpecs.viscosity = $(${PYTHON} -c "print(1/${RE})")" \
    --train_split="${TRAIN_NC}" \
    --eval_split="${TRAIN_NC}" \
    --train_steps=${TRAIN_STEPS} \
    --train_lr_init=${LR} \
    --train_lr_warmup_epochs=0.0 \
    --early_stop_loss_delta=${EARLY_STOP_DELTA} \
    --early_stop_patience=${EARLY_STOP_PATIENCE} \
    --train_init_random_seed=${MSEED} \
    --max_train_samples=${NTRAJ} \
    --max_time_steps=${TLEN} \
    --model_encode_steps=${ENCODE_STEPS} \
    --model_decode_steps=${DECODE_STEPS} \
    --delta_time=${DELTA_TIME} \
    --train_device_batch_size=${BATCH_SIZE} \
    --train_weight_decay=0.0 \
    --mp_skip_nonfinite \
    --mp_scale_value=1.0 \
    --dataset_num_workers=0 \
    --output_dir="${MODEL_DIR}" \
    > "${LOG}" 2>&1

  log "[GPU${GPU}] DONE  re${RE} tlen=${TLEN} d${DSEED} m${MSEED}"
}

# ── eval + plot（data_seed 组完成时触发） ─────────────────────────
check_and_plot() {
  local DSEED=$1
  local total=$(( ${#RE_LIST[@]} * ${#TLEN_LIST[@]} * ${#MODEL_SEEDS[@]} ))
  local done=0
  for RE in "${RE_LIST[@]}"; do
    for TLEN in "${TLEN_LIST[@]}"; do
      for MSEED in "${MODEL_SEEDS[@]}"; do
        local MODEL_DIR="${ROOT}/models_tlen_v2/re${RE}_tlen${TLEN}_dseed${DSEED}_mseed${MSEED}"
        ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp" && ((done++)) || true
      done
    done
  done
  log "  dseed${DSEED}: ${done}/${total} done"
  if [[ $done -eq $total ]]; then
    (
      flock -x 9
      local marker="${ROOT}/results/tlen_v2/.dseed${DSEED}_plotted"
      if [[ ! -f "$marker" ]]; then
        touch "$marker"
        log "=== data_seed=${DSEED} 全部完成！触发 eval + plot ==="
        bash "${ROOT}/scripts/eval_and_plot_tlen_v2.sh" \
          >> "${ROOT}/logs/train_tlen_v2/eval_dseed${DSEED}.log" 2>&1 \
          && log "=== data_seed=${DSEED} phase plot 完成 ===" \
          || log "!!! data_seed=${DSEED} eval 失败"
      fi
    ) 9>"${PLOT_LOCK}"
  fi
}

# ── GPU Worker ───────────────────────────────────────────────────
worker() {
  local GPU=$1
  log "[GPU${GPU}] Worker 启动"
  while true; do
    local job
    job=$(pop_job)
    [[ -z "$job" ]] && break
    read -r RE TLEN DSEED MSEED <<< "$job"
    train_one "$RE" "$TLEN" "$DSEED" "$MSEED" "$GPU"
    check_and_plot "$DSEED"
  done
  log "[GPU${GPU}] Worker 结束"
}

# =====================================================================
# 主逻辑
# =====================================================================
log "===== train_tlen_v2.sh 开始 $(date) ====="
log "  5Re x 6tlen x 3data_seeds x 3model_seeds = 270 个模型"
log "  early_stop_delta=${EARLY_STOP_DELTA}, patience=${EARLY_STOP_PATIENCE}"

# ---- Phase A: data_seed=0（现有数据，立即可跑） ----
log "=== Phase A: 写入 data_seed=0 的 90 个任务 ==="
> "${QUEUE_FILE}"
for MSEED in "${MODEL_SEEDS[@]}"; do
  for TLEN in "${TLEN_LIST[@]}"; do
    for RE in "${RE_LIST[@]}"; do
      printf '%s %s %s %s\n' "$RE" "$TLEN" "0" "$MSEED" >> "${QUEUE_FILE}"
    done
  done
done
log "队列写入 $(wc -l < ${QUEUE_FILE}) 个任务"

for GPU in "${GPUS[@]}"; do
  worker "$GPU" &
done
wait
log "=== Phase A 完成 ==="

# ---- Phase B: 生成 data_seed=1,2 ----
log "=== Phase B: 生成 data_seed=1,2（每 Re 用 1 GPU） ==="
# 分配：Re 500,1000 → GPU4,5；Re 2000,3000 → GPU6,7；Re 4000 复用 GPU4
(
  generate_data 500  1 4
  generate_data 1000 1 4
  generate_data 4000 1 4
) &
(
  generate_data 500  2 5
  generate_data 1000 2 5
  generate_data 4000 2 5
) &
(
  generate_data 2000 1 6
  generate_data 3000 1 6
) &
(
  generate_data 2000 2 7
  generate_data 3000 2 7
) &
wait
log "=== Phase B 完成 ==="

# ---- Phase C: data_seed=1,2 的 180 个模型 ----
log "=== Phase C: 写入 data_seed=1,2 的 180 个任务 ==="
> "${QUEUE_FILE}"
for DSEED in 1 2; do
  for MSEED in "${MODEL_SEEDS[@]}"; do
    for TLEN in "${TLEN_LIST[@]}"; do
      for RE in "${RE_LIST[@]}"; do
        printf '%s %s %s %s\n' "$RE" "$TLEN" "$DSEED" "$MSEED" >> "${QUEUE_FILE}"
      done
    done
  done
done
log "队列写入 $(wc -l < ${QUEUE_FILE}) 个任务"

for GPU in "${GPUS[@]}"; do
  worker "$GPU" &
done
wait

log "===== 全部 270 个模型完成 $(date) ====="
