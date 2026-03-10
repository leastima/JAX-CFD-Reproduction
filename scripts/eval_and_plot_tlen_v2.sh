#!/usr/bin/env bash
# 评估 models_tlen_v2 所有已完成模型 → 更新 results/tlen_v2/phase_plot.png
# 幂等：跳过已经有结果的 (re, tlen, dseed, mseed) 组合

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
INNER_STEPS=10

RE_LIST=(500 1000 2000 3000 4000)
TLEN_LIST=(50 100 500 1000 2000 4000)
DATA_SEEDS=(0 1 2)
MODEL_SEEDS=(0 1 2)

CSV="${ROOT}/results/tlen_v2/phase_metrics.csv"
OUT_DIR="${ROOT}/results/tlen_v2"

mkdir -p "${OUT_DIR}" "${ROOT}/logs/eval_tlen_v2"
# 不删旧 CSV，追加模式（eval_one_model.py 自带追加逻辑）
# 但若想重新生成，手动删除 phase_metrics.csv 即可

eval_batch() {
  local GPU=$1
  shift
  for spec in "$@"; do
    read -r RE TLEN DSEED MSEED <<< "${spec//_/ }"
    local MODEL_DIR="${ROOT}/models_tlen_v2/re${RE}_tlen${TLEN}_dseed${DSEED}_mseed${MSEED}"
    local EVAL_NC="${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc"
    local TRAIN_NC
    if [[ $DSEED -eq 0 ]]; then
      TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
    else
      TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64_dseed${DSEED}.nc"
    fi
    local LOG="${ROOT}/logs/eval_tlen_v2/re${RE}_tlen${TLEN}_d${DSEED}_m${MSEED}.log"
    ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp" || {
      echo "  SKIP (no ckpt): re${RE}_tlen${TLEN}_d${DSEED}_m${MSEED}"; continue
    }
    # 检查 CSV 里是否已有这条记录（避免重复 eval）
    if [[ -f "${CSV}" ]] && grep -q "^${RE},${TLEN},${MSEED},${DSEED}," "${CSV}" 2>/dev/null; then
      echo "  CACHED: re${RE}_tlen${TLEN}_d${DSEED}_m${MSEED}"; continue
    fi
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u "${ROOT}/scripts/eval_one_model.py" \
      --model_dir "${MODEL_DIR}" \
      --eval_nc   "${EVAL_NC}" \
      --train_nc  "${TRAIN_NC}" \
      --re ${RE} --ntraj 32 --tlen ${TLEN} --seed ${MSEED} --data_seed ${DSEED} \
      --output_csv "${CSV}" \
      --length 200 --inner_steps ${INNER_STEPS} \
      > "${LOG}" 2>&1 \
      && echo "  OK: re${RE}_tlen${TLEN}_d${DSEED}_m${MSEED}" \
      || echo "  ERR: re${RE}_tlen${TLEN}_d${DSEED}_m${MSEED} (see ${LOG})"
  done
}

# 把所有已完成的模型分成 4 批，分配到 GPU 4-7
echo "=== 收集已完成模型 ==="
ALL_SPECS=()
for RE in "${RE_LIST[@]}"; do
  for TLEN in "${TLEN_LIST[@]}"; do
    for DSEED in "${DATA_SEEDS[@]}"; do
      for MSEED in "${MODEL_SEEDS[@]}"; do
        MODEL_DIR="${ROOT}/models_tlen_v2/re${RE}_tlen${TLEN}_dseed${DSEED}_mseed${MSEED}"
        ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp" \
          && ALL_SPECS+=("${RE}_${TLEN}_${DSEED}_${MSEED}") || true
      done
    done
  done
done
echo "  共 ${#ALL_SPECS[@]} 个模型需要 eval"

# 按 GPU 分配（轮询）
declare -a BATCH4 BATCH5 BATCH6 BATCH7
for i in "${!ALL_SPECS[@]}"; do
  case $(( i % 4 )) in
    0) BATCH4+=("${ALL_SPECS[$i]}") ;;
    1) BATCH5+=("${ALL_SPECS[$i]}") ;;
    2) BATCH6+=("${ALL_SPECS[$i]}") ;;
    3) BATCH7+=("${ALL_SPECS[$i]}") ;;
  esac
done

( eval_batch 4 "${BATCH4[@]}" ) &
( eval_batch 5 "${BATCH5[@]}" ) &
( eval_batch 6 "${BATCH6[@]}" ) &
( eval_batch 7 "${BATCH7[@]}" ) &
wait

echo "=== 绘制 phase plot (6 个指标) ==="
${PYTHON} "${ROOT}/scripts/plot_phase.py" \
  --csv "${CSV}" \
  --output_dir "${OUT_DIR}" \
  --yparam tlen \
  --ylabel "Trajectory Length (frames)" \
  --title "tlen v2: Re vs tlen (3x3 seeds, 50k steps, early_stop 1e-6)"

echo "Phase plot → ${OUT_DIR}/phase_plot.png"
