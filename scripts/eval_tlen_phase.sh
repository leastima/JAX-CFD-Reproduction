#!/usr/bin/env bash
# 评估 models_tlen_1ep (tlen 实验, re=500/1000/2000/3000, seed=0) 并画 phase plot
# GPU6=re500+re1000, GPU7=re2000+re3000

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
CSV="${ROOT}/results/phase_metrics_tlen_1ep.csv"

mkdir -p "${ROOT}/results/tlen_phase" "${ROOT}/logs/eval_tlen_phase"
rm -f "${CSV}"

eval_one() {
    local RE=$1 TLEN=$2 SEED=$3 GPU=$4
    local MODEL_DIR="${ROOT}/models_tlen_1ep/re${RE}_tlen${TLEN}_seed${SEED}"
    local EVAL_NC="${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc"
    local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
    local LOG="${ROOT}/logs/eval_tlen_phase/re${RE}_tlen${TLEN}_seed${SEED}.log"

    ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp" || { echo "[SKIP no ckpt] re${RE}_tlen${TLEN}_seed${SEED}"; return; }

    echo "[Re=${RE} tlen=${TLEN} seed=${SEED} GPU${GPU}] evaluating..."
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
        --model_dir "${MODEL_DIR}" \
        --eval_nc   "${EVAL_NC}" \
        --train_nc  "${TRAIN_NC}" \
        --re ${RE} --ntraj 32 --tlen ${TLEN} --seed ${SEED} \
        --output_csv "${CSV}" \
        --length 200 --inner_steps 10 \
        > "${LOG}" 2>&1 \
        && echo "  ✓ re${RE}_tlen${TLEN}_seed${SEED}" \
        || echo "  ✗ re${RE}_tlen${TLEN}_seed${SEED} (check ${LOG})"
}

# GPU6: re500, re1000 串行
(
    for RE in 500 1000; do
        for TLEN in 100 500 1000 2000 4000; do
            eval_one ${RE} ${TLEN} 0 6
        done
    done
    echo "=== GPU6 done ==="
) &

# GPU7: re2000, re3000 串行
(
    for RE in 2000 3000; do
        for TLEN in 100 500 1000 2000 4000; do
            eval_one ${RE} ${TLEN} 0 7
        done
    done
    echo "=== GPU7 done ==="
) &

echo "====== tlen 评估已启动 (20 模型, GPU6+GPU7) ======"
wait

echo "=== 绘制 tlen phase plot ==="
${PYTHON} ${ROOT}/scripts/plot_phase.py \
    --csv "${CSV}" \
    --output_dir "${ROOT}/results/tlen_phase/" \
    --yparam tlen \
    --ylabel "Trajectory Length (frames)" \
    --title "Phase Plot: LI Model vs Re and Training Trajectory Length (1 epoch, ntraj=32)"

echo "====== 完成！图在 ${ROOT}/results/tlen_phase/ ======"
