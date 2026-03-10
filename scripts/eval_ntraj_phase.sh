#!/usr/bin/env bash
# 评估 models_phase (ntraj 实验, re=500/1000/2000/3000) 并画 phase plot
# GPU4=re500+re1000, GPU5=re2000+re3000（避免干扰 tlen4000 训练）

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
CSV="${ROOT}/results/phase_metrics_ntraj_phase.csv"

mkdir -p "${ROOT}/results/ntraj_phase" "${ROOT}/logs/eval_ntraj_phase"
rm -f "${CSV}"

eval_one() {
    local RE=$1 NTRAJ=$2 SEED=$3 GPU=$4
    local MODEL_DIR="${ROOT}/models_phase/re${RE}_ntraj${NTRAJ}_seed${SEED}"
    local EVAL_NC="${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc"
    local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
    local LOG="${ROOT}/logs/eval_ntraj_phase/re${RE}_ntraj${NTRAJ}_seed${SEED}.log"

    ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp" || { echo "[SKIP no ckpt] re${RE}_ntraj${NTRAJ}_seed${SEED}"; return; }

    echo "[Re=${RE} ntraj=${NTRAJ} seed=${SEED} GPU${GPU}] evaluating..."
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
        --model_dir "${MODEL_DIR}" \
        --eval_nc   "${EVAL_NC}" \
        --train_nc  "${TRAIN_NC}" \
        --re ${RE} --ntraj ${NTRAJ} --seed ${SEED} \
        --output_csv "${CSV}" \
        --length 200 --inner_steps 10 \
        > "${LOG}" 2>&1 \
        && echo "  ✓ re${RE}_ntraj${NTRAJ}_seed${SEED}" \
        || echo "  ✗ re${RE}_ntraj${NTRAJ}_seed${SEED} (check ${LOG})"
}

# GPU4: re500, re1000 串行
(
    for RE in 500 1000; do
        for NTRAJ in 2 8 16 32; do
            for SEED in 0 1 2; do
                eval_one ${RE} ${NTRAJ} ${SEED} 4
            done
        done
    done
    echo "=== GPU4 done ==="
) &

# GPU5: re2000, re3000 串行
(
    for RE in 2000 3000; do
        for NTRAJ in 2 8 16 32; do
            for SEED in 0 1 2; do
                eval_one ${RE} ${NTRAJ} ${SEED} 5
            done
        done
    done
    echo "=== GPU5 done ==="
) &

echo "====== 评估已启动 (48 模型, GPU4+GPU5) ======"
wait

echo "=== 绘制 ntraj phase plot ==="
${PYTHON} ${ROOT}/scripts/plot_phase.py \
    --csv "${CSV}" \
    --output_dir "${ROOT}/results/ntraj_phase/" \
    --yparam ntraj \
    --ylabel "# Training Trajectories" \
    --title "Phase Plot: LI Model vs Re and # Trajectories (1 epoch)"

echo "====== 完成！图在 ${ROOT}/results/ntraj_phase/ ======"
