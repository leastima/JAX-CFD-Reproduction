#!/usr/bin/env bash
# 评估 models_ntraj_1k/ 和 models_tlen_1k/ 并生成 phase plot
# 自动跳过未完成的模型（只评估有 checkpoint 的）

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
CSV_NTRAJ="${ROOT}/results/phase_metrics_ntraj_1k.csv"
CSV_TLEN="${ROOT}/results/phase_metrics_tlen_1k.csv"

mkdir -p "${ROOT}/results/ntraj_1k" "${ROOT}/results/tlen_1k"
mkdir -p "${ROOT}/logs/eval_ntraj_1k" "${ROOT}/logs/eval_tlen_1k"
rm -f "${CSV_NTRAJ}" "${CSV_TLEN}"

eval_ntraj() {
    local RE=$1 GPU=$2
    for NTRAJ in 2 8 32; do
        for SEED in 0 1 2; do
            local MODEL_DIR="${ROOT}/models_ntraj_1k/re${RE}_ntraj${NTRAJ}_seed${SEED}"
            local LOG="${ROOT}/logs/eval_ntraj_1k/re${RE}_ntraj${NTRAJ}_seed${SEED}.log"
            ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp" || continue
            echo "[ntraj Re=${RE} ntraj=${NTRAJ} seed=${SEED} GPU${GPU}] eval..."
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
                --model_dir "${MODEL_DIR}" \
                --eval_nc "${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc" \
                --train_nc "${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc" \
                --re ${RE} --ntraj ${NTRAJ} --seed ${SEED} \
                --output_csv "${CSV_NTRAJ}" \
                --length 200 --inner_steps 10 \
                > "${LOG}" 2>&1 \
                && echo "  ✓ ntraj re${RE}_n${NTRAJ}_s${SEED}" \
                || echo "  ✗ ntraj re${RE}_n${NTRAJ}_s${SEED}"
        done
    done
}

eval_tlen() {
    local RE=$1 GPU=$2
    for TLEN in 100 1000 4000; do
        for SEED in 0 1 2; do
            local MODEL_DIR="${ROOT}/models_tlen_1k/re${RE}_tlen${TLEN}_seed${SEED}"
            local LOG="${ROOT}/logs/eval_tlen_1k/re${RE}_tlen${TLEN}_seed${SEED}.log"
            ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp" || continue
            echo "[tlen Re=${RE} tlen=${TLEN} seed=${SEED} GPU${GPU}] eval..."
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
                --model_dir "${MODEL_DIR}" \
                --eval_nc "${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc" \
                --train_nc "${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc" \
                --re ${RE} --ntraj 32 --tlen ${TLEN} --seed ${SEED} \
                --output_csv "${CSV_TLEN}" \
                --length 200 --inner_steps 10 \
                > "${LOG}" 2>&1 \
                && echo "  ✓ tlen re${RE}_t${TLEN}_s${SEED}" \
                || echo "  ✗ tlen re${RE}_t${TLEN}_s${SEED}"
        done
    done
}

echo "=== 评估 ntraj 实验 (GPU4+5+6) ==="
( for RE in 500;        do eval_ntraj ${RE} 4; done ) &
( for RE in 1000;       do eval_ntraj ${RE} 5; done ) &
( for RE in 3000;       do eval_ntraj ${RE} 6; done ) &
wait

echo "=== 评估 tlen 实验 (GPU4+5+6) ==="
( for RE in 500;        do eval_tlen ${RE} 4; done ) &
( for RE in 1000;       do eval_tlen ${RE} 5; done ) &
( for RE in 3000;       do eval_tlen ${RE} 6; done ) &
wait

echo "=== 绘图 ==="
${PYTHON} ${ROOT}/scripts/plot_phase.py \
    --csv "${CSV_NTRAJ}" \
    --output_dir "${ROOT}/results/ntraj_1k/" \
    --yparam ntraj \
    --ylabel "# Training Trajectories" \
    --title "Phase Plot: LI Model vs Re and # Trajectories (1000 steps)"

${PYTHON} ${ROOT}/scripts/plot_phase.py \
    --csv "${CSV_TLEN}" \
    --output_dir "${ROOT}/results/tlen_1k/" \
    --yparam tlen \
    --ylabel "Trajectory Length (frames)" \
    --title "Phase Plot: LI Model vs Re and Trajectory Length (1000 steps, ntraj=32)"

echo "====== 完成！图在 results/ntraj_1k/ 和 results/tlen_1k/ ======"
