#!/usr/bin/env bash
# 评估已完成的模型并生成 phase plot
# 用法:
#   bash scripts/eval_and_plot.sh ntraj   # 评估实验1(ntraj)
#   bash scripts/eval_and_plot.sh tlen    # 评估实验2(time_length)
#   bash scripts/eval_and_plot.sh both    # 两个都评估

set -eo pipefail

EXPERIMENT=${1:-both}

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"

eval_ntraj() {
    local GPU=$1 RE=$2
    for NTRAJ in 2 8 16 32; do
        for SEED in 0 1 2; do
            local MODEL_DIR="${ROOT}/models_ntraj/re${RE}_ntraj${NTRAJ}_seed${SEED}"
            local EVAL_NC="${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc"
            local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
            local LOG="${ROOT}/logs/eval_ntraj/re${RE}_ntraj${NTRAJ}_seed${SEED}.log"
            ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp" || continue
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
                --model_dir "${MODEL_DIR}" --eval_nc "${EVAL_NC}" --train_nc "${TRAIN_NC}" \
                --re ${RE} --ntraj ${NTRAJ} --seed ${SEED} \
                --output_csv "${ROOT}/results/phase_metrics_ntraj.csv" \
                --length 200 --inner_steps 10 > "${LOG}" 2>&1 \
                && echo "✓ ntraj re${RE}_n${NTRAJ}_s${SEED}" || echo "✗ ntraj re${RE}_n${NTRAJ}_s${SEED}"
        done
    done
}

eval_tlen() {
    local GPU=$1 RE=$2
    for TLEN in 100 500 1000 2000 4000; do
        for SEED in 0 1 2; do
            local MODEL_DIR="${ROOT}/models_tlen/re${RE}_tlen${TLEN}_seed${SEED}"
            local EVAL_NC="${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc"
            local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
            local LOG="${ROOT}/logs/eval_tlen/re${RE}_tlen${TLEN}_seed${SEED}.log"
            ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp" || continue
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
                --model_dir "${MODEL_DIR}" --eval_nc "${EVAL_NC}" --train_nc "${TRAIN_NC}" \
                --re ${RE} --ntraj 32 --tlen ${TLEN} --seed ${SEED} \
                --output_csv "${ROOT}/results/phase_metrics_tlen.csv" \
                --length 200 --inner_steps 10 > "${LOG}" 2>&1 \
                && echo "✓ tlen re${RE}_t${TLEN}_s${SEED}" || echo "✗ tlen re${RE}_t${TLEN}_s${SEED}"
        done
    done
}

mkdir -p "${ROOT}/results" "${ROOT}/logs/eval_ntraj" "${ROOT}/logs/eval_tlen"

if [[ "$EXPERIMENT" == "ntraj" || "$EXPERIMENT" == "both" ]]; then
    echo "=== 评估实验1 (ntraj) ==="
    rm -f "${ROOT}/results/phase_metrics_ntraj.csv"
    ( for RE in 500 1000; do eval_ntraj 4 ${RE}; done ) &
    ( for RE in 2000 3000 4000; do eval_ntraj 5 ${RE}; done ) &
    wait
    echo "=== 绘制 ntraj phase plot ==="
    ${PYTHON} ${ROOT}/scripts/plot_phase.py \
        --csv "${ROOT}/results/phase_metrics_ntraj.csv" \
        --output_dir "${ROOT}/results/ntraj/" \
        --yparam ntraj \
        --ylabel "# Training Trajectories" \
        --title "Phase Plot: LI Model vs Re and # Trajectories (5 epochs)"
fi

if [[ "$EXPERIMENT" == "tlen" || "$EXPERIMENT" == "both" ]]; then
    echo "=== 评估实验2 (tlen) ==="
    rm -f "${ROOT}/results/phase_metrics_tlen.csv"
    ( for RE in 500 1000; do eval_tlen 6 ${RE}; done ) &
    ( for RE in 2000 3000 4000; do eval_tlen 7 ${RE}; done ) &
    wait
    echo "=== 绘制 tlen phase plot ==="
    ${PYTHON} ${ROOT}/scripts/plot_phase.py \
        --csv "${ROOT}/results/phase_metrics_tlen.csv" \
        --output_dir "${ROOT}/results/tlen/" \
        --yparam tlen \
        --ylabel "Trajectory Length (frames)" \
        --title "Phase Plot: LI Model vs Re and Training Trajectory Length (5 epochs, ntraj=32)"
fi

echo "====== 评估与绘图全部完成 ======"
echo "结果:"
echo "  ntraj plot: ${ROOT}/results/ntraj/phase_plot.png"
echo "  tlen  plot: ${ROOT}/results/tlen/phase_plot.png"
