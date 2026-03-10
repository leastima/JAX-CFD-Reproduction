#!/usr/bin/env bash
# 评估 models_200k / models_tlen_200k 并生成 phase plot
# 用法:
#   bash scripts/eval_and_plot_200k.sh ntraj   # 评估 ntraj 实验
#   bash scripts/eval_and_plot_200k.sh tlen    # 评估 tlen 实验
#   bash scripts/eval_and_plot_200k.sh both    # 两个都评估

set -eo pipefail

EXPERIMENT=${1:-both}

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"

# train save_dt=0.007012, long_eval save_dt=0.07012（10倍子采样）
# → 推理时每输出帧需走 10 个 solver 步才能对齐 eval 时间轴
INNER_STEPS=10

eval_ntraj() {
    local GPU=$1 RE=$2
    for NTRAJ in 2 4 8 16 32; do
        for SEED in 0; do
            local MODEL_DIR="${ROOT}/models_200k/re${RE}_ntraj${NTRAJ}_seed${SEED}"
            local EVAL_NC="${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc"
            local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
            local LOG="${ROOT}/logs/eval_ntraj_200k/re${RE}_ntraj${NTRAJ}_seed${SEED}.log"
            ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp" || { echo "✗ SKIP (no ckpt) ntraj re${RE}_n${NTRAJ}_s${SEED}"; continue; }
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
                --model_dir "${MODEL_DIR}" --eval_nc "${EVAL_NC}" --train_nc "${TRAIN_NC}" \
                --re ${RE} --ntraj ${NTRAJ} --seed ${SEED} \
                --output_csv "${ROOT}/results/phase_metrics_ntraj_200k.csv" \
                --length 200 --inner_steps ${INNER_STEPS} > "${LOG}" 2>&1 \
                && echo "✓ ntraj re${RE}_n${NTRAJ}_s${SEED}" || echo "✗ ntraj re${RE}_n${NTRAJ}_s${SEED}"
        done
    done
}

eval_tlen() {
    local GPU=$1 RE=$2
    for TLEN in 100 500 2000 4000; do
        for SEED in 0; do
            local MODEL_DIR="${ROOT}/models_tlen_200k/re${RE}_tlen${TLEN}_seed${SEED}"
            local EVAL_NC="${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc"
            local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
            local LOG="${ROOT}/logs/eval_tlen_200k/re${RE}_tlen${TLEN}_seed${SEED}.log"
            ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp" || { echo "✗ SKIP (no ckpt) tlen re${RE}_t${TLEN}_s${SEED}"; continue; }
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
                --model_dir "${MODEL_DIR}" --eval_nc "${EVAL_NC}" --train_nc "${TRAIN_NC}" \
                --re ${RE} --ntraj 32 --tlen ${TLEN} --seed ${SEED} \
                --output_csv "${ROOT}/results/phase_metrics_tlen_200k.csv" \
                --length 200 --inner_steps ${INNER_STEPS} > "${LOG}" 2>&1 \
                && echo "✓ tlen re${RE}_t${TLEN}_s${SEED}" || echo "✗ tlen re${RE}_t${TLEN}_s${SEED}"
        done
    done
}

mkdir -p "${ROOT}/results" \
         "${ROOT}/logs/eval_ntraj_200k" \
         "${ROOT}/logs/eval_tlen_200k" \
         "${ROOT}/results/ntraj_200k" \
         "${ROOT}/results/tlen_200k"

if [[ "$EXPERIMENT" == "ntraj" || "$EXPERIMENT" == "both" ]]; then
    echo "=== 评估 ntraj 实验 (models_200k) ==="
    rm -f "${ROOT}/results/phase_metrics_ntraj_200k.csv"
    ( for RE in 500 1000; do eval_ntraj 4 ${RE}; done ) &
    ( for RE in 2000 3000; do eval_ntraj 5 ${RE}; done ) &
    wait
    echo "=== 绘制 ntraj phase plot ==="
    ${PYTHON} ${ROOT}/scripts/plot_phase.py \
        --csv "${ROOT}/results/phase_metrics_ntraj_200k.csv" \
        --output_dir "${ROOT}/results/ntraj_200k/" \
        --yparam ntraj \
        --ylabel "# Training Trajectories" \
        --title "Phase Plot: LI Model vs Re and # Trajectories (50k steps, seed=0)"
    echo "ntraj phase plot → ${ROOT}/results/ntraj_200k/phase_plot.png"
fi

if [[ "$EXPERIMENT" == "tlen" || "$EXPERIMENT" == "both" ]]; then
    echo "=== 评估 tlen 实验 (models_tlen_200k) ==="
    rm -f "${ROOT}/results/phase_metrics_tlen_200k.csv"
    ( for RE in 500 1000; do eval_tlen 6 ${RE}; done ) &
    ( for RE in 2000 3000; do eval_tlen 7 ${RE}; done ) &
    wait
    echo "=== 绘制 tlen phase plot ==="
    ${PYTHON} ${ROOT}/scripts/plot_phase.py \
        --csv "${ROOT}/results/phase_metrics_tlen_200k.csv" \
        --output_dir "${ROOT}/results/tlen_200k/" \
        --yparam tlen \
        --ylabel "Trajectory Length (frames)" \
        --title "Phase Plot: LI Model vs Re and Training Traj Length (50k steps, ntraj=32, seed=0)"
    echo "tlen phase plot → ${ROOT}/results/tlen_200k/phase_plot.png"
fi

echo "====== 评估与绘图全部完成 ======"
