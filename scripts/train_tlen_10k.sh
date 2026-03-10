#!/usr/bin/env bash
# tlen phase plot: re={500,1000,3000} × tlen={100,1000,4000} × seed={0,1,2}
# train_steps=10000, ntraj=32 固定, seed=0 优先
# GPU7 一卡：3 个 Re 并行（各用 0.27 显存），每 Re 内 tlen 顺序执行

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.27

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
TRAIN_STEPS=10000
GPU=7

mkdir -p "${ROOT}/logs/train_tlen_10k" "${ROOT}/models_tlen_10k"

# ---- 训练函数 ----
train_tlen() {
    local RE=$1 TLEN=$2 SEED=$3
    local MODEL_DIR="${ROOT}/models_tlen_10k/re${RE}_tlen${TLEN}_seed${SEED}"
    local LOG="${ROOT}/logs/train_tlen_10k/re${RE}_tlen${TLEN}_seed${SEED}.log"
    if ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp"; then
        echo "[SKIP] tlen re${RE}_t${TLEN}_s${SEED}"; return
    fi
    mkdir -p "${MODEL_DIR}"
    echo "[tlen Re=${RE} tlen=${TLEN} seed=${SEED} GPU${GPU}] start $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/models/train.py \
        --gin_file="${ROOT}/models/configs/official_li_config.gin" \
        --gin_file="${ROOT}/models/configs/kolmogorov_forcing.gin" \
        --gin_param="fixed_scale.rescaled_one = 0.2" \
        --gin_param="my_forward_tower_factory.num_hidden_channels = 128" \
        --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
        --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
        --train_split="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc" \
        --eval_split="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc" \
        --train_steps=${TRAIN_STEPS} \
        --train_lr_init=0.001 \
        --train_lr_warmup_epochs=1 \
        --train_init_random_seed=${SEED} \
        --max_train_samples=32 \
        --max_time_steps=${TLEN} \
        --dataset_num_workers=0 \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1
    echo "[tlen Re=${RE} tlen=${TLEN} seed=${SEED}] done ✓ $(date +%H:%M)"
}

# ---- eval + plot 函数 ----
run_eval_plot() {
    echo "===== 开始 eval & plot ($(date)) ====="
    CSV="${ROOT}/results/phase_metrics_tlen_10k.csv"
    mkdir -p "${ROOT}/results/tlen_10k" "${ROOT}/logs/eval_tlen_10k"
    rm -f "${CSV}"
    for RE in 500 1000 3000; do
        for TLEN in 100 1000 4000; do
            for SEED in 0 1 2; do
                MODEL_DIR="${ROOT}/models_tlen_10k/re${RE}_tlen${TLEN}_seed${SEED}"
                ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv tmp || continue
                LOG="${ROOT}/logs/eval_tlen_10k/re${RE}_tlen${TLEN}_seed${SEED}.log"
                CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
                    --model_dir "${MODEL_DIR}" \
                    --eval_nc "${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc" \
                    --train_nc "${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc" \
                    --re ${RE} --seed ${SEED} \
                    --yparam_name tlen --yparam_val ${TLEN} \
                    --output_csv "${CSV}" \
                    --length 200 --inner_steps 10 \
                    > "${LOG}" 2>&1
            done
        done
    done
    ${PYTHON} ${ROOT}/scripts/plot_phase.py \
        --csv "${CSV}" \
        --output_dir "${ROOT}/results/tlen_10k/" \
        --yparam tlen \
        --ylabel "Training Time Length (frames)" \
        --title "Phase Plot: LI Model vs Re and Time Length (10000 steps)"
    echo "===== eval & plot 完成 ====="
}

# ===== 主流程 =====
echo "===== train_tlen_10k.sh 开始 $(date) ====="

for SEED in 0 1 2; do
    echo "--- [seed=${SEED}] ---"
    # 3 个 Re 并行，每 Re 内 tlen 顺序
    (for TLEN in 100 1000 4000; do train_tlen 500  ${TLEN} ${SEED}; done) &
    (for TLEN in 100 1000 4000; do train_tlen 1000 ${TLEN} ${SEED}; done) &
    (for TLEN in 100 1000 4000; do train_tlen 3000 ${TLEN} ${SEED}; done) &
    wait
    echo "--- seed=${SEED} 全部完成 $(date) ---"
    run_eval_plot
done

echo "===== train_tlen_10k.sh 全部完成 $(date) ====="
