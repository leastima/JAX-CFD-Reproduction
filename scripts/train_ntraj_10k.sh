#!/usr/bin/env bash
# ntraj phase plot: re={500,1000,3000} × ntraj={2,8,32} × seed={0,1,2}
# train_steps=10000, seed=0 优先
# GPU4=re500  GPU5=re1000  GPU6=re3000  GPU7=re3000(并行加速 seed=1,2)

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
TRAIN_STEPS=10000

mkdir -p "${ROOT}/logs/train_ntraj_10k" "${ROOT}/models_ntraj_10k"

# ---- 训练函数 ----
train_ntraj() {
    local RE=$1 NTRAJ=$2 SEED=$3 GPU=$4
    local MODEL_DIR="${ROOT}/models_ntraj_10k/re${RE}_ntraj${NTRAJ}_seed${SEED}"
    local LOG="${ROOT}/logs/train_ntraj_10k/re${RE}_ntraj${NTRAJ}_seed${SEED}.log"
    if ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp"; then
        echo "[SKIP] ntraj re${RE}_n${NTRAJ}_s${SEED}"; return
    fi
    mkdir -p "${MODEL_DIR}"
    echo "[ntraj Re=${RE} ntraj=${NTRAJ} seed=${SEED} GPU${GPU}] start $(date +%H:%M)"
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
        --max_train_samples=${NTRAJ} \
        --dataset_num_workers=0 \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1
    echo "[ntraj Re=${RE} ntraj=${NTRAJ} seed=${SEED}] done ✓ $(date +%H:%M)"
}

# ---- eval + plot 函数 ----
run_eval_plot() {
    echo "===== 开始 eval & plot ($(date)) ====="
    CSV="${ROOT}/results/phase_metrics_ntraj_10k.csv"
    mkdir -p "${ROOT}/results/ntraj_10k" "${ROOT}/logs/eval_ntraj_10k"
    rm -f "${CSV}"
    for RE in 500 1000 3000; do
        EVAL_GPU=$((RE==500 ? 4 : RE==1000 ? 5 : 6))
        for NTRAJ in 2 8 32; do
            for SEED in 0 1 2; do
                MODEL_DIR="${ROOT}/models_ntraj_10k/re${RE}_ntraj${NTRAJ}_seed${SEED}"
                ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv tmp || continue
                LOG="${ROOT}/logs/eval_ntraj_10k/re${RE}_ntraj${NTRAJ}_seed${SEED}.log"
                CUDA_VISIBLE_DEVICES=${EVAL_GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
                    --model_dir "${MODEL_DIR}" \
                    --eval_nc "${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc" \
                    --train_nc "${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc" \
                    --re ${RE} --ntraj ${NTRAJ} --seed ${SEED} \
                    --output_csv "${CSV}" \
                    --length 200 --inner_steps 10 \
                    > "${LOG}" 2>&1 &
            done
        done
        wait
    done
    ${PYTHON} ${ROOT}/scripts/plot_phase.py \
        --csv "${CSV}" \
        --output_dir "${ROOT}/results/ntraj_10k/" \
        --yparam ntraj \
        --ylabel "# Training Trajectories" \
        --title "Phase Plot: LI Model vs Re and # Trajectories (10000 steps)"
    echo "===== eval & plot 完成 ====="
}

# ===== 主流程：seed=0 优先 =====
echo "===== train_ntraj_10k.sh 开始 $(date) ====="

# --- seed=0（最高优先级，3 Re 并行） ---
echo "--- [seed=0] ---"
for NTRAJ in 2 8 32; do
    train_ntraj 500  ${NTRAJ} 0 4
done &
for NTRAJ in 2 8 32; do
    train_ntraj 1000 ${NTRAJ} 0 5
done &
for NTRAJ in 2 8 32; do
    train_ntraj 3000 ${NTRAJ} 0 6
done &
wait
echo "--- seed=0 全部完成 $(date), 生成初步 phase plot ---"
run_eval_plot

# --- seed=1（GPU4/5/6/7 四卡并行，每卡一个 Re+ntraj 组合） ---
echo "--- [seed=1] ---"
for NTRAJ in 2 8 32; do
    train_ntraj 500  ${NTRAJ} 1 4
done &
for NTRAJ in 2 8 32; do
    train_ntraj 1000 ${NTRAJ} 1 5
done &
for NTRAJ in 2 8 32; do
    train_ntraj 3000 ${NTRAJ} 1 6
done &
for NTRAJ in 2 8 32; do
    train_ntraj 3000 ${NTRAJ} 1 7   # GPU7 加速 re=3000（与 GPU6 竞争同目录，skip 已完成的）
done &
wait

# --- seed=2 ---
echo "--- [seed=2] ---"
for NTRAJ in 2 8 32; do
    train_ntraj 500  ${NTRAJ} 2 4
done &
for NTRAJ in 2 8 32; do
    train_ntraj 1000 ${NTRAJ} 2 5
done &
for NTRAJ in 2 8 32; do
    train_ntraj 3000 ${NTRAJ} 2 6
done &
wait

echo "===== 全部训练完成，最终 eval & plot ====="
run_eval_plot
echo "===== train_ntraj_10k.sh 全部完成 $(date) ====="
