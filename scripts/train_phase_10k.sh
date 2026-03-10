#!/usr/bin/env bash
# Phase plot: ntraj + tlen 两个实验
# 5000 steps, LR=1e-3, 4 进程/GPU, GPU4-7
# seed=0 优先 → eval+plot → seed=1,2

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.22

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
TRAIN_STEPS=5000

mkdir -p "${ROOT}/logs/train_phase_5k" "${ROOT}/models_ntraj_5k" "${ROOT}/models_tlen_5k"

# ---- 通用训练函数 ----
train_model() {
    local TYPE=$1 RE=$2 PARAM=$3 SEED=$4 GPU=$5
    local MODEL_DIR PARAM_FLAG LOG
    if [[ "${TYPE}" == "ntraj" ]]; then
        MODEL_DIR="${ROOT}/models_ntraj_5k/re${RE}_ntraj${PARAM}_seed${SEED}"
        PARAM_FLAG="--max_train_samples=${PARAM}"
    else
        MODEL_DIR="${ROOT}/models_tlen_5k/re${RE}_tlen${PARAM}_seed${SEED}"
        PARAM_FLAG="--max_train_samples=32 --max_time_steps=${PARAM}"
    fi
    LOG="${ROOT}/logs/train_phase_5k/${TYPE}_re${RE}_p${PARAM}_s${SEED}.log"

    if ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp"; then
        echo "[SKIP] ${TYPE} re${RE} p${PARAM} s${SEED}"; return
    fi
    mkdir -p "${MODEL_DIR}"
    echo "[${TYPE} Re=${RE} p=${PARAM} s=${SEED} GPU${GPU}] $(date +%H:%M)"
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
        --dataset_num_workers=0 \
        ${PARAM_FLAG} \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1
    echo "[${TYPE} Re=${RE} p=${PARAM} s=${SEED}] done ✓ $(date +%H:%M)"
}

# ---- eval + plot ----
run_eval_plot() {
    local SEED_LABEL=$1
    echo "===== eval & plot (${SEED_LABEL}) $(date) ====="
    CSV_N="${ROOT}/results/phase_metrics_ntraj_5k.csv"
    CSV_T="${ROOT}/results/phase_metrics_tlen_5k.csv"
    [[ "$SEED_LABEL" == "seed0" ]] && rm -f "${CSV_N}" "${CSV_T}"
    mkdir -p "${ROOT}/results/ntraj_5k" "${ROOT}/results/tlen_5k" "${ROOT}/logs/eval_phase_5k"

    for RE in 500 1000 3000; do
        EVAL_GPU=$((RE==500 ? 4 : RE==1000 ? 5 : 6))
        for NTRAJ in 2 8 32; do
            for SEED in 0 1 2; do
                D="${ROOT}/models_ntraj_5k/re${RE}_ntraj${NTRAJ}_seed${SEED}"
                ls "${D}"/checkpoint_* 2>/dev/null | grep -qv tmp || continue
                CUDA_VISIBLE_DEVICES=${EVAL_GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
                    --model_dir "${D}" \
                    --eval_nc "${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc" \
                    --train_nc "${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc" \
                    --re ${RE} --ntraj ${NTRAJ} --seed ${SEED} \
                    --output_csv "${CSV_N}" --length 200 --inner_steps 10 \
                    >> "${ROOT}/logs/eval_phase_5k/ntraj_re${RE}_n${NTRAJ}_s${SEED}.log" 2>&1 &
            done
        done
        wait
        for TLEN in 100 1000 4000; do
            for SEED in 0 1 2; do
                D="${ROOT}/models_tlen_5k/re${RE}_tlen${TLEN}_seed${SEED}"
                ls "${D}"/checkpoint_* 2>/dev/null | grep -qv tmp || continue
                CUDA_VISIBLE_DEVICES=${EVAL_GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
                    --model_dir "${D}" \
                    --eval_nc "${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc" \
                    --train_nc "${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc" \
                    --re ${RE} --seed ${SEED} \
                    --yparam_name tlen --yparam_val ${TLEN} \
                    --output_csv "${CSV_T}" --length 200 --inner_steps 10 \
                    >> "${ROOT}/logs/eval_phase_5k/tlen_re${RE}_t${TLEN}_s${SEED}.log" 2>&1 &
            done
        done
        wait
    done

    [[ -f "${CSV_N}" ]] && ${PYTHON} ${ROOT}/scripts/plot_phase.py \
        --csv "${CSV_N}" --output_dir "${ROOT}/results/ntraj_5k/" \
        --yparam ntraj --ylabel "# Training Trajectories" \
        --title "Phase Plot: ntraj (5000 steps, LR=1e-3, ${SEED_LABEL})"
    [[ -f "${CSV_T}" ]] && ${PYTHON} ${ROOT}/scripts/plot_phase.py \
        --csv "${CSV_T}" --output_dir "${ROOT}/results/tlen_5k/" \
        --yparam tlen --ylabel "Training Time Length (frames)" \
        --title "Phase Plot: tlen (5000 steps, LR=1e-3, ${SEED_LABEL})"
    echo "===== eval & plot 完成 ====="
}

# ===== 主流程 =====
echo "===== train_phase_10k.sh 开始 $(date) ====="

for SEED in 0 1 2; do
    echo "===== [seed=${SEED}] ====="

    # GPU4: re=500 ntraj 全部 + tlen100
    # GPU5: re=1000 ntraj 全部 + tlen100
    # GPU6: re=3000 ntraj 全部 + tlen100
    # GPU7: 全部 tlen1000 + tlen4000（3 Re，3 tlen 各一个，共 6 个，顺序分 2 batch）
    (
        train_model ntraj 500  2    ${SEED} 4 &
        train_model ntraj 500  8    ${SEED} 4 &
        train_model ntraj 500  32   ${SEED} 4 &
        train_model tlen  500  100  ${SEED} 4 &
        wait
        train_model tlen  500  1000 ${SEED} 4 &
        train_model tlen  500  4000 ${SEED} 4 &
        wait
    ) &

    (
        train_model ntraj 1000 2    ${SEED} 5 &
        train_model ntraj 1000 8    ${SEED} 5 &
        train_model ntraj 1000 32   ${SEED} 5 &
        train_model tlen  1000 100  ${SEED} 5 &
        wait
        train_model tlen  1000 1000 ${SEED} 5 &
        train_model tlen  1000 4000 ${SEED} 5 &
        wait
    ) &

    (
        train_model ntraj 3000 2    ${SEED} 6 &
        train_model ntraj 3000 8    ${SEED} 6 &
        train_model ntraj 3000 32   ${SEED} 6 &
        train_model tlen  3000 100  ${SEED} 6 &
        wait
        train_model tlen  3000 1000 ${SEED} 6 &
        train_model tlen  3000 4000 ${SEED} 6 &
        wait
    ) &

    wait
    echo "===== seed=${SEED} 全部完成 $(date) ====="
    run_eval_plot "seed${SEED}"
done

echo "===== 全部完成 $(date) ====="
