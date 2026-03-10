#!/usr/bin/env bash
# 固定 1000 steps 训练实验
# ntraj: re={500,1000,3000} × ntraj={2,8,32} × seed={0,1,2}  → models_ntraj_1k/
# tlen:  re={500,1000,3000} × tlen={100,1000,4000} × seed={0,1,2} → models_tlen_1k/
#
# GPU 分配：GPU4=re500  GPU5=re1000  GPU6=re3000
# seed=0 优先：每个 Re 先跑完所有 seed=0 后再跑 seed=1,2
# seed=0 完成预计: ~15.7h | 全部完成: ~47h

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
TRAIN_STEPS=1000

mkdir -p "${ROOT}/logs/train_ntraj_1k" "${ROOT}/logs/train_tlen_1k"
mkdir -p "${ROOT}/models_ntraj_1k" "${ROOT}/models_tlen_1k"

# ---- 训练函数 ----
train_ntraj() {
    local RE=$1 NTRAJ=$2 SEED=$3 GPU=$4
    local MODEL_DIR="${ROOT}/models_ntraj_1k/re${RE}_ntraj${NTRAJ}_seed${SEED}"
    local LOG="${ROOT}/logs/train_ntraj_1k/re${RE}_ntraj${NTRAJ}_seed${SEED}.log"
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
        --train_init_random_seed=${SEED} \
        --max_train_samples=${NTRAJ} \
        --dataset_num_workers=0 \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1
    echo "[ntraj Re=${RE} ntraj=${NTRAJ} seed=${SEED}] done ✓ $(date +%H:%M)"
}

train_tlen() {
    local RE=$1 TLEN=$2 SEED=$3 GPU=$4
    local MODEL_DIR="${ROOT}/models_tlen_1k/re${RE}_tlen${TLEN}_seed${SEED}"
    local LOG="${ROOT}/logs/train_tlen_1k/re${RE}_tlen${TLEN}_seed${SEED}.log"
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
        --train_init_random_seed=${SEED} \
        --max_train_samples=32 \
        --max_time_steps=${TLEN} \
        --dataset_num_workers=0 \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1
    echo "[tlen Re=${RE} tlen=${TLEN} seed=${SEED}] done ✓ $(date +%H:%M)"
}

# ---- eval + plot 函数（seed=0 跑完后自动触发） ----
run_eval_plot() {
    echo "=== [$(date +%H:%M)] seed=0 全部完成，开始评估 ==="
    bash "${ROOT}/scripts/eval_1ksteps.sh"
}

# ---- 每个 GPU 处理一个 Re，seed=0 优先 ----
run_re() {
    local RE=$1 GPU=$2
    # seed=0 优先
    for SEED in 0 1 2; do
        for NTRAJ in 2 8 32; do
            train_ntraj ${RE} ${NTRAJ} ${SEED} ${GPU}
        done
        for TLEN in 100 1000 4000; do
            train_tlen ${RE} ${TLEN} ${SEED} ${GPU}
        done
        if [ ${SEED} -eq 0 ]; then
            echo "=== [$(date +%H:%M)] Re=${RE} seed=0 完成 ===" >> "${ROOT}/logs/seed0_done.log"
        fi
    done
    echo "=== GPU${GPU}/Re=${RE} 全部完成 $(date +%H:%M) ==="
}

echo "====== 1000-steps 实验启动 $(date) ======"
echo "  GPU4: re=500  (ntraj+tlen 顺序)"
echo "  GPU5: re=1000 (ntraj+tlen 顺序)"
echo "  GPU6: re=3000 ntraj 专用"
echo "  GPU7: re=3000 tlen  专用"
echo "  预计 seed=0 完成: ~15.7h 后 | 全部完成: ~47h 后"

# GPU4: re=500 ntraj → tlen
( run_re 500  4 ) &

# GPU5: re=1000 ntraj → tlen
( run_re 1000 5 ) &

# GPU6: re=3000 ntraj 专用（只跑 ntraj）
(
    for SEED in 0 1 2; do
        for NTRAJ in 2 8 32; do
            train_ntraj 3000 ${NTRAJ} ${SEED} 6
        done
        echo "=== [$(date +%H:%M)] re=3000 ntraj seed=${SEED} 完成 ===" >> "${ROOT}/logs/seed0_done.log"
    done
    echo "=== GPU6/re=3000 ntraj 全部完成 $(date +%H:%M) ==="
) &

# GPU7: re=3000 tlen 专用（只跑 tlen）
(
    for SEED in 0 1 2; do
        for TLEN in 100 1000 4000; do
            train_tlen 3000 ${TLEN} ${SEED} 7
        done
    done
    echo "=== GPU7/re=3000 tlen 全部完成 $(date +%H:%M) ==="
) &

wait
echo "====== 全部训练完成 $(date) ======"
bash "${ROOT}/scripts/eval_1ksteps.sh"
