#!/usr/bin/env bash
# tlen phase plot: 1 epoch, re=500/1000/2000/3000, seed=0, tlen=100/500/1000/2000/4000
# GPU4=re500, GPU5=re1000, GPU6=re2000, GPU7=re3000 (各跑所有 tlen 串行)
# 输出目录: models_tlen_1ep/

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80  # 每 GPU 仅 1 进程

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
mkdir -p "${ROOT}/logs/train_tlen_1ep"

train_one() {
    local RE=$1 TLEN=$2 SEED=$3 GPU=$4
    local MODEL_DIR="${ROOT}/models_tlen_1ep/re${RE}_tlen${TLEN}_seed${SEED}"
    local LOG="${ROOT}/logs/train_tlen_1ep/re${RE}_tlen${TLEN}_seed${SEED}.log"

    if ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp"; then
        echo "[SKIP] re${RE}_tlen${TLEN}_seed${SEED}"; return
    fi
    mkdir -p "${MODEL_DIR}"
    echo "[Re=${RE} tlen=${TLEN} seed=${SEED} GPU${GPU}] start $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/models/train.py \
        --gin_file="${ROOT}/models/configs/official_li_config.gin" \
        --gin_file="${ROOT}/models/configs/kolmogorov_forcing.gin" \
        --gin_param="fixed_scale.rescaled_one = 0.2" \
        --gin_param="my_forward_tower_factory.num_hidden_channels = 128" \
        --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
        --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
        --train_split="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc" \
        --eval_split="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc" \
        --train_epochs=1 \
        --train_init_random_seed=${SEED} \
        --max_train_samples=32 \
        --max_time_steps=${TLEN} \
        --dataset_num_workers=0 \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1
    echo "[Re=${RE} tlen=${TLEN} seed=${SEED}] done ✓ $(date +%H:%M)"
}

# 4 块 GPU 各负责 1 个 Re，串行跑所有 tlen，seed=0
(
    for TLEN in 100 500 1000 2000 4000; do
        train_one 500  ${TLEN} 0 4
    done
    echo "=== GPU4/Re=500 完成 ==="
) &

(
    for TLEN in 100 500 1000 2000 4000; do
        train_one 1000 ${TLEN} 0 5
    done
    echo "=== GPU5/Re=1000 完成 ==="
) &

(
    for TLEN in 100 500 1000 2000 4000; do
        train_one 2000 ${TLEN} 0 6
    done
    echo "=== GPU6/Re=2000 完成 ==="
) &

(
    for TLEN in 100 500 1000 2000 4000; do
        train_one 3000 ${TLEN} 0 7
    done
    echo "=== GPU7/Re=3000 完成 ==="
) &

echo "====== 1-epoch tlen 已启动：GPU4-7 各跑 1 个 Re，预计 ~4.3h ======"
echo "日志: ${ROOT}/logs/train_tlen_1ep/"
wait
echo "====== 全部完成 $(date) ======"
