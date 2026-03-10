#!/usr/bin/env bash
# Phase plot 批量训练脚本（10 epochs，seed 优先顺序）
# Re={500,1000,2000,3000} × n_traj={2,8,16,32} × seed={0,1,2} = 48 模型
# 顺序：先跑所有 seed=0（ntraj=2→32），再 seed=1，再 seed=2
# 这样约 18h 后就有完整的 seed=0 可以出图
#
# 用法: bash scripts/train_phase_plot_10ep.sh
#
# GPU 分配：
#   GPU 4: Re=500  （12 模型）
#   GPU 5: Re=1000 （12 模型）
#   GPU 6: Re=2000 （12 模型）
#   GPU 7: Re=3000 （12 模型）

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"

mkdir -p "${ROOT}/logs/train_phase_10ep"

train_one() {
    local RE=$1
    local N_TRAJ=$2
    local SEED=$3
    local GPU=$4

    local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
    local MODEL_DIR="${ROOT}/models_phase_10ep/re${RE}_ntraj${N_TRAJ}_seed${SEED}"
    local LOG="${ROOT}/logs/train_phase_10ep/re${RE}_ntraj${N_TRAJ}_seed${SEED}.log"

    # 已有 checkpoint 则跳过（断点续训保护）
    if ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp"; then
        echo "[SKIP] re${RE}_ntraj${N_TRAJ}_seed${SEED} (checkpoint exists)"
        return
    fi

    mkdir -p "${MODEL_DIR}"
    echo "[Re=${RE} | ntraj=${N_TRAJ} | seed=${SEED} | GPU${GPU}] start → ${MODEL_DIR}"

    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/models/train.py \
        --gin_file="${ROOT}/models/configs/official_li_config.gin" \
        --gin_file="${ROOT}/models/configs/kolmogorov_forcing.gin" \
        --gin_param="fixed_scale.rescaled_one = 0.2" \
        --gin_param="my_forward_tower_factory.num_hidden_channels = 128" \
        --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
        --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
        --train_split="${TRAIN_NC}" \
        --eval_split="${TRAIN_NC}" \
        --train_epochs=10 \
        --train_init_random_seed=${SEED} \
        --max_train_samples=${N_TRAJ} \
        --dataset_num_workers=0 \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1

    echo "[Re=${RE} | ntraj=${N_TRAJ} | seed=${SEED}] done ✓"
}

# seed=0 最优先，确保约 18h 后能出图
# 每块 GPU 按 seed → ntraj 顺序串行跑
(
    for SEED in 0 1 2; do
        for N_TRAJ in 2 8 16 32; do
            train_one 500 ${N_TRAJ} ${SEED} 4
        done
    done
    echo "=== GPU4 (Re=500) 全部完成 ==="
) &

(
    for SEED in 0 1 2; do
        for N_TRAJ in 2 8 16 32; do
            train_one 1000 ${N_TRAJ} ${SEED} 5
        done
    done
    echo "=== GPU5 (Re=1000) 全部完成 ==="
) &

(
    for SEED in 0 1 2; do
        for N_TRAJ in 2 8 16 32; do
            train_one 2000 ${N_TRAJ} ${SEED} 6
        done
    done
    echo "=== GPU6 (Re=2000) 全部完成 ==="
) &

(
    for SEED in 0 1 2; do
        for N_TRAJ in 2 8 16 32; do
            train_one 3000 ${N_TRAJ} ${SEED} 7
        done
    done
    echo "=== GPU7 (Re=3000) 全部完成 ==="
) &

echo "====== 已启动 4 组训练（共 48 模型，10 epoch，seed 优先）======"
echo "预计 seed=0 全部完成（可出初版图）：~18h"
echo "全部完成：~55h"
echo "查看进度（seed=0 完成数）："
echo "  ls ${ROOT}/models_phase_10ep/ | grep seed0 | wc -l  （应为 16）"

wait
echo "====== 全部训练完成 ======"
