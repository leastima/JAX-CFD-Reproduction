#!/usr/bin/env bash
# Phase plot 批量训练脚本
# 在 Re={500,1000,2000,3000,4000} × n_traj={2,8,16,32} × seed={0,1,2} 上各训练 1 epoch
# 共 5×4×3 = 60 个模型
#
# 用法: bash scripts/train_phase_plot.sh
#
# GPU 分配：每块 GPU 串行跑多个任务，按 Re 分组：
#   GPU 4: Re=500
#   GPU 5: Re=1000
#   GPU 6: Re=2000
#   GPU 7: Re=3000，然后 Re=4000

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"

mkdir -p "${ROOT}/logs/train_phase"

# -------------------------------------------------------
# 训练单个模型
# train_one RE N_TRAJ SEED GPU
# -------------------------------------------------------
train_one() {
    local RE=$1
    local N_TRAJ=$2
    local SEED=$3
    local GPU=$4

    local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
    local MODEL_DIR="${ROOT}/models_phase/re${RE}_ntraj${N_TRAJ}_seed${SEED}"
    local LOG="${ROOT}/logs/train_phase/re${RE}_ntraj${N_TRAJ}_seed${SEED}.log"

    mkdir -p "${MODEL_DIR}"
    echo "[Re=${RE} | ntraj=${N_TRAJ} | seed=${SEED} | GPU${GPU}] → ${MODEL_DIR}"

    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/models/train.py \
        --gin_file="${ROOT}/models/configs/official_li_config.gin" \
        --gin_file="${ROOT}/models/configs/kolmogorov_forcing.gin" \
        --gin_param="fixed_scale.rescaled_one = 0.2" \
        --gin_param="my_forward_tower_factory.num_hidden_channels = 128" \
        --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
        --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
        --train_split="${TRAIN_NC}" \
        --eval_split="${TRAIN_NC}" \
        --train_epochs=1 \
        --train_init_random_seed=${SEED} \
        --max_train_samples=${N_TRAJ} \
        --dataset_num_workers=0 \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1

    echo "[Re=${RE} | ntraj=${N_TRAJ} | seed=${SEED}] done ✓"
}

# -------------------------------------------------------
# GPU4: Re=500（4 n_traj × 3 seeds = 12 个模型）
# -------------------------------------------------------
(
    for N_TRAJ in 2 8 16 32; do
        for SEED in 0 1 2; do
            train_one 500 ${N_TRAJ} ${SEED} 4
        done
    done
    echo "=== GPU4 (Re=500) 全部完成 ==="
) &

# -------------------------------------------------------
# GPU5: Re=1000
# -------------------------------------------------------
(
    for N_TRAJ in 2 8 16 32; do
        for SEED in 0 1 2; do
            train_one 1000 ${N_TRAJ} ${SEED} 5
        done
    done
    echo "=== GPU5 (Re=1000) 全部完成 ==="
) &

# -------------------------------------------------------
# GPU6: Re=2000
# -------------------------------------------------------
(
    for N_TRAJ in 2 8 16 32; do
        for SEED in 0 1 2; do
            train_one 2000 ${N_TRAJ} ${SEED} 6
        done
    done
    echo "=== GPU6 (Re=2000) 全部完成 ==="
) &

# -------------------------------------------------------
# GPU7: Re=3000 → Re=4000
# -------------------------------------------------------
(
    for RE in 3000 4000; do
        for N_TRAJ in 2 8 16 32; do
            for SEED in 0 1 2; do
                train_one ${RE} ${N_TRAJ} ${SEED} 7
            done
        done
    done
    echo "=== GPU7 (Re=3000+4000) 全部完成 ==="
) &

echo "====== 已后台启动 4 组训练任务（共 60 模型）======"
echo "查看进度: ls ${ROOT}/logs/train_phase/ | wc -l  (最终应有60个log)"
echo "查看某个: tail -f ${ROOT}/logs/train_phase/re1000_ntraj32_seed0.log"

wait
echo "====== 全部训练完成 ======"
