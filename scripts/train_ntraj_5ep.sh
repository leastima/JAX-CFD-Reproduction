#!/usr/bin/env bash
# 实验1: ntraj phase plot（5 epoch，seed 优先顺序）
# Re={500,1000,2000,3000,4000} × ntraj={2,8,16,32} × seed={0,1,2} = 60 模型
# 输出目录: models_ntraj/
#
# GPU 分配：
#   GPU 4: Re=500, 1000
#   GPU 5: Re=2000, 3000, 4000

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40   # 每 GPU 跑 2 进程，各限 40%

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
mkdir -p "${ROOT}/logs/train_ntraj"

train_one() {
    local RE=$1 N_TRAJ=$2 SEED=$3 GPU=$4
    local MODEL_DIR="${ROOT}/models_ntraj/re${RE}_ntraj${N_TRAJ}_seed${SEED}"
    local LOG="${ROOT}/logs/train_ntraj/re${RE}_ntraj${N_TRAJ}_seed${SEED}.log"

    if ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp"; then
        echo "[SKIP] re${RE}_ntraj${N_TRAJ}_seed${SEED}"; return
    fi
    mkdir -p "${MODEL_DIR}"
    echo "[Re=${RE} ntraj=${N_TRAJ} seed=${SEED} GPU${GPU}] start"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/models/train.py \
        --gin_file="${ROOT}/models/configs/official_li_config.gin" \
        --gin_file="${ROOT}/models/configs/kolmogorov_forcing.gin" \
        --gin_param="fixed_scale.rescaled_one = 0.2" \
        --gin_param="my_forward_tower_factory.num_hidden_channels = 128" \
        --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
        --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
        --train_split="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc" \
        --eval_split="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc" \
        --train_epochs=5 \
        --train_init_random_seed=${SEED} \
        --max_train_samples=${N_TRAJ} \
        --dataset_num_workers=0 \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1
    echo "[Re=${RE} ntraj=${N_TRAJ} seed=${SEED}] done ✓"
}

# 每块 GPU 同时跑 2 个 Re（并行），seed=0 优先
# GPU4: Re=500 || Re=1000 同时
(
    for SEED in 0 1 2; do
        for N_TRAJ in 2 8 16 32; do
            train_one 500 ${N_TRAJ} ${SEED} 4
        done
    done
    echo "=== GPU4/Re=500 完成 ==="
) &
(
    for SEED in 0 1 2; do
        for N_TRAJ in 2 8 16 32; do
            train_one 1000 ${N_TRAJ} ${SEED} 4
        done
    done
    echo "=== GPU4/Re=1000 完成 ==="
) &

# GPU5: Re=2000 || Re=3000 同时，完成后串行跑 Re=4000
(
    for SEED in 0 1 2; do
        for N_TRAJ in 2 8 16 32; do
            train_one 2000 ${N_TRAJ} ${SEED} 5
        done
    done
    echo "=== GPU5/Re=2000 完成 ==="
) &
(
    for SEED in 0 1 2; do
        for N_TRAJ in 2 8 16 32; do
            train_one 3000 ${N_TRAJ} ${SEED} 5
        done
    done
    # Re=4000 在 Re=3000 完成后串行（显存已释放）
    for SEED in 0 1 2; do
        for N_TRAJ in 2 8 16 32; do
            train_one 4000 ${N_TRAJ} ${SEED} 5
        done
    done
    echo "=== GPU5/Re=3000+4000 完成 ==="
) &

echo "====== 实验1(ntraj) 已启动：每 GPU 并行 2 个 Re，seed 优先 ======"
echo "预计 seed=0 Re=500~3000 完成时间：~9h（减半）"
echo "日志: ${ROOT}/logs/train_ntraj/"
wait
echo "====== 实验1 全部完成 ======"
