#!/usr/bin/env bash
# 实验2: time_length phase plot（5 epoch，seed 优先顺序）
# Re={500,1000,2000,3000,4000} × tlen={100,500,1000,2000,4000} × seed={0,1,2} = 75 模型
# ntraj 固定为 32，限制每条轨迹的时间帧数
# 注意: tlen=10 因 encode_steps=64 不可用，最小值 tlen=100（36 有效样本/轨迹）
# 输出目录: models_tlen/
#
# GPU 分配：
#   GPU 6: Re=500, 1000
#   GPU 7: Re=2000, 3000, 4000

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40   # 每 GPU 跑 2 进程，各限 40%

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
mkdir -p "${ROOT}/logs/train_tlen"

train_one() {
    local RE=$1 TLEN=$2 SEED=$3 GPU=$4
    local MODEL_DIR="${ROOT}/models_tlen/re${RE}_tlen${TLEN}_seed${SEED}"
    local LOG="${ROOT}/logs/train_tlen/re${RE}_tlen${TLEN}_seed${SEED}.log"

    if ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp"; then
        echo "[SKIP] re${RE}_tlen${TLEN}_seed${SEED}"; return
    fi
    mkdir -p "${MODEL_DIR}"
    echo "[Re=${RE} tlen=${TLEN} seed=${SEED} GPU${GPU}] start"
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
        --max_train_samples=32 \
        --max_time_steps=${TLEN} \
        --dataset_num_workers=0 \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1
    echo "[Re=${RE} tlen=${TLEN} seed=${SEED}] done ✓"
}

# 每块 GPU 同时跑 2 个 Re（并行），seed=0 优先
# GPU6: Re=500 || Re=1000 同时
(
    for SEED in 0 1 2; do
        for TLEN in 100 500 1000 2000 4000; do
            train_one 500 ${TLEN} ${SEED} 6
        done
    done
    echo "=== GPU6/Re=500 完成 ==="
) &
(
    for SEED in 0 1 2; do
        for TLEN in 100 500 1000 2000 4000; do
            train_one 1000 ${TLEN} ${SEED} 6
        done
    done
    echo "=== GPU6/Re=1000 完成 ==="
) &

# GPU7: Re=2000 || Re=3000 同时，完成后串行跑 Re=4000
(
    for SEED in 0 1 2; do
        for TLEN in 100 500 1000 2000 4000; do
            train_one 2000 ${TLEN} ${SEED} 7
        done
    done
    echo "=== GPU7/Re=2000 完成 ==="
) &
(
    for SEED in 0 1 2; do
        for TLEN in 100 500 1000 2000 4000; do
            train_one 3000 ${TLEN} ${SEED} 7
        done
    done
    # Re=4000 在 Re=3000 完成后串行（显存已释放）
    for SEED in 0 1 2; do
        for TLEN in 100 500 1000 2000 4000; do
            train_one 4000 ${TLEN} ${SEED} 7
        done
    done
    echo "=== GPU7/Re=3000+4000 完成 ==="
) &

echo "====== 实验2(tlen) 已启动：每 GPU 并行 2 个 Re，seed 优先 ======"
echo "预计 seed=0 Re=500~3000 完成时间：~9h（减半）"
echo "日志: ${ROOT}/logs/train_tlen/"
wait
echo "====== 实验2 全部完成 ======"
