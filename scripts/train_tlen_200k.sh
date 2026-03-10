#!/usr/bin/env bash
# tlen phase plot: Re={500,1000,2000,3000,4000} × tlen={100,500,2000,4000} × seed=0
# ntraj 固定 32，50000 steps, lr=1e-3，参数与 train_200k.sh 完全对齐
# 输出目录: models_tlen_200k/
#
# GPU 分配（每个 Re 独占一块）：
#   GPU4: Re=500  → 跑完后跑 Re=4000
#   GPU5: Re=1000
#   GPU6: Re=2000
#   GPU7: Re=3000
#
# 每个 Re：4 个 tlen 分 2 批并行（2+2）
#   batch1: tlen=100, 500
#   batch2: tlen=2000, 4000

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45   # 2 进程/GPU × 45% ≈ 90%
export OMP_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=4
export TF_NUM_INTRAOP_THREADS=4

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
TRAIN_STEPS=50000
LR=0.001
EARLY_STOP_DELTA=1e-7
ENCODE_STEPS=16
DECODE_STEPS=32
BATCH_SIZE=16
DELTA_TIME=0.007012483601762931
NTRAJ=32
SEED=0

mkdir -p "${ROOT}/logs/train_tlen_200k"

train_one() {
    local RE=$1 TLEN=$2 GPU=$3
    local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
    local MODEL_DIR="${ROOT}/models_tlen_200k/re${RE}_tlen${TLEN}_seed${SEED}"
    local LOG="${ROOT}/logs/train_tlen_200k/re${RE}_tlen${TLEN}_seed${SEED}.log"

    if ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp"; then
        echo "[SKIP] re${RE}_tlen${TLEN}_seed${SEED} already done"
        return
    fi

    mkdir -p "${MODEL_DIR}"
    echo "[re${RE} tlen=${TLEN} seed=${SEED} GPU${GPU}] start $(date +%H:%M)"

    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/models/train.py \
        --gin_file="${ROOT}/models/configs/official_li_config.gin" \
        --gin_file="${ROOT}/models/configs/kolmogorov_forcing.gin" \
        --gin_param="fixed_scale.rescaled_one = 0.2" \
        --gin_param="my_forward_tower_factory.num_hidden_channels = 128" \
        --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
        --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
        --gin_param="physics_specifications.NavierStokesPhysicsSpecs.viscosity = $(python3 -c "print(1/${RE})")" \
        --train_split="${TRAIN_NC}" \
        --eval_split="${TRAIN_NC}" \
        --train_steps=${TRAIN_STEPS} \
        --train_lr_init=${LR} \
        --train_lr_warmup_epochs=0.0 \
        --early_stop_loss_delta=${EARLY_STOP_DELTA} \
        --early_stop_patience=10 \
        --train_init_random_seed=${SEED} \
        --max_train_samples=${NTRAJ} \
        --max_time_steps=${TLEN} \
        --model_encode_steps=${ENCODE_STEPS} \
        --model_decode_steps=${DECODE_STEPS} \
        --delta_time=${DELTA_TIME} \
        --train_device_batch_size=${BATCH_SIZE} \
        --train_weight_decay=0.0 \
        --mp_skip_nonfinite \
        --mp_scale_value=1.0 \
        --dataset_num_workers=0 \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1

    echo "[re${RE} tlen=${TLEN} seed=${SEED}] done ✓ $(date +%H:%M)"
}

run_re() {
    local RE=$1 GPU=$2
    echo "  [re${RE} GPU${GPU}] 开始 tlen 实验..."

    train_one ${RE} 100  ${GPU} &
    train_one ${RE} 500  ${GPU} &
    wait
    echo "  [re${RE} GPU${GPU}] batch1 done (tlen=100,500)"

    train_one ${RE} 2000 ${GPU} &
    train_one ${RE} 4000 ${GPU} &
    wait
    echo "  [re${RE} GPU${GPU}] batch2 done (tlen=2000,4000)"

    echo "=== Re=${RE} GPU${GPU} ALL DONE ==="
}

echo "===== train_tlen_200k.sh 开始 $(date) ====="
echo "  Re={500,1000,2000,3000,4000} × tlen={100,500,2000,4000} × seed=0, ntraj=32"
echo "  train_steps=${TRAIN_STEPS}, lr=${LR}, delta=${EARLY_STOP_DELTA} x10"
echo ""

( run_re 500  4; echo "=== GPU4 全部完成 $(date) ===" ) &
sleep 90
( run_re 1000 5; echo "=== GPU5 全部完成 $(date) ===" ) &
sleep 90
( run_re 2000 6; echo "=== GPU6 全部完成 $(date) ===" ) &
sleep 90
( run_re 3000 7; echo "=== GPU7 全部完成 $(date) ===" ) &

wait
echo "===== 全部 16 个模型训练完成 $(date) ====="
