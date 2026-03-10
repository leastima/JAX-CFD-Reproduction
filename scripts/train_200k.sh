#!/usr/bin/env bash
# 大规模实验：Re={500,1000,2000,3000,4000} × ntraj={2,4,8,16,32} × seed={0,1,2}
# 50000 steps, lr=1e-3, early stop: loss_delta<3e-6 连续3次
# 输出目录: models_200k/
#
# ★ 论文对齐参数：
#   model_decode_steps=32  (paper: "unroll 32 time steps when calculating loss")
#   model_encode_steps=64  (paper default)
#   train_device_batch_size=32  (从128降低，因为decode=32时BPTT内存约4-5×增大)
#
# ★ 每 GPU 最多同时 2 个进程（MEM_FRACTION=0.45）
#
# GPU 分配（每个 Re 独占一块 GPU）：
#   GPU4: Re=500  → 跑完后跑 Re=4000
#   GPU5: Re=1000
#   GPU6: Re=2000
#   GPU7: Re=3000
#
# 每个 Re / seed 组合：5 个 ntraj 分 3 批运行（2+2+1）
#   batch1: ntraj=2, ntraj=4   (并行)
#   batch2: ntraj=8, ntraj=16  (并行)
#   batch3: ntraj=32            (单独)
#
# Seed 串行: 0 → 1 → 2

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45   # 2 进程/GPU × 45% ≈ 90%
export OMP_NUM_THREADS=4          # 限制每进程 OpenMP 线程数，防止CPU爆炸
export TF_NUM_INTEROP_THREADS=4   # 限制 TF 跨操作并行线程数
export TF_NUM_INTRAOP_THREADS=4   # 限制 TF 操作内并行线程数

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
TRAIN_STEPS=50000
LR=0.001
WARMUP_EPOCHS=1
EARLY_STOP_DELTA=1e-7    # 连续10次(每次100步)变化<1e-7才停，避免欠训练
ENCODE_STEPS=16    # 论文参数: encode 16 步 context（model_fixed_1ep 脚本确认）
DECODE_STEPS=32    # 论文对齐: "unroll 32 time steps when calculating loss"
BATCH_SIZE=16      # 适中：model_fixed_1ep 用 4，我们用 16 加速
DELTA_TIME=0.007012483601762931  # 数据帧间隔 → inner_steps=1（默认0.001会导致inner_steps=7，224层BPTT）

mkdir -p "${ROOT}/logs/train_200k"

# -------------------------------------------------------
# train_one RE NTRAJ SEED GPU
# -------------------------------------------------------
train_one() {
    local RE=$1 NTRAJ=$2 SEED=$3 GPU=$4
    local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
    local MODEL_DIR="${ROOT}/models_200k/re${RE}_ntraj${NTRAJ}_seed${SEED}"
    local LOG="${ROOT}/logs/train_200k/re${RE}_ntraj${NTRAJ}_seed${SEED}.log"

    if ls "${MODEL_DIR}"/checkpoint_* 2>/dev/null | grep -qv "tmp"; then
        echo "[SKIP] re${RE}_ntraj${NTRAJ}_seed${SEED} already done"
        return
    fi

    mkdir -p "${MODEL_DIR}"
    echo "[re${RE} ntraj${NTRAJ} seed${SEED} GPU${GPU}] start"

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
        --train_lr_warmup_epochs=${WARMUP_EPOCHS} \
        --early_stop_loss_delta=${EARLY_STOP_DELTA} \
        --train_init_random_seed=${SEED} \
        --max_train_samples=${NTRAJ} \
        --model_encode_steps=${ENCODE_STEPS} \
        --model_decode_steps=${DECODE_STEPS} \
        --delta_time=${DELTA_TIME} \
        --train_device_batch_size=${BATCH_SIZE} \
        --train_weight_decay=0.0 \
        --train_lr_warmup_epochs=0.0 \
        --mp_skip_nonfinite \
        --mp_scale_value=1.0 \
        --dataset_num_workers=0 \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1

    echo "[re${RE} ntraj${NTRAJ} seed${SEED}] done ✓"
}

# -------------------------------------------------------
# run_re_all_seeds RE GPU
# 5 ntraj 分 3 批（2+2+1），每批最多 2 个并行
# -------------------------------------------------------
run_re_all_seeds() {
    local RE=$1 GPU=$2
    for SEED in 0; do
        echo "  [re${RE} GPU${GPU}] seed=${SEED} starting..."

        # batch1: ntraj=2, 4
        train_one ${RE}  2 ${SEED} ${GPU} &
        train_one ${RE}  4 ${SEED} ${GPU} &
        wait
        echo "  [re${RE} GPU${GPU}] seed=${SEED} batch1 done (ntraj=2,4)"

        # batch2: ntraj=8, 16
        train_one ${RE}  8 ${SEED} ${GPU} &
        train_one ${RE} 16 ${SEED} ${GPU} &
        wait
        echo "  [re${RE} GPU${GPU}] seed=${SEED} batch2 done (ntraj=8,16)"

        # batch3: ntraj=32
        train_one ${RE} 32 ${SEED} ${GPU}
        echo "  [re${RE} GPU${GPU}] seed=${SEED} batch3 done (ntraj=32)"

        echo "  [re${RE} GPU${GPU}] seed=${SEED} ALL DONE ✓"
    done
    echo "=== Re=${RE} GPU${GPU} ALL SEEDS DONE ==="
}

# -------------------------------------------------------
# 主流程：4 GPU 并行
# -------------------------------------------------------
echo "===== train_200k.sh 开始 $(date) ====="
echo "  Re={500,1000,2000,3000,4000} × ntraj={2,4,8,16,32} × seed={0,1,2}"
  echo "  train_steps=${TRAIN_STEPS}, lr=${LR}, early_stop_delta=${EARLY_STOP_DELTA} (3 consecutive)"
echo "  ★ max 2 processes per GPU (MEM_FRACTION=0.45)"
echo ""

( run_re_all_seeds 500  4; run_re_all_seeds 4000 4; echo "=== GPU4 全部完成 $(date) ===" ) &
sleep 90  # 错峰启动：等第一批编译结束再启动下一个GPU，避免8进程同时抢CPU
( run_re_all_seeds 1000 5;                           echo "=== GPU5 全部完成 $(date) ===" ) &
sleep 90
( run_re_all_seeds 2000 6;                           echo "=== GPU6 全部完成 $(date) ===" ) &
sleep 90
( run_re_all_seeds 3000 7;                           echo "=== GPU7 全部完成 $(date) ===" ) &

wait
echo "===== 全部 75 个模型训练完成 $(date) ====="
