#!/usr/bin/env bash
# 批量生成不同 Re 的训练和评估数据
# GPU 4: Re 500, 1000, 1500
# GPU 5: Re 2000, 2500, 3000
# GPU 7: Re 4000, 4500
#
# 用法: bash scripts/gen_multi_re.sh

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
# 禁止 JAX 预分配全部显存，避免 XLA 编译时 OOM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
SCRIPT="${ROOT}/scripts/generate_kolmogorov_data.py"

# 公共参数
DNS_SIZE=2048
SAVE_SIZE=64
WARMUP=40.0
TSF_TRAIN=1     # train: 每外步保存一帧，dt_frame=0.007012s（与原始 train 一致，4270帧/30s）
TSF_EVAL=10     # eval:  每10外步保存一帧，dt_frame=0.07012s（与原始 eval 一致，285帧/20s）

mkdir -p "${ROOT}/logs"

# -------------------------------------------------------
# 生成单个 Re 的 train + eval（串行，避免同 GPU 双进程 OOM）
# -------------------------------------------------------
gen_one_re() {
    local RE=$1
    local GPU=$2
    local DATA_DIR="${ROOT}/content/kolmogorov_re${RE}"
    mkdir -p "${DATA_DIR}"

    local TRAIN_NC="${DATA_DIR}/train_${DNS_SIZE}x${DNS_SIZE}_${SAVE_SIZE}x${SAVE_SIZE}.nc"
    local EVAL_NC="${DATA_DIR}/long_eval_${DNS_SIZE}x${DNS_SIZE}_${SAVE_SIZE}x${SAVE_SIZE}.nc"
    local LOG_TRAIN="${ROOT}/logs/gen_re${RE}_train.log"
    local LOG_EVAL="${ROOT}/logs/gen_re${RE}_eval.log"

    echo "[Re=${RE} | GPU${GPU}] train → ${TRAIN_NC}"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${SCRIPT} \
        --re            ${RE} \
        --output        "${TRAIN_NC}" \
        --num_samples   32 \
        --dns_size      ${DNS_SIZE} \
        --save_size     ${SAVE_SIZE} \
        --warmup_time   ${WARMUP} \
        --simulation_time 30.0 \
        --time_subsample_factor ${TSF_TRAIN} \
        --seed 0 \
        > "${LOG_TRAIN}" 2>&1
    echo "[Re=${RE} | GPU${GPU}] train done ✓"

    echo "[Re=${RE} | GPU${GPU}] eval  → ${EVAL_NC}"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${SCRIPT} \
        --re            ${RE} \
        --output        "${EVAL_NC}" \
        --num_samples   16 \
        --dns_size      ${DNS_SIZE} \
        --save_size     ${SAVE_SIZE} \
        --warmup_time   ${WARMUP} \
        --simulation_time 20.0 \
        --time_subsample_factor ${TSF_EVAL} \
        --seed 2 \
        > "${LOG_EVAL}" 2>&1
    echo "[Re=${RE} | GPU${GPU}] eval  done ✓"
}

# -------------------------------------------------------
# GPU 4: Re 500 → 1000 → 1500（串行）
# -------------------------------------------------------
(
    gen_one_re 500  4
    gen_one_re 1000 4
    gen_one_re 1500 4
    echo "=== GPU4 全部完成 ==="
) &
PID4=$!

# -------------------------------------------------------
# GPU 5: Re 2000 → 2500 → 3000（串行）
# -------------------------------------------------------
(
    gen_one_re 2000 5
    gen_one_re 2500 5
    gen_one_re 3000 5
    echo "=== GPU5 全部完成 ==="
) &
PID5=$!

# -------------------------------------------------------
# GPU 7: Re 4000 → 4500（串行）
# -------------------------------------------------------
(
    gen_one_re 4000 7
    gen_one_re 4500 7
    echo "=== GPU7 全部完成 ==="
) &
PID7=$!

echo "====== 已后台启动三组生成任务 ======"
echo "  GPU4 PID: ${PID4}  (Re 500, 1000, 1500)"
echo "  GPU5 PID: ${PID5}  (Re 2000, 2500, 3000)"
echo "  GPU7 PID: ${PID7}  (Re 4000, 4500)"
echo ""
echo "实时查看进度："
echo "  tail -f ${ROOT}/logs/gen_re*_train.log"
echo ""
echo "等待所有任务完成..."
wait ${PID4} ${PID5} ${PID7}
echo "====== 全部 Re 数据生成完毕 ======"
