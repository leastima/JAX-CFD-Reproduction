#!/usr/bin/env bash
# 批量评估 phase plot 所有模型
# Re={500,1000,2000,3000,4000} × ntraj={2,8,16,32} × seed={0,1,2}
#
# GPU 分配:  GPU4=Re500, GPU5=Re1000, GPU6=Re2000, GPU7=Re3000+Re4000
# 用法: bash scripts/run_eval_phase.sh

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
OUTPUT_CSV="${ROOT}/results/phase_metrics.csv"

mkdir -p "${ROOT}/results" "${ROOT}/logs/eval_phase"

eval_one() {
    local RE=$1 NTRAJ=$2 SEED=$3 GPU=$4
    local MODEL_DIR="${ROOT}/models_phase/re${RE}_ntraj${NTRAJ}_seed${SEED}"
    local EVAL_NC="${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc"
    local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
    local LOG="${ROOT}/logs/eval_phase/re${RE}_ntraj${NTRAJ}_seed${SEED}.log"

    if [ ! -d "${MODEL_DIR}" ]; then
        echo "[SKIP] ${MODEL_DIR} not found"
        return
    fi

    echo "[Re=${RE} ntraj=${NTRAJ} seed=${SEED} GPU${GPU}]"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
        --model_dir  "${MODEL_DIR}" \
        --eval_nc    "${EVAL_NC}" \
        --train_nc   "${TRAIN_NC}" \
        --re         ${RE} \
        --ntraj      ${NTRAJ} \
        --seed       ${SEED} \
        --output_csv "${OUTPUT_CSV}" \
        --length     200 \
        --inner_steps 10 \
        > "${LOG}" 2>&1 \
    && echo "  ✓ done" \
    || echo "  ✗ FAILED (see ${LOG})"
}

# GPU4: Re=500
(
    for NTRAJ in 2 8 16 32; do
        for SEED in 0 1 2; do eval_one 500  ${NTRAJ} ${SEED} 4; done
    done
    echo "=== GPU4 done ==="
) &

# GPU5: Re=1000
(
    for NTRAJ in 2 8 16 32; do
        for SEED in 0 1 2; do eval_one 1000 ${NTRAJ} ${SEED} 5; done
    done
    echo "=== GPU5 done ==="
) &

# GPU6: Re=2000
(
    for NTRAJ in 2 8 16 32; do
        for SEED in 0 1 2; do eval_one 2000 ${NTRAJ} ${SEED} 6; done
    done
    echo "=== GPU6 done ==="
) &

# GPU7: Re=3000 + Re=4000
(
    for RE in 3000 4000; do
        for NTRAJ in 2 8 16 32; do
            for SEED in 0 1 2; do eval_one ${RE} ${NTRAJ} ${SEED} 7; done
        done
    done
    echo "=== GPU7 done ==="
) &

echo "====== 已启动 4 组评估任务（共 60 模型）======"
echo "结果将写入: ${OUTPUT_CSV}"
wait
echo "====== 全部评估完成 ======"
