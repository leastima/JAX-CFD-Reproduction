#!/usr/bin/env bash
# 快速验证训练流程能否跑通：极少步数 + 最小模型，约 1～2 分钟完成。
# 通过即说明数据、模型、checkpoint、计时均正常；正式训练用 run_train_li_short.sh 或 run_train_fast.sh。
# 使用 cfd-gpu 时默认 GPU；可设置 JAX_PLATFORMS=cpu 强制 CPU。
set -e
set -x
set -o pipefail

# NVIDIA GPU 用 cuda；填 gpu 时自动改为 cuda
if [ "${JAX_PLATFORMS}" = "gpu" ]; then
  export JAX_PLATFORMS=cuda
fi
export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
# 用 GPU 时优先加载 conda 环境里的 CuDNN（需 >=9.12），避免系统 9.10 导致 DNN 初始化失败
if [ "$JAX_PLATFORMS" = "cuda" ] && [ -n "${CONDA_PREFIX}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ "$JAX_PLATFORMS" = "cuda" ] && [ -n "${CONDA_PREFIX}" ]; then
  if ! ls "${CONDA_PREFIX}"/lib/libcudnn* 1>/dev/null 2>&1; then
    echo "ERROR: JAX_PLATFORMS=cuda but no CuDNN in ${CONDA_PREFIX}/lib. JAX needs CuDNN >= 9.12."
    echo "Run: conda install -y -c nvidia \"cudnn>=9.12\"   then retry."
    exit 1
  fi
fi
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0
# Smoke 用单卡避免 NCCL 多卡通信问题
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

STORAGE_PATH=/jumbo/yaoqingyang/yuxin/JAX-CFD
TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000/eval_2048x2048_64x64.nc
OUTPUT_DIR=$STORAGE_PATH/model_smoke
mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

START_TIME=$(date +%s)
echo "=== Smoke run started at $(date) ==="

# #region agent log
DEBUG_LOG="/jumbo/yaoqingyang/yuxin/.cursor/debug-7618bd.log"
_llp="${LD_LIBRARY_PATH:-<unset>}"
_llp_trim="${_llp:0:400}"
_cp="${CONDA_PREFIX:-<unset>}"
_jp="${JAX_PLATFORMS:-<unset>}"
_conda_lib_has_cudnn=""
[ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ] && _conda_lib_has_cudnn=$(ls "$CONDA_PREFIX"/lib/libcudnn* 2>/dev/null | head -1 || echo "none")
printf '%s\n' "{\"sessionId\":\"7618bd\",\"hypothesisId\":\"H1\",\"location\":\"run_train_smoke.sh:pre-train\",\"message\":\"env before python\",\"data\":{\"JAX_PLATFORMS\":\"$_jp\",\"CONDA_PREFIX\":\"$_cp\",\"LD_LIBRARY_PATH_pre\":\"$_llp_trim\",\"conda_lib_has_cudnn\":\"$_conda_lib_has_cudnn\"},\"timestamp\":$(date +%s)000}" >> "$DEBUG_LOG" 2>/dev/null || true
# #endregion

# 极简：约 2～4 个 step（0.002 epoch）；encode=2 decode=2 predict=2；batch=2；16 通道 1 层
python -u models/train.py \
  --model_encode_steps=2 \
  --model_decode_steps=2 \
  --model_predict_steps=2 \
  --train_device_batch_size=2 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --eval_batch_size=32 \
  --train_weight_decay=0.0 \
  --train_lr_init=0.0001 \
  --train_lr_warmup_epochs=0.0 \
  --mp_scale_value=1.0 \
  --train_epochs=0.002 \
  --train_log_every=1 \
  --decoding_warmup_steps=0 \
  --mp_skip_nonfinite \
  --do_eval \
  --do_predict \
  --predict_result=predict.nc \
  --simulation_time=0.2 \
  --output_dir="$OUTPUT_DIR" \
  --gin_file="models/configs/official_li_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.2" \
  --gin_param="my_forward_tower_factory.num_hidden_channels = 16" \
  --gin_param="my_forward_tower_factory.num_hidden_layers = 1" \
  --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
  --dataset_num_workers=0 \
  2>&1 | tee ./logs/train_log_smoke.txt
TRAIN_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Smoke run finished at $(date) ==="
printf "=== Wall time: %ds (%dm %ds) ===\n" $ELAPSED $((ELAPSED/60)) $((ELAPSED%60))
[ $TRAIN_EXIT -ne 0 ] && exit $TRAIN_EXIT

# 可选：把 smoke 的 ckpt 拷到 model/LI_ckpt.pkl 便于后续 plot；这里只打日志
python -u - << PY
import os
import pickle
from flax.training import checkpoints

output_dir = "$OUTPUT_DIR"
pkl_path = os.path.join(output_dir, "LI_ckpt.pkl")
latest = checkpoints.latest_checkpoint(output_dir)
if latest:
    state = checkpoints.restore_checkpoint(output_dir, target=None)
    if state is not None:
        with open(pkl_path, "wb") as f:
            pickle.dump(state, f)
        print("Saved", pkl_path)
else:
    print("No checkpoint under", output_dir)
PY

# 若训练阶段 do_predict 未写出 predict.nc（例如报错），则仅做 predict 并写 predict.nc
if [ ! -f "$OUTPUT_DIR/predict.nc" ]; then
  echo "=== predict.nc missing, running prediction-only (no_train + do_predict + resume_checkpoint) ==="
  python -u models/train.py \
    --no_train \
    --do_predict \
    --resume_checkpoint \
    --predict_result=predict.nc \
    --simulation_time=0.2 \
    --model_encode_steps=2 \
    --model_decode_steps=2 \
    --model_predict_steps=2 \
    --delta_time=0.007012483601762931 \
    --train_split="$STORAGE_PATH/$TRAINDATA" \
    --eval_split="$STORAGE_PATH/$EVALDATA" \
    --output_dir="$OUTPUT_DIR" \
    --gin_file="models/configs/official_li_config.gin" \
    --gin_file="models/configs/kolmogorov_forcing.gin" \
    --gin_param="fixed_scale.rescaled_one = 0.2" \
    --gin_param="my_forward_tower_factory.num_hidden_channels = 16" \
    --gin_param="my_forward_tower_factory.num_hidden_layers = 1" \
    --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
    --dataset_num_workers=0 \
    2>&1 | tee ./logs/predict_only_smoke.txt || true
fi

# 绘制训练 loss 曲线 + 用 predict.nc 与多 baseline 做对比图（与 ml_model_inference_demo 一致）
# 若 content/kolmogorov_re_1000 下有多份 eval_*x*_64x64.nc 则画多 baseline + LI；否则仅用 --baseline_nc
BASELINE_DIR="$STORAGE_PATH/content/kolmogorov_re_1000"
echo "=== Plotting loss curve and eval vs baseline(s) ==="
python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" \
  --baseline_dir="$BASELINE_DIR" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" \
  --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label=LI \
  2>&1 | tee ./logs/plot_eval_smoke.txt || true

echo "Smoke test OK. Checkpoints and plots in $OUTPUT_DIR"
echo "  predict: $OUTPUT_DIR/predict.nc"
echo "  loss curve: $OUTPUT_DIR/train_loss_curve.png"
echo "  eval vs baseline: $OUTPUT_DIR/eval_*.png"
printf "=== Total wall time: %ds ===\n" $ELAPSED
