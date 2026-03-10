#!/usr/bin/env bash
# 与 ML4HPC/JAX-CFD run_interpolation.sh 对齐的配置，使用 GPU 4–7。
# 见 https://github.com/ML4HPC/JAX-CFD/blob/main/scripts/run_interpolation.sh
set -e
set -x

# 确保用 cuda，并优先加载 conda 环境里的 CuDNN（>=9.12），避免系统 9.10 导致 DNN 初始化失败
if [ "${JAX_PLATFORMS}" = "gpu" ]; then
  export JAX_PLATFORMS=cuda
fi
export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
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
# 单卡避免 NCCL "Address already in use"；若环境支持多卡 NCCL 可改为 4,5,6,7
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

STORAGE_PATH=/jumbo/yaoqingyang/yuxin/JAX-CFD
TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
# Codebase 用 long_eval_2048x2048_64x64.nc；若无则用 eval
EVALDATA=content/kolmogorov_re_1000/eval_2048x2048_64x64.nc
[ -f "$STORAGE_PATH/content/kolmogorov_re_1000/long_eval_2048x2048_64x64.nc" ] && EVALDATA=content/kolmogorov_re_1000/long_eval_2048x2048_64x64.nc
OUTPUT_DIR=$STORAGE_PATH/model
mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

START_TIME=$(date +%s)
echo "=== Training (codebase config) started at $(date) ==="

# 与 ML4HPC run_interpolation.sh 一致：encode=16 decode=160 predict=16, batch=4, epochs=0.2, 128ch 6层, log_every=10
# 不默认 resume，避免 model/ 里旧 checkpoint（如 smoke/其他配置）结构与当前 6 层模型不一致导致恢复失败；续训时改为加上 --resume_checkpoint
# 多卡若遇 DataLoader fork 错误可加 --dataset_num_workers=0
python -u models/train.py \
  --model_encode_steps=16 \
  --model_decode_steps=160 \
  --model_predict_steps=16 \
  --train_device_batch_size=4 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --eval_batch_size=48 \
  --train_weight_decay=0.0 \
  --train_lr_init=0.0001 \
  --train_lr_warmup_epochs=0.0 \
  --mp_scale_value=1.0 \
  --train_epochs=0.2 \
  --train_log_every=10 \
  --decoding_warmup_steps=0 \
  --mp_skip_nonfinite \
  --do_eval \
  --do_predict \
  --predict_result=predict.nc \
  --output_dir="$OUTPUT_DIR" \
  --gin_file="models/configs/official_li_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.2" \
  --gin_param="my_forward_tower_factory.num_hidden_channels = 128" \
  --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
  --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
  --dataset_num_workers=0 \
  2>&1 | tee ./logs/train_log_li_codebase.txt
TRAIN_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Training finished at $(date) ==="
printf "=== Wall time: %ds (%dm %ds) ===\n" $ELAPSED $((ELAPSED/60)) $((ELAPSED%60))
[ $TRAIN_EXIT -ne 0 ] && exit $TRAIN_EXIT

# 最新 checkpoint 另存为 LI_ckpt.pkl
python -u - << 'PY'
import os
import pickle
from flax.training import checkpoints

output_dir = "/jumbo/yaoqingyang/yuxin/JAX-CFD/model"
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

# 可选：画 loss 与 eval 图（需有 predict.nc）
if [ -f "$OUTPUT_DIR/predict.nc" ]; then
  echo "=== Plotting loss and eval (ref: EVALDATA=$EVALDATA) ==="
  python -u scripts/plot_loss_and_eval_from_predict.py \
    --output_dir="$OUTPUT_DIR" \
    --baseline_dir="$STORAGE_PATH/content/kolmogorov_re_1000" \
    --baseline_nc="$STORAGE_PATH/$EVALDATA" \
    --predict_nc="$OUTPUT_DIR/predict.nc" \
    --model_label=LI \
    2>&1 | tee ./logs/plot_eval_codebase.txt || true
fi

# 用 train 数据做 predict（初值从 train nc 取），再和 train nc 对比，得到有意义的 train_ref 图
if [ -f "$OUTPUT_DIR/predict.nc" ]; then
  echo "=== Predict from train initial conditions (predict_train.nc) ==="
  python -u models/train.py \
    --no_train \
    --resume_checkpoint \
    --do_predict \
    --predict_split="$STORAGE_PATH/$TRAINDATA" \
    --predict_result=predict_train.nc \
    --model_encode_steps=16 \
    --model_decode_steps=160 \
    --model_predict_steps=16 \
    --delta_time=0.007012483601762931 \
    --train_split="$STORAGE_PATH/$TRAINDATA" \
    --eval_split="$STORAGE_PATH/$EVALDATA" \
    --output_dir="$OUTPUT_DIR" \
    --gin_file="models/configs/official_li_config.gin" \
    --gin_file="models/configs/kolmogorov_forcing.gin" \
    --gin_param="fixed_scale.rescaled_one = 0.2" \
    --gin_param="my_forward_tower_factory.num_hidden_channels = 128" \
    --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
    --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
    --dataset_num_workers=0 \
    2>&1 | tee ./logs/predict_train_log.txt || true
fi

if [ -f "$OUTPUT_DIR/predict_train.nc" ]; then
  echo "=== Plotting eval with train nc as reference (model init from train) ==="
  python -u scripts/plot_loss_and_eval_from_predict.py \
    --output_dir="$OUTPUT_DIR" \
    --baseline_nc="$STORAGE_PATH/$TRAINDATA" \
    --predict_nc="$OUTPUT_DIR/predict_train.nc" \
    --model_label=LI \
    --plot_suffix=_train_ref \
    --max_time_steps=500 \
    2>&1 | tee ./logs/plot_eval_train_ref.txt || true
fi

echo "Done. Checkpoints: $OUTPUT_DIR; predict: $OUTPUT_DIR/predict.nc; predict_train: $OUTPUT_DIR/predict_train.nc"
printf "=== Total wall time: %ds ===\n" $ELAPSED
