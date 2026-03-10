#!/usr/bin/env bash
# 一键流程：训练 → eval predict → 画 eval 图 → 用 train 做 predict_train → 画 train_ref 图
# 用法：./scripts/run_full_train_and_eval.sh
# 若已有 checkpoint 只做 predict+画图：SKIP_TRAIN=1 ./scripts/run_full_train_and_eval.sh
set -e
set -x

# ========== 可改配置（路径、GPU、是否跳过训练） ==========
STORAGE_PATH="${STORAGE_PATH:-/jumbo/yaoqingyang/yuxin/JAX-CFD}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_PATH/model}"
export OUTPUT_DIR
TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000/eval_2048x2048_64x64.nc
[ -f "$STORAGE_PATH/content/kolmogorov_re_1000/long_eval_2048x2048_64x64.nc" ] && EVALDATA=content/kolmogorov_re_1000/long_eval_2048x2048_64x64.nc
SKIP_TRAIN="${SKIP_TRAIN:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
if [ "${JAX_PLATFORMS}" = "gpu" ]; then export JAX_PLATFORMS=cuda; fi
if [ "$JAX_PLATFORMS" = "cuda" ] && [ -n "${CONDA_PREFIX}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0
mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

START_TIME=$(date +%s)

# ========== 1. 训练 + eval predict（生成 predict.nc） ==========
if [ "$SKIP_TRAIN" = "0" ]; then
  echo "=== [1/5] Training + eval predict ==="
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
    --train_epochs=5 \
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
    2>&1 | tee ./logs/train_log_full.txt
  [ ${PIPESTATUS[0]} -ne 0 ] && exit 1

  echo "=== Saving LI_ckpt.pkl ==="
  python -u - << PY
import os, pickle
from flax.training import checkpoints
output_dir = os.environ.get("OUTPUT_DIR", "$OUTPUT_DIR")
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
else
  echo "=== [1/5] SKIP_TRAIN=1, skip training ==="
fi

# ========== 2. 画 loss + eval 图（predict.nc vs eval baselines） ==========
if [ -f "$OUTPUT_DIR/predict.nc" ]; then
  echo "=== [2/5] Plot loss + eval (ref: eval) ==="
  python -u scripts/plot_loss_and_eval_from_predict.py \
    --output_dir="$OUTPUT_DIR" \
    --baseline_dir="$STORAGE_PATH/content/kolmogorov_re_1000" \
    --baseline_nc="$STORAGE_PATH/$EVALDATA" \
    --predict_nc="$OUTPUT_DIR/predict.nc" \
    --model_label=LI \
    --max_time_steps=200 \
    2>&1 | tee ./logs/plot_eval_full.txt || true
else
  echo "=== [2/5] No predict.nc, skip eval plot ==="
fi

# ========== 3. 用 train 初值做 predict（生成 predict_train.nc） ==========
if [ -f "$OUTPUT_DIR/predict.nc" ]; then
  echo "=== [3/5] Predict from train ICs (predict_train.nc) ==="
  # 限制 simulation_time 使 predict_train 体积可写入（避免 netcdf4 HDF / scipy int32 溢出）
  if ! python -u models/train.py \
    --no_train \
    --resume_checkpoint \
    --do_predict \
    --predict_split="$STORAGE_PATH/$TRAINDATA" \
    --predict_result=predict_train.nc \
    --simulation_time=3.5 \
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
    2>&1 | tee ./logs/predict_train_log.txt; then
    rm -f "$OUTPUT_DIR/predict_train.nc"
    echo "=== [3/5] predict_train failed, removed partial predict_train.nc ==="
  fi
else
  echo "=== [3/5] No predict.nc, skip predict_train ==="
fi

# ========== 4. 画 train_ref 图（predict_train.nc vs train nc） ==========
if [ -f "$OUTPUT_DIR/predict_train.nc" ]; then
  echo "=== [4/5] Plot eval with train as reference ==="
  python -u scripts/plot_loss_and_eval_from_predict.py \
    --output_dir="$OUTPUT_DIR" \
    --baseline_nc="$STORAGE_PATH/$TRAINDATA" \
    --predict_nc="$OUTPUT_DIR/predict_train.nc" \
    --model_label=LI \
    --plot_suffix=_train_ref \
    --max_time_steps=500 \
    2>&1 | tee ./logs/plot_eval_train_ref.txt || true
else
  echo "=== [4/5] No predict_train.nc, skip train_ref plot ==="
fi

ELAPSED=$(($(date +%s) - START_TIME))
echo ""
echo "Done. Output: $OUTPUT_DIR"
echo "  predict.nc, predict_train.nc, train_loss_curve.png, eval_vorticity_correlation.png, eval_vorticity_correlation_train_ref.png"
printf "=== Total wall time: %ds ===\n" $ELAPSED
