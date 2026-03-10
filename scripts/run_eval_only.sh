#!/usr/bin/env bash
# 只跑 predict + plot，不训练，用于查看当前 checkpoint 效果
# 用法：bash scripts/run_eval_only.sh
set -e
set -x

STORAGE_PATH="${STORAGE_PATH:-/jumbo/yaoqingyang/yuxin/JAX-CFD}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE_PATH/model}"
export OUTPUT_DIR
TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000/long_eval_2048x2048_64x64.nc
[ ! -f "$STORAGE_PATH/$EVALDATA" ] && EVALDATA=content/kolmogorov_re_1000/eval_2048x2048_64x64.nc

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
if [ -n "${CONDA_PREFIX}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0
mkdir -p logs

COMMON_ARGS=(
  --no_train
  --resume_checkpoint
  --model_encode_steps=16
  --model_decode_steps=160
  --model_predict_steps=16
  --delta_time=0.007012483601762931
  --train_split="$STORAGE_PATH/$TRAINDATA"
  --eval_split="$STORAGE_PATH/$EVALDATA"
  --output_dir="$OUTPUT_DIR"
  --gin_file=models/configs/official_li_config.gin
  --gin_file=models/configs/kolmogorov_forcing.gin
  "--gin_param=fixed_scale.rescaled_one = 0.2"
  "--gin_param=my_forward_tower_factory.num_hidden_channels = 128"
  "--gin_param=my_forward_tower_factory.num_hidden_layers = 6"
  "--gin_param=MyFusedLearnedInterpolation.pattern = \"simple\""
  --dataset_num_workers=0
)

# --- 1. 用 eval 初值生成 predict.nc ---
echo "=== [1/4] Predict from eval ICs ==="
python -u models/train.py \
  --do_predict \
  --predict_split="$STORAGE_PATH/$EVALDATA" \
  --predict_result=predict.nc \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee logs/eval_only_predict_eval.txt
[ ${PIPESTATUS[0]} -ne 0 ] && { echo "predict eval failed"; exit 1; }

# --- 2. 画 eval 对比图 ---
echo "=== [2/4] Plot eval vorticity correlation ==="
  python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" \
  --baseline_dir="$STORAGE_PATH/content/kolmogorov_re_1000" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" \
  --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label=LI \
  --max_time_steps=200 \
  2>&1 | tee logs/eval_only_plot_eval.txt || true

# --- 3. 用 train 初值生成 predict_train.nc ---
echo "=== [3/4] Predict from train ICs ==="
if ! python -u models/train.py \
  --do_predict \
  --predict_split="$STORAGE_PATH/$TRAINDATA" \
  --predict_result=predict_train.nc \
  --simulation_time=3.5 \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee logs/eval_only_predict_train.txt; then
  rm -f "$OUTPUT_DIR/predict_train.nc"
  echo "predict_train failed, removed partial file"
fi

# --- 4. 画 train_ref 对比图 ---
if [ -f "$OUTPUT_DIR/predict_train.nc" ]; then
  echo "=== [4/4] Plot train_ref vorticity correlation ==="
  python -u scripts/plot_loss_and_eval_from_predict.py \
    --output_dir="$OUTPUT_DIR" \
    --baseline_nc="$STORAGE_PATH/$TRAINDATA" \
    --predict_nc="$OUTPUT_DIR/predict_train.nc" \
    --model_label=LI \
    --plot_suffix=_train_ref \
    --max_time_steps=500 \
    2>&1 | tee logs/eval_only_plot_train_ref.txt || true
fi

echo ""
echo "Done. Plots saved to $OUTPUT_DIR/"
echo "  eval_vorticity_correlation.png"
echo "  eval_vorticity_correlation_train_ref.png"
