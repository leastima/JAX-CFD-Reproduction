#!/usr/bin/env bash
# 用已有 checkpoint 重新跑 predict（生成带正确时间坐标的 predict.nc）+ 画图。
# 修复时间对齐后，无需重新训练，只需重跑本脚本。
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD

STORAGE_PATH=/jumbo/yaoqingyang/yuxin/JAX-CFD
DATA_DIR=$STORAGE_PATH/content/kolmogorov_re_1000_gen
EVALDATA=content/kolmogorov_re_1000_gen/long_eval_2048x2048_64x64.nc
OUTPUT_DIR=$STORAGE_PATH/model_paper_gen_start_only

echo "========== 1) Predict only (load checkpoint, save predict.nc with correct time coord) =========="
python -u models/train.py \
  --model_encode_steps=64 --model_decode_steps=32 --model_predict_steps=64 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$EVALDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --eval_batch_size=256 \
  --train_epochs=0 \
  --no_train \
  --do_predict \
  --predict_result=predict.nc \
  --simulation_time=20.0 \
  --output_dir="$OUTPUT_DIR" \
  --gin_file="models/configs/official_li_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.2" \
  2>&1 | tee ./logs/rerun_predict_log.txt

echo "========== 2) Plot (LI aligned to reference time grid: ref t=0..63, LI from t=64) =========="
python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" \
  --baseline_dir="$DATA_DIR" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" \
  --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label=LI \
  --max_time_steps=200 \
  2>&1 | tee ./logs/rerun_plot.txt || true

python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" \
  --baseline_dir="$DATA_DIR" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" \
  --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label=LI \
  --max_time_steps=32 \
  --plot_suffix=_first32 \
  2>&1 | tee ./logs/rerun_plot_first32.txt || true

echo "========== Done. Plots: $OUTPUT_DIR/eval_vorticity_correlation.png, ..._first32.png =========="
