#!/usr/bin/env bash
# 训练脚本（修复了 @gin.configurable bug 后首次正式训练）
# 目标：跑完 2 epochs，确认 LI 模型真正在工作
# 用法：
#   nohup bash scripts/run_train_fixed_test.sh > logs/run_fixed_test.txt 2>&1 &
set -e
set -x

# ========== 路径 & GPU ==========
STORAGE_PATH="/jumbo/yaoqingyang/yuxin/JAX-CFD"
OUTPUT_DIR="$STORAGE_PATH/model_fixed"          # 新目录，与旧 broken model 区分
LOG_DIR="$STORAGE_PATH/logs"

TRAINDATA="content/kolmogorov_re_1000/train_2048x2048_64x64.nc"
EVALDATA="content/kolmogorov_re_1000/long_eval_2048x2048_64x64.nc"
[ -f "$STORAGE_PATH/$EVALDATA" ] || EVALDATA="content/kolmogorov_re_1000/eval_2048x2048_64x64.nc"

export CUDA_VISIBLE_DEVICES=7
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0
[ -n "${CONDA_PREFIX}" ] && [ -d "${CONDA_PREFIX}/lib" ] && \
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$STORAGE_PATH"
export PYTHONPATH=$PWD
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

START_TIME=$(date +%s)

# ========== 1. 训练（2 epochs 测试） ==========
# 论文参数（Kochkov et al. 2021）：
#   · 64×64 grid from 2048×2048 DNS
#   · Adam lr=1e-4, no weight decay
#   · encode_steps=16, decode_steps=160（unroll 160 steps）
#   · CNN: 128 channels, 6 layers, kernel=3
#   · batch_size=4 per device（单卡）
echo "=== [1/4] Training (2 epochs) ==="
python -u models/train.py \
  --model_encode_steps=16 \
  --model_decode_steps=32 \
  --model_predict_steps=16 \
  --train_device_batch_size=4 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --eval_batch_size=32 \
  --train_weight_decay=0.0 \
  --train_lr_init=0.001 \
  --train_lr_warmup_epochs=0.0 \
  --mp_scale_value=1.0 \
  --train_epochs=2 \
  --train_log_every=100 \
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
  2>&1 | tee "$LOG_DIR/train_fixed_test.txt"
[ ${PIPESTATUS[0]} -ne 0 ] && { echo "Training failed"; exit 1; }

# ========== 2. 画 eval 图（predict.nc vs eval baselines） ==========
echo "=== [2/4] Plot eval correlation ==="
python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" \
  --baseline_dir="$STORAGE_PATH/content/kolmogorov_re_1000" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" \
  --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label="LI_fixed_1ep" \
  --max_time_steps=200 \
  2>&1 | tee "$LOG_DIR/plot_fixed_eval.txt" || true

# ========== 3. 用 train 初值做推理（predict_train.nc） ==========
echo "=== [3/4] Predict from train ICs ==="
python -u models/train.py \
  --no_train \
  --resume_checkpoint \
  --do_predict \
  --predict_split="$STORAGE_PATH/$TRAINDATA" \
  --predict_result=predict_train.nc \
  --simulation_time=3.5 \
  --model_encode_steps=16 \
  --model_decode_steps=32 \
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
  2>&1 | tee "$LOG_DIR/predict_train_fixed.txt" || \
  { rm -f "$OUTPUT_DIR/predict_train.nc"; echo "predict_train failed, skipped"; }

# ========== 4. 画 train_ref 图 ==========
if [ -f "$OUTPUT_DIR/predict_train.nc" ]; then
  echo "=== [4/4] Plot train_ref correlation ==="
  python -u scripts/plot_loss_and_eval_from_predict.py \
    --output_dir="$OUTPUT_DIR" \
    --baseline_nc="$STORAGE_PATH/$TRAINDATA" \
    --predict_nc="$OUTPUT_DIR/predict_train.nc" \
  --model_label="LI_fixed_1ep" \
  --plot_suffix=_train_ref \
    --max_time_steps=500 \
    2>&1 | tee "$LOG_DIR/plot_fixed_train_ref.txt" || true
fi

ELAPSED=$(($(date +%s) - START_TIME))
echo ""
echo "=== Done. Output: $OUTPUT_DIR ==="
printf "Total wall time: %dh %dm\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60))
