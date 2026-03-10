#!/usr/bin/env bash
# 少量轮次试跑：用 content/kolmogorov_re_1000 下的 train/eval 数据，模型写到 model/。
# 数据已检查：train_2048x2048_64x64.nc (32 samples, 4880 steps), eval_2048x2048_64x64.nc (32 samples, 488 steps)。
# Flax 会把 checkpoint 存到 output_dir 下（checkpoint_0, checkpoint_100, ...）；训练结束后把最新一次另存为 model/LI_ckpt.pkl。
set -e
set -x

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

STORAGE_PATH=/jumbo/yaoqingyang/yuxin/JAX-CFD
TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000/eval_2048x2048_64x64.nc
OUTPUT_DIR=$STORAGE_PATH/model
mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

# 计时：记录开始时间
START_TIME=$(date +%s)
echo "=== Training started at $(date) ==="

# 少量轮次看耗时（约 0.05 epoch）；正式训练可改为 --train_epochs=0.2 或更大
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
  --train_epochs=0.05 \
  --train_log_every=5 \
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
  2>&1 | tee ./logs/train_log_li_short.txt
TRAIN_EXIT=$?

# 计时：训练阶段耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Training finished at $(date) ==="
printf "=== Training wall time: %ds (%dm %ds) ===\n" $ELAPSED $((ELAPSED/60)) $((ELAPSED%60))
[ $TRAIN_EXIT -ne 0 ] && exit $TRAIN_EXIT

# 把最新 checkpoint 另存为 LI_ckpt.pkl（单文件，便于后续加载）
python -u - << 'PY'
import os
import pickle
from flax.training import checkpoints

output_dir = "/jumbo/yaoqingyang/yuxin/JAX-CFD/model"
pkl_path = os.path.join(output_dir, "LI_ckpt.pkl")
latest = checkpoints.latest_checkpoint(output_dir)
if latest:
    # 只恢复为 dict（不依赖 SaveState 结构），便于 pickle
    state = checkpoints.restore_checkpoint(output_dir, target=None)
    if state is not None:
        with open(pkl_path, "wb") as f:
            pickle.dump(state, f)
        print("Saved latest checkpoint to", pkl_path)
    else:
        print("No checkpoint state to save to", pkl_path)
else:
    print("No checkpoint found under", output_dir)
PY

# 若训练阶段未写出 predict.nc，则仅做 predict（no_train + do_predict + resume_checkpoint）
if [ ! -f "$OUTPUT_DIR/predict.nc" ]; then
  echo "=== predict.nc missing, running prediction-only ==="
  python -u models/train.py \
    --no_train \
    --do_predict \
    --resume_checkpoint \
    --predict_result=predict.nc \
    --simulation_time=1.0 \
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
    2>&1 | tee ./logs/predict_only_log.txt || true
fi

# 绘制训练 loss 曲线 + 多 baseline 对比图（与 ml_model_inference_demo 一致）
BASELINE_DIR="$STORAGE_PATH/content/kolmogorov_re_1000"
echo "=== Plotting loss curve and eval vs baseline(s) ==="
python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" \
  --baseline_dir="$BASELINE_DIR" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" \
  --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label=LI \
  2>&1 | tee ./logs/plot_eval_log.txt || true

echo "Done. Checkpoints in $OUTPUT_DIR; predict: $OUTPUT_DIR/predict.nc"
echo "Plots: $OUTPUT_DIR/train_loss_curve.png, $OUTPUT_DIR/eval_*.png"
printf "=== Total wall time: %ds (%dm %ds) ===\n" $ELAPSED $((ELAPSED/60)) $((ELAPSED%60))
