#!/usr/bin/env bash
# 在 CPU 上能较快跑完的“迷你 paper”配置：1 epoch、小 batch、短序列，用于验证流程 + 明天看 eval。
# 完整 90 epoch 请在有 GPU/TPU 时用 run_train_paper.sh。
set -e
set -x

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

STORAGE_PATH=/jumbo/yaoqingyang/yuxin/JAX-CFD
TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000/eval_2048x2048_64x64.nc
OUTPUT_DIR=$STORAGE_PATH/model_paper_cpu_mini
mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

START_TIME=$(date +%s)
echo "=== Paper CPU-mini (1 epoch, small batch) started at $(date) ==="

python -u models/train.py \
  --model_encode_steps=16 \
  --model_decode_steps=16 \
  --model_predict_steps=16 \
  --train_device_batch_size=4 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --eval_batch_size=32 \
  --train_weight_decay=1e-4 \
  --train_lr_init=0.001 \
  --train_lr_warmup_epochs=0.0 \
  --mp_scale_value=1.0 \
  --train_epochs=1.0 \
  --train_log_every=50 \
  --decoding_warmup_steps=0 \
  --mp_skip_nonfinite \
  --do_eval \
  --do_predict \
  --predict_result=predict.nc \
  --simulation_time=5.0 \
  --output_dir="$OUTPUT_DIR" \
  --gin_file="models/configs/official_li_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.2" \
  --gin_param="my_forward_tower_factory.num_hidden_channels = 64" \
  --gin_param="my_forward_tower_factory.num_hidden_layers = 4" \
  2>&1 | tee ./logs/train_log_paper_cpu_mini.txt
TRAIN_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Paper CPU-mini finished at $(date) ==="
printf "=== Wall time: %ds (%dm %ds) ===\n" $ELAPSED $((ELAPSED/60)) $((ELAPSED%60))
[ $TRAIN_EXIT -ne 0 ] && exit $TRAIN_EXIT

python -u - << PY
import os, pickle
from flax.training import checkpoints
out = "$OUTPUT_DIR"
pkl = os.path.join(out, "LI_ckpt.pkl")
state = checkpoints.restore_checkpoint(out, target=None)
if state is not None:
    with open(pkl, "wb") as f: pickle.dump(state, f)
    print("Saved", pkl)
PY

if [ ! -f "$OUTPUT_DIR/predict.nc" ]; then
  echo "=== Running prediction-only ==="
  python -u models/train.py --no_train --do_predict --resume_checkpoint \
    --predict_result=predict.nc --simulation_time=5.0 \
    --model_encode_steps=16 --model_decode_steps=16 --model_predict_steps=16 \
    --delta_time=0.007012483601762931 \
    --train_split="$STORAGE_PATH/$TRAINDATA" --eval_split="$STORAGE_PATH/$EVALDATA" \
    --output_dir="$OUTPUT_DIR" \
    --gin_file="models/configs/official_li_config.gin" --gin_file="models/configs/kolmogorov_forcing.gin" \
    --gin_param="fixed_scale.rescaled_one = 0.2" \
    --gin_param="my_forward_tower_factory.num_hidden_channels = 64" \
    --gin_param="my_forward_tower_factory.num_hidden_layers = 4" \
    2>&1 | tee ./logs/predict_only_cpu_mini.txt || true
fi

BASELINE_DIR="$STORAGE_PATH/content/kolmogorov_re_1000"
python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" --baseline_dir="$BASELINE_DIR" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label=LI --max_time_steps=200 \
  2>&1 | tee ./logs/plot_eval_cpu_mini.txt || true

echo "Done. Output: $OUTPUT_DIR. Total wall time: ${ELAPSED}s"
