#!/usr/bin/env bash
# 最快试跑：最少步数 + 最小 batch + 小模型 + 不做 eval，仅验证流程与计时。
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

START_TIME=$(date +%s)
echo "=== Training started at $(date) ==="

# 最少步数：约 0.01 epoch；最小 batch=2；短序列 encode=4 decode=4 predict=4；小模型 32 通道 2 层；不跑 eval
python -u models/train.py \
  --model_encode_steps=4 \
  --model_decode_steps=4 \
  --model_predict_steps=4 \
  --train_device_batch_size=2 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --eval_batch_size=8 \
  --train_weight_decay=0.0 \
  --train_lr_init=0.0001 \
  --train_lr_warmup_epochs=0.0 \
  --mp_scale_value=1.0 \
  --train_epochs=0.01 \
  --train_log_every=1 \
  --decoding_warmup_steps=0 \
  --mp_skip_nonfinite \
  --output_dir="$OUTPUT_DIR" \
  --gin_file="models/configs/official_li_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.2" \
  --gin_param="my_forward_tower_factory.num_hidden_channels = 32" \
  --gin_param="my_forward_tower_factory.num_hidden_layers = 2" \
  --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
  2>&1 | tee ./logs/train_log_fast.txt
TRAIN_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Training finished at $(date) ==="
printf "=== Training wall time: %ds (%dm %ds) ===\n" $ELAPSED $((ELAPSED/60)) $((ELAPSED%60))
[ $TRAIN_EXIT -ne 0 ] && exit $TRAIN_EXIT

python -u - << PY
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
        print("Saved latest checkpoint to", pkl_path)
else:
    print("No checkpoint found under", output_dir)
PY

echo "Done. Checkpoints in $OUTPUT_DIR; single-file: $OUTPUT_DIR/LI_ckpt.pkl"
printf "=== Total wall time: %ds (%dm %ds) ===\n" $ELAPSED $((ELAPSED/60)) $((ELAPSED%60))
