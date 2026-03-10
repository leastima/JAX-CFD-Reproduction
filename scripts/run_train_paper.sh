#!/usr/bin/env bash
# 按论文 "Machine learning accelerated computational fluid dynamics" (arxiv 2102.01010) 的配置训练。
# 在 smoke 脚本基础上改为作者原始规模，并尽量与论文/附录一致。
# 使用 GPU 6、7 时：CUDA_VISIBLE_DEVICES=6,7 bash scripts/run_train_paper.sh
# 或直接运行 run_train_paper_gpu_6_7.sh（仅用 6、7 卡，输出到 model_paper_gpu67）。
#
# === 论文 vs 代码 对照（有冲突处已注明）===
# 论文 Appendix B (Kolmogorov Re=1000): 2048→64, burn-in 40; 32 trajectories, 4800 steps each.
# 论文 Appendix G (Adam): lr=1e-3, b1=0.9, b2=0.99.
# 论文 III / Appendix: "We unroll the model for 32 time steps when calculating the loss" → decode_steps=32.
# 论文 Appendix D: Basic ConvNet, N_out=120 for LI; gin 里为 64 channels × 6 layers（与附录“略大略好”一致）。
#
# 冲突/说明:
# - 学习率: 论文 1e-3，代码默认 train_lr_init=0.1；此处显式 --train_lr_init=0.001 与论文一致。
# - 单步 unroll: 论文 32 steps，代码默认 model_decode_steps=1；此处 --model_decode_steps=32。
# - 数据: 若你的 train/eval 为 32 samples、4880/488 steps，与论文“32 trajectories、4800 steps”接近；eval 步数较少时可用 --simulation_time 控制 predict 长度以接近 notebook 的 length=200。
set -e
set -x

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

STORAGE_PATH=/jumbo/yaoqingyang/yuxin/JAX-CFD
TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000/eval_2048x2048_64x64.nc
OUTPUT_DIR=$STORAGE_PATH/model_paper
mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

START_TIME=$(date +%s)
echo "=== Paper-config training started at $(date) ==="

# 论文配置: 64 步 encode/decode/predict（与 default 一致），unroll 32 步算 loss；lr=1e-3；90 epochs
python -u models/train.py \
  --model_encode_steps=64 \
  --model_decode_steps=32 \
  --model_predict_steps=64 \
  --train_device_batch_size=128 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --eval_batch_size=256 \
  --train_weight_decay=1e-4 \
  --train_lr_init=0.001 \
  --train_lr_warmup_epochs=5 \
  --mp_scale_value=1.0 \
  --train_epochs=90 \
  --train_log_every=100 \
  --decoding_warmup_steps=0 \
  --mp_skip_nonfinite \
  --do_eval \
  --do_predict \
  --predict_result=predict.nc \
  --simulation_time=20.0 \
  --output_dir="$OUTPUT_DIR" \
  --gin_file="models/configs/official_li_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.2" \
  2>&1 | tee ./logs/train_log_paper.txt
TRAIN_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Paper-config training finished at $(date) ==="
printf "=== Training wall time: %ds (%dm %ds) ===\n" $ELAPSED $((ELAPSED/60)) $((ELAPSED%60))
[ $TRAIN_EXIT -ne 0 ] && exit $TRAIN_EXIT

# 最新 checkpoint 另存为 LI_ckpt.pkl
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

if [ ! -f "$OUTPUT_DIR/predict.nc" ]; then
  echo "=== predict.nc missing, running prediction-only ==="
  python -u models/train.py \
    --no_train \
    --do_predict \
    --resume_checkpoint \
    --predict_result=predict.nc \
    --simulation_time=20.0 \
    --model_encode_steps=64 \
    --model_decode_steps=32 \
    --model_predict_steps=64 \
    --delta_time=0.007012483601762931 \
    --train_split="$STORAGE_PATH/$TRAINDATA" \
    --eval_split="$STORAGE_PATH/$EVALDATA" \
    --output_dir="$OUTPUT_DIR" \
    --gin_file="models/configs/official_li_config.gin" \
    --gin_file="models/configs/kolmogorov_forcing.gin" \
    --gin_param="fixed_scale.rescaled_one = 0.2" \
    2>&1 | tee ./logs/predict_only_paper.txt || true
fi

BASELINE_DIR="$STORAGE_PATH/content/kolmogorov_re_1000"
echo "=== Plotting loss and eval vs baselines (optional --max_time_steps=200 to match notebook) ==="
python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" \
  --baseline_dir="$BASELINE_DIR" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" \
  --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label=LI \
  --max_time_steps=200 \
  2>&1 | tee ./logs/plot_eval_paper.txt || true

echo "Done. Output: $OUTPUT_DIR"
printf "=== Total wall time: %ds ===\n" $ELAPSED
