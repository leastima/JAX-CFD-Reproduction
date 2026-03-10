#!/usr/bin/env bash
# 验证「训练/评估是否同一批轨迹」：用评估集做训练（仍只训起点），评估也用评估集。
# 若此模式下 LI 曲线变好（从 1.0 单调下降），说明之前差是因为「用训练集训、用评估集评」轨迹不同；
# 评估集与训练集只是步数更少（488 vs 4880），格式一致，此处 train_split 和 eval_split 都指向 EVALDATA。
# 保留 only_trajectory_start_windows + repeat=200、epochs=17，与 start_only 设置一致。
#
# 注意：eval nc 的 time 步长是 0.07（train 是 0.007），DataLoader 用「数据 dt / FLAGS.delta_time」
# 算 inner_steps，故用 eval 时 inner_steps=10、用 train 时=1，同一段 unroll 显存约 10 倍，所以
# 只有「用评估集当训练数据」时才会 OOM；之前都用的 train nc，所以没出过问题。
# train_device_batch_size=8（总 batch 32）以压住显存。
set -e
set -x
export CUDA_VISIBLE_DEVICES=4,5,6,7
export JAX_PLATFORMS=cuda
[ -n "${CONDA_PREFIX}" ] && [ -d "${CONDA_PREFIX}/lib" ] && export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -n "${CONDA_PREFIX}" ] && ! ls "${CONDA_PREFIX}"/lib/libcudnn* 1>/dev/null 2>&1; then
  echo "WARN: CuDNN not found in conda env. Install with: conda install -y -c nvidia cudnn"
  echo "      Or use env with CuDNN (e.g. cfd). Continuing anyway..."
fi
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0
STORAGE_PATH=/jumbo/yaoqingyang/yuxin/JAX-CFD
EVALDATA=content/kolmogorov_re_1000/eval_2048x2048_64x64.nc
OUTPUT_DIR=$STORAGE_PATH/model_paper_gpu4567_start_only_on_eval
mkdir -p "$OUTPUT_DIR" ./logs
START_TIME=$(date +%s)
python -u models/train.py \
  --model_encode_steps=64 --model_decode_steps=32 --model_predict_steps=64 \
  --train_device_batch_size=8 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$EVALDATA" --eval_split="$STORAGE_PATH/$EVALDATA" \
  --eval_batch_size=256 \
  --train_weight_decay=1e-4 --train_lr_init=0.001 --train_lr_warmup_epochs=5 \
  --mp_scale_value=1.0 --train_epochs=17 --train_log_every=50 --decoding_warmup_steps=0 \
  --only_trajectory_start_windows --only_trajectory_start_repeat=200 \
  --mp_skip_nonfinite --do_eval --do_predict --predict_result=predict.nc \
  --simulation_time=20.0 --output_dir="$OUTPUT_DIR" \
  --gin_file="models/configs/official_li_config.gin" --gin_file="models/configs/kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.2" \
  2>&1 | tee ./logs/train_log_start_only_on_eval_gpu4567.txt
TRAIN_EXIT=${PIPESTATUS[0]}
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf "=== Wall time: %ds (%dm) ===\n" $ELAPSED $((ELAPSED/60))
[ $TRAIN_EXIT -ne 0 ] && exit $TRAIN_EXIT
python -u -c "
import os, pickle
from flax.training import checkpoints
out='$OUTPUT_DIR'
state=checkpoints.restore_checkpoint(out, target=None)
if state is not None:
  with open(os.path.join(out,'LI_ckpt.pkl'),'wb') as f: pickle.dump(state,f)
  print('Saved LI_ckpt.pkl')
"
BASELINE_DIR="$STORAGE_PATH/content/kolmogorov_re_1000"
python -u scripts/plot_loss_and_eval_from_predict.py --output_dir="$OUTPUT_DIR" --baseline_dir="$BASELINE_DIR" --baseline_nc="$STORAGE_PATH/$EVALDATA" --predict_nc="$OUTPUT_DIR/predict.nc" --model_label=LI --max_time_steps=200 2>&1 | tee ./logs/plot_eval_start_only_on_eval_gpu4567.txt || true
echo "Done. Output: $OUTPUT_DIR"
