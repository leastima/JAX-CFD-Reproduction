#!/usr/bin/env bash
# 激进测试：只采样「轨迹起点」窗口（与评估完全一致：前64步->预测32步），看是否能把 eval 曲线拉上去。
# 若此模式下 loss 下降且 vorticity correlation 明显变好，说明之前差是分布不匹配；若仍差则需查别处。
# 与评估对齐：评估用 200 时间步，故 only_trajectory_start_repeat=200（num_examples=32*200=6400），
#   train_epochs 取 17 时总训练步数 ≈ 200（6400*17/512≈212），与 200 步评估一致。
# 画图 --max_time_steps=200。使用 GPU 4,5,6,7 加速。
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
TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000/eval_2048x2048_64x64.nc
OUTPUT_DIR=$STORAGE_PATH/model_paper_gpu4567_start_only
mkdir -p "$OUTPUT_DIR" ./logs
START_TIME=$(date +%s)
python -u models/train.py \
  --model_encode_steps=64 --model_decode_steps=32 --model_predict_steps=64 \
  --train_device_batch_size=128 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" --eval_split="$STORAGE_PATH/$EVALDATA" \
  --eval_batch_size=256 \
  --train_weight_decay=1e-4 --train_lr_init=0.001 --train_lr_warmup_epochs=5 \
  --mp_scale_value=1.0 --train_epochs=17 --train_log_every=50 --decoding_warmup_steps=0 \
  --only_trajectory_start_windows --only_trajectory_start_repeat=200 \
  --mp_skip_nonfinite --do_eval --do_predict --predict_result=predict.nc \
  --simulation_time=20.0 --output_dir="$OUTPUT_DIR" \
  --gin_file="models/configs/official_li_config.gin" --gin_file="models/configs/kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.2" \
  2>&1 | tee ./logs/train_log_start_only_gpu4567.txt
TRAIN_EXIT=$?
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
python -u scripts/plot_loss_and_eval_from_predict.py --output_dir="$OUTPUT_DIR" --baseline_dir="$BASELINE_DIR" --baseline_nc="$STORAGE_PATH/$EVALDATA" --predict_nc="$OUTPUT_DIR/predict.nc" --model_label=LI --max_time_steps=200 2>&1 | tee ./logs/plot_eval_start_only_gpu4567.txt || true
echo "Done. Output: $OUTPUT_DIR"
