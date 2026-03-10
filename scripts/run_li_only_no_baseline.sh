#!/usr/bin/env bash
# 只跑：生成数据 -> 训 LI -> 画图。不跑 DNS baseline，用于先看 LI 涡量相关曲线是否正常。
# 用法：cd /jumbo/yaoqingyang/yuxin/JAX-CFD && bash scripts/run_li_only_no_baseline.sh
# 后台：nohup bash scripts/run_li_only_no_baseline.sh > logs/li_only.txt 2>&1 &
set -e
set -x
export CUDA_VISIBLE_DEVICES=4,5,6,7
export JAX_PLATFORMS=cuda
[ -n "${CONDA_PREFIX}" ] && [ -d "${CONDA_PREFIX}/lib" ] && export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -n "${CONDA_PREFIX}" ] && ! ls "${CONDA_PREFIX}"/lib/libcudnn* 1>/dev/null 2>&1; then
  echo "WARN: CuDNN not found. Continuing anyway..."
fi

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

STORAGE_PATH=/jumbo/yaoqingyang/yuxin/JAX-CFD
DATA_DIR=$STORAGE_PATH/content/kolmogorov_re_1000_gen
EVALDATA=content/kolmogorov_re_1000_gen/long_eval_2048x2048_64x64.nc
TRAINDATA=content/kolmogorov_re_1000_gen/train_2048x2048_64x64.nc
OUTPUT_DIR=$STORAGE_PATH/model_paper_gen_start_only
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" ./logs

echo "========== 1) Generate train + eval (与 overnight 相同参数) =========="
python -u data/generate_train_eval_nc.py \
  --output_dir "$DATA_DIR" \
  --train_samples 32 \
  --train_steps 4880 \
  --eval_samples 32 \
  --eval_steps 488 \
  --grid_size 64 \
  --delta_time 0.007012483601762931 \
  --seed 42
echo "Generated: $TRAINDATA, $EVALDATA"

echo "========== 1b) 参考 nc（供画图用，不跑 DNS baseline） =========="
cp "$STORAGE_PATH/$EVALDATA" "$DATA_DIR/eval_2048x2048_64x64.nc"
echo "Reference: $DATA_DIR/eval_2048x2048_64x64.nc"

echo "========== 2) Train LI =========="
START_TRAIN=$(date +%s)
python -u models/train.py \
  --model_encode_steps=64 --model_decode_steps=32 --model_predict_steps=64 \
  --train_device_batch_size=128 \
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
  2>&1 | tee ./logs/li_only_train_log.txt
TRAIN_EXIT=${PIPESTATUS[0]}
END_TRAIN=$(date +%s)
echo "=== Train wall time: $((END_TRAIN - START_TRAIN))s ==="
[ $TRAIN_EXIT -ne 0 ] && { echo "Train failed (exit $TRAIN_EXIT). Stop."; exit $TRAIN_EXIT; }

python -u -c "
import os, pickle
from flax.training import checkpoints
out='$OUTPUT_DIR'
state=checkpoints.restore_checkpoint(out, target=None)
if state is not None:
  with open(os.path.join(out,'LI_ckpt.pkl'),'wb') as f: pickle.dump(state,f)
  print('Saved LI_ckpt.pkl')
"

echo "========== 3) Plot LI vs reference（仅参考，无 baseline_64） =========="
python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" \
  --baseline_dir="$DATA_DIR" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" \
  --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label=LI \
  --max_time_steps=200 \
  2>&1 | tee ./logs/li_only_plot.txt || true

echo "========== 3b) 32-step 诊断图 =========="
python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" \
  --baseline_dir="$DATA_DIR" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" \
  --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label=LI \
  --max_time_steps=32 \
  --plot_suffix=_first32 \
  2>&1 | tee ./logs/li_only_plot_first32.txt || true

echo "========== Done. 看涡量相关: $OUTPUT_DIR/eval_vorticity_correlation.png (及 _first32) =========="
