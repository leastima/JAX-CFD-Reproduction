#!/usr/bin/env bash
# Paper arxiv 2102.01010 on GPU 6,7. Same config as run_train_paper.sh.
# Oversample trajectory-start (t=0) windows so the model learns "first step from IC" better (LI curve from 1.0).
# About 54k steps (90 epochs, 2 GPUs). Total time typically 5-15 h; check log for steps/s.
set -e
set -x
export CUDA_VISIBLE_DEVICES=6,7
export JAX_PLATFORMS=cuda
[ -n "${CONDA_PREFIX}" ] && [ -d "${CONDA_PREFIX}/lib" ] && export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -n "${CONDA_PREFIX}" ] && ! ls "${CONDA_PREFIX}"/lib/libcudnn* 1>/dev/null 2>&1 && { echo "Install CuDNN: conda install -y -c nvidia cudnn"; exit 1; }
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0
STORAGE_PATH=/jumbo/yaoqingyang/yuxin/JAX-CFD
TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000/eval_2048x2048_64x64.nc
OUTPUT_DIR=$STORAGE_PATH/model_paper_gpu67
mkdir -p "$OUTPUT_DIR" ./logs
START_TIME=$(date +%s)
python -u models/train.py --model_encode_steps=64 --model_decode_steps=32 --model_predict_steps=64 --train_device_batch_size=128 --delta_time=0.007012483601762931 --train_split="$STORAGE_PATH/$TRAINDATA" --eval_split="$STORAGE_PATH/$EVALDATA" --eval_batch_size=256 --train_weight_decay=1e-4 --train_lr_init=0.001 --train_lr_warmup_epochs=5 --mp_scale_value=1.0 --train_epochs=90 --train_log_every=100 --decoding_warmup_steps=0 --oversample_trajectory_start_ratio=15.0 --mp_skip_nonfinite --do_eval --do_predict --predict_result=predict.nc --simulation_time=20.0 --output_dir="$OUTPUT_DIR" --gin_file="models/configs/official_li_config.gin" --gin_file="models/configs/kolmogorov_forcing.gin" --gin_param="fixed_scale.rescaled_one = 0.2" 2>&1 | tee ./logs/train_log_paper_gpu67.txt
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
python -u scripts/plot_loss_and_eval_from_predict.py --output_dir="$OUTPUT_DIR" --baseline_dir="$BASELINE_DIR" --baseline_nc="$STORAGE_PATH/$EVALDATA" --predict_nc="$OUTPUT_DIR/predict.nc" --model_label=LI --max_time_steps=200 2>&1 | tee ./logs/plot_eval_paper_gpu67.txt || true
echo "Done. Output: $OUTPUT_DIR"
