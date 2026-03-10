#!/usr/bin/env bash
# 仿照 ML4HPC/JAX-CFD 的流程：用已有数据 → 训练（极简 smoke）→ 画图。
# 不生成数据、不跑 DNS baseline、不用 only_trajectory_start，尽量贴近「能跑通」的原始流程。
# 数据路径与 run_train_smoke.sh / run_train_paper.sh 一致：content/kolmogorov_re_1000/
#
# 用法：先激活环境（如 conda activate cfd-gpu），再：
#   cd /jumbo/yaoqingyang/yuxin/JAX-CFD && bash scripts/run_restore_smoke.sh
# 若只有 4–7 卡可用，脚本已限制为 CUDA_VISIBLE_DEVICES=4,5,6,7 且 num_workers=0 避免 netCDF 资源错误。
set -e
set -x
set -o pipefail
# 只用 4–7 卡时设置（避免 8 卡导致资源竞争）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
if [ "${JAX_PLATFORMS}" = "gpu" ]; then
  export JAX_PLATFORMS=cuda
fi
export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
[ -n "${CONDA_PREFIX}" ] && [ -d "${CONDA_PREFIX}/lib" ] && export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HAIKU_FLATMAPPING=0

STORAGE_PATH=/jumbo/yaoqingyang/yuxin/JAX-CFD
TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000/eval_2048x2048_64x64.nc
OUTPUT_DIR=$STORAGE_PATH/model_restore_smoke
mkdir -p "$OUTPUT_DIR" ./logs

if [ ! -f "$STORAGE_PATH/$TRAINDATA" ] || [ ! -f "$STORAGE_PATH/$EVALDATA" ]; then
  echo "Missing data. Expected: $STORAGE_PATH/$TRAINDATA and $EVALDATA"
  echo "Create content/kolmogorov_re_1000/ and put train/eval nc there, or run data/generate_train_eval_nc.py and symlink/copy."
  exit 1
fi

echo "========== 1) Train (smoke: 2 encode/decode/predict, 0.002 epoch) =========="
python -u models/train.py \
  --model_encode_steps=2 \
  --model_decode_steps=2 \
  --model_predict_steps=2 \
  --train_device_batch_size=2 \
  --dataset_num_workers=0 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --eval_batch_size=32 \
  --train_weight_decay=0.0 \
  --train_lr_init=0.0001 \
  --train_lr_warmup_epochs=0.0 \
  --mp_scale_value=1.0 \
  --train_epochs=0.002 \
  --train_log_every=1 \
  --decoding_warmup_steps=0 \
  --mp_skip_nonfinite \
  --do_eval \
  --do_predict \
  --predict_result=predict.nc \
  --simulation_time=0.2 \
  --output_dir="$OUTPUT_DIR" \
  --gin_file="models/configs/official_li_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  --gin_param="fixed_scale.rescaled_one = 0.2" \
  --gin_param="my_forward_tower_factory.num_hidden_channels = 16" \
  --gin_param="my_forward_tower_factory.num_hidden_layers = 1" \
  --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
  2>&1 | tee ./logs/restore_smoke_train.txt
TRAIN_EXIT=${PIPESTATUS[0]}
[ $TRAIN_EXIT -ne 0 ] && { echo "Train failed (exit $TRAIN_EXIT)."; exit $TRAIN_EXIT; }

echo "========== 2) Plot loss + eval vs baseline =========="
BASELINE_DIR="$STORAGE_PATH/content/kolmogorov_re_1000"
python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" \
  --baseline_dir="$BASELINE_DIR" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" \
  --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label=LI \
  2>&1 | tee ./logs/restore_smoke_plot.txt || true

echo "========== Done. Output: $OUTPUT_DIR (train_loss_curve.png, eval_*.png) =========="
