#!/usr/bin/env bash
# 一键：生成 train/eval 数据（与论文/现有格式一致、dt 统一）-> 只训起点 LI -> 画图。
# 可后台运行：nohup bash scripts/run_overnight_gen_data_and_li.sh > logs/overnight_gen_li.txt 2>&1 &
#
# 同一批轨迹既训又评：LI 的 train_split 和 eval_split 都指向 long_eval，这样训练和评估用同一 32 条轨迹。
# 与论文关系：论文 32 条轨迹、每条 4800 步；此处 train 32 条 4880 步（与现有 nc 一致），
# eval 32 条 488 步；train/eval 用同一 delta_time=0.00701248，故无 inner_steps=10 的显存问题。
# Baseline：跑 DNS（无 NN）得到 baseline_64，并把生成的 long_eval 拷为参考 eval_2048x2048_64x64.nc，
# 画图时 baseline_dir 下即有 baseline_64 + 参考，再加 LI。
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
TRAINDATA=content/kolmogorov_re_1000_gen/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000_gen/long_eval_2048x2048_64x64.nc
OUTPUT_DIR=$STORAGE_PATH/model_paper_gen_start_only
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" ./logs

echo "========== 1) Generate train + eval (32/4880, 32/488, same dt) =========="
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

echo "========== 1b) Baseline: 参考 + DNS@64 =========="
cp "$STORAGE_PATH/$EVALDATA" "$DATA_DIR/eval_2048x2048_64x64.nc"
echo "Reference: $DATA_DIR/eval_2048x2048_64x64.nc (copy of long_eval)"

BASELINE_DNS_DIR=$STORAGE_PATH/model_dns_64_gen
mkdir -p "$BASELINE_DNS_DIR"
# DNS 用 jax_cfd 的 DynamicalSystem.trajectory()，不接受 is_training；必须走 no_dropout/no_train 分支。
# 若出现 "GridArray does not contain all interior grid values"，系 DNS encoder 与 nc 网格布局不一致，可忽略此步。
python -u models/train.py \
  --model_encode_steps=32 \
  --model_decode_steps=16 \
  --model_predict_steps=16 \
  --delta_time=0.007012483601762931 \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --train_device_batch_size=4 \
  --eval_batch_size=16 \
  --train_epochs=0 \
  --no_train \
  --no_dropout \
  --do_predict \
  --simulation_time=20.0 \
  --output_dir="$BASELINE_DNS_DIR" \
  --gin_file="models/configs/implicit_diffusion_dns_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  2>&1 | tee ./logs/overnight_baseline_dns_log.txt || true
if [ -f "$BASELINE_DNS_DIR/predict.nc" ]; then
  cp "$BASELINE_DNS_DIR/predict.nc" "$DATA_DIR/eval_64x64_64x64.nc"
  echo "Baseline DNS@64: $DATA_DIR/eval_64x64_64x64.nc"
else
  echo "WARN: DNS baseline predict.nc not found, skip eval_64x64_64x64.nc"
fi

echo "========== 2) Train LI (only trajectory-start, 同一批轨迹既训又评: train_split=eval) =========="
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
  2>&1 | tee ./logs/overnight_train_log.txt
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

echo "========== 3) Plot LI vs baselines (reference + baseline_64) =========="
python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" \
  --baseline_dir="$DATA_DIR" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" \
  --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label=LI \
  --max_time_steps=200 \
  2>&1 | tee ./logs/overnight_plot.txt || true

echo "========== 3b) 32-step 诊断图（仅前 32 步=训练首 chunk，排查「先升后降」） =========="
python -u scripts/plot_loss_and_eval_from_predict.py \
  --output_dir="$OUTPUT_DIR" \
  --baseline_dir="$DATA_DIR" \
  --baseline_nc="$STORAGE_PATH/$EVALDATA" \
  --predict_nc="$OUTPUT_DIR/predict.nc" \
  --model_label=LI \
  --max_time_steps=32 \
  --plot_suffix=_first32 \
  2>&1 | tee ./logs/overnight_plot_first32.txt || true

echo "========== Done. Data: $DATA_DIR | Model/plots: $OUTPUT_DIR =========="
echo ""
echo "后台运行示例（当前脚本已跑完则忽略）："
echo "  cd $STORAGE_PATH && nohup bash scripts/run_overnight_gen_data_and_li.sh > logs/overnight_gen_li.txt 2>&1 &"
echo "  tail -f logs/overnight_gen_li.txt"
