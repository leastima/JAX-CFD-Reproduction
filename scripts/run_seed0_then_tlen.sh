#!/usr/bin/env bash
# 等 batch1 跑完 → 重启 train_200k.sh (seed=0 only) 跑完 batch2/3
# → 画 ntraj phase plot → 跑 tlen 实验 → 画 tlen phase plot

set -eo pipefail

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"

LOG="${ROOT}/logs/master_run.log"
mkdir -p "${ROOT}/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"; }

# ── 1. 等 batch1 的 8 个 train.py 进程跑完 ──────────────────────────
log "等待当前 batch1 (ntraj=2,4 seed=0 × 4Re) 完成..."
while true; do
    count=$(ps aux | grep "train.py" | grep -v grep | wc -l)
    if [[ $count -eq 0 ]]; then break; fi
    log "  $count 个 train.py 仍在运行，等 60s..."
    sleep 60
done
log "batch1 全部完成 ✓"

# ── 2. 重启 train_200k.sh (seed=0 only，已改好) ───────────────────────
log "启动 train_200k.sh (batch2/3: ntraj=8,16,32 seed=0)..."
bash "${ROOT}/scripts/train_200k.sh" >> "${ROOT}/logs/train_200k_batch23.log" 2>&1
log "ntraj 实验 seed=0 全部完成 ✓"

# ── 3. 画 ntraj phase plot ────────────────────────────────────────────
log "开始评估 ntraj models 并绘制 phase plot..."
bash "${ROOT}/scripts/eval_and_plot_200k.sh" ntraj >> "${ROOT}/logs/eval_ntraj.log" 2>&1
log "ntraj phase plot 完成 ✓"

# ── 4. 启动 tlen 实验 ─────────────────────────────────────────────────
log "启动 tlen 实验 (tlen=100,500,2000,4000 seed=0)..."
bash "${ROOT}/scripts/train_tlen_200k.sh" >> "${ROOT}/logs/train_tlen_200k.log" 2>&1
log "tlen 实验全部完成 ✓"

# ── 5. 画 tlen phase plot ─────────────────────────────────────────────
log "开始评估 tlen models 并绘制 phase plot..."
bash "${ROOT}/scripts/eval_and_plot_200k.sh" tlen >> "${ROOT}/logs/eval_tlen.log" 2>&1
log "tlen phase plot 完成 ✓"

log "===== 全部流程完成 ====="
