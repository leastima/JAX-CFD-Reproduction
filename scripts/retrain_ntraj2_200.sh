#!/usr/bin/env bash
# 补跑 ntraj=2，200 步（与 models_phase 同配置）
# 然后重新评估，合并到 v2 CSV，生成3格 phase plot：train_loss / l2_t10 / t_095

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80   # 每 GPU 1 个进程

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
mkdir -p "${ROOT}/logs/retrain_ntraj2"

# -------------------------------------------------------
# 训练函数：与 train_phase_plot.sh 完全相同参数，但用 --train_steps=200
# -------------------------------------------------------
train_ntraj2() {
    local RE=$1 SEED=$2 GPU=$3
    local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
    local MODEL_DIR="${ROOT}/models_phase/re${RE}_ntraj2_seed${SEED}"
    local LOG="${ROOT}/logs/retrain_ntraj2/re${RE}_seed${SEED}.log"

    # 清理旧 checkpoint（保留 train_loss.csv 如已存在）
    rm -f "${MODEL_DIR}"/checkpoint_* 2>/dev/null || true
    rm -f "${MODEL_DIR}"/train_loss.csv 2>/dev/null || true
    mkdir -p "${MODEL_DIR}"

    echo "[re${RE} ntraj2 seed${SEED} GPU${GPU}] start 200 steps..."
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/models/train.py \
        --gin_file="${ROOT}/models/configs/official_li_config.gin" \
        --gin_file="${ROOT}/models/configs/kolmogorov_forcing.gin" \
        --gin_param="fixed_scale.rescaled_one = 0.2" \
        --gin_param="my_forward_tower_factory.num_hidden_channels = 128" \
        --gin_param="my_forward_tower_factory.num_hidden_layers = 6" \
        --gin_param="MyFusedLearnedInterpolation.pattern = \"simple\"" \
        --train_split="${TRAIN_NC}" \
        --eval_split="${TRAIN_NC}" \
        --train_steps=200 \
        --train_init_random_seed=${SEED} \
        --max_train_samples=2 \
        --dataset_num_workers=0 \
        --output_dir="${MODEL_DIR}" \
        > "${LOG}" 2>&1
    echo "[re${RE} ntraj2 seed${SEED}] done ✓"
}

# -------------------------------------------------------
# 评估函数
# -------------------------------------------------------
eval_ntraj2() {
    local RE=$1 SEED=$2 GPU=$3
    local MODEL_DIR="${ROOT}/models_phase/re${RE}_ntraj2_seed${SEED}"
    local EVAL_NC="${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc"
    local TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
    local LOG="${ROOT}/logs/retrain_ntraj2/eval_re${RE}_seed${SEED}.log"

    echo "[re${RE} ntraj2 seed${SEED} GPU${GPU}] evaluating..."
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u ${ROOT}/scripts/eval_one_model.py \
        --model_dir "${MODEL_DIR}" \
        --eval_nc   "${EVAL_NC}" \
        --train_nc  "${TRAIN_NC}" \
        --re ${RE} --ntraj 2 --seed ${SEED} \
        --output_csv "${ROOT}/results/phase_metrics_ntraj2_200.csv" \
        --length 200 --inner_steps 10 \
        > "${LOG}" 2>&1 \
        && echo "  ✓ eval re${RE}_ntraj2_seed${SEED}" \
        || echo "  ✗ eval re${RE}_ntraj2_seed${SEED} (check ${LOG})"
}

# -------------------------------------------------------
# STEP 1: 训练（4 GPU 并行，每 GPU 跑 3 seeds 串行）
# -------------------------------------------------------
echo "===== STEP 1: Training ntraj=2 (200 steps) ====="

( train_ntraj2 500  0 4; train_ntraj2 500  1 4; train_ntraj2 500  2 4; echo "GPU4 done" ) &
( train_ntraj2 1000 0 5; train_ntraj2 1000 1 5; train_ntraj2 1000 2 5; echo "GPU5 done" ) &
( train_ntraj2 2000 0 6; train_ntraj2 2000 1 6; train_ntraj2 2000 2 6; echo "GPU6 done" ) &
( train_ntraj2 3000 0 7; train_ntraj2 3000 1 7; train_ntraj2 3000 2 7; echo "GPU7 done" ) &

wait
echo "===== All ntraj=2 training done ====="

# -------------------------------------------------------
# STEP 2: 评估（4 GPU 并行）
# -------------------------------------------------------
echo "===== STEP 2: Evaluating ntraj=2 ====="
rm -f "${ROOT}/results/phase_metrics_ntraj2_200.csv"

( eval_ntraj2 500  0 4; eval_ntraj2 500  1 4; eval_ntraj2 500  2 4; echo "GPU4 eval done" ) &
( eval_ntraj2 1000 0 5; eval_ntraj2 1000 1 5; eval_ntraj2 1000 2 5; echo "GPU5 eval done" ) &
( eval_ntraj2 2000 0 6; eval_ntraj2 2000 1 6; eval_ntraj2 2000 2 6; echo "GPU6 eval done" ) &
( eval_ntraj2 3000 0 7; eval_ntraj2 3000 1 7; eval_ntraj2 3000 2 7; echo "GPU7 eval done" ) &

wait
echo "===== All ntraj=2 eval done ====="

# -------------------------------------------------------
# STEP 3: 合并 CSV + 生成 3 格 phase plot
# -------------------------------------------------------
echo "===== STEP 3: Merge CSV and plot ====="

${PYTHON} - <<'PYEOF'
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = '/jumbo/yaoqingyang/yuxin/JAX-CFD'

# 读取旧 v2 CSV（ntraj=8/16/32 行）
df_old = pd.read_csv(f'{ROOT}/results/phase_metrics_ntraj_v2.csv')
df_old = df_old[df_old['ntraj'] != 2].copy()

# 读取新 ntraj=2 数据
df_new = pd.read_csv(f'{ROOT}/results/phase_metrics_ntraj2_200.csv')

# 合并
df = pd.concat([df_old, df_new], ignore_index=True)
print(f'Combined: {len(df)} rows')
print(df[['re','ntraj','seed','final_train_loss','t_095','l2_t10']].sort_values(['ntraj','re','seed']))

# 保存合并后的 CSV
out_csv = f'{ROOT}/results/phase_metrics_ntraj_v3.csv'
df.to_csv(out_csv, index=False)
print(f'Saved merged CSV: {out_csv}')

# 对 seed 取均值
def nanmean(x): return np.nanmean(x) if not x.isna().all() else np.nan
agg = df.groupby(['re', 'ntraj']).agg(
    train_loss_mean=('final_train_loss', nanmean),
    t095_mean      =('t_095',            nanmean),
    l2t10_mean     =('l2_t10',           nanmean),
).reset_index()

RE_LIST    = sorted(df['re'].unique())
NTRAJ_LIST = sorted(df['ntraj'].unique())

def make_matrix(col):
    mat = np.full((len(NTRAJ_LIST), len(RE_LIST)), np.nan)
    for _, row in agg.iterrows():
        i = NTRAJ_LIST.index(int(row['ntraj']))
        j = RE_LIST.index(int(row['re']))
        mat[i, j] = row[col]
    return mat

def draw_heatmap(ax, mat, title, cmap, log_scale):
    valid = mat[np.isfinite(mat) & (mat > 0)] if log_scale else mat[np.isfinite(mat)]
    if len(valid) == 0:
        ax.set_title(title + '\n(no data)'); return
    norm = mcolors.LogNorm(vmin=valid.min(), vmax=valid.max()) if log_scale \
           else mcolors.Normalize(vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, aspect='auto', cmap=cmap, norm=norm, origin='lower')
    ax.set_facecolor('#cccccc')
    for i in range(len(NTRAJ_LIST)):
        for j in range(len(RE_LIST)):
            v = mat[i, j]
            if np.isfinite(v):
                txt = f'{v:.2e}' if log_scale else f'{v:.2f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                        color='white' if norm(v) > 0.6 else 'black')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=8, color='#555555')
    ax.set_xticks(range(len(RE_LIST)))
    ax.set_xticklabels([f'Re={r}' for r in RE_LIST], rotation=30, ha='right')
    ax.set_yticks(range(len(NTRAJ_LIST)))
    ax.set_yticklabels([f'ntraj={n}' for n in NTRAJ_LIST])
    ax.set_xlabel('Reynolds Number (Re)')
    ax.set_ylabel('# Training Trajectories')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.85)

metrics = [
    ('train_loss_mean', 'Final Train Loss',               'viridis_r', True),
    ('l2t10_mean',      'L2 Error @ t=10 steps',          'RdYlGn_r',  False),
    ('t095_mean',       'Time until Corr < 0.95 (s)',     'RdYlGn',    False),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (col, title, cmap, log_scale) in zip(axes, metrics):
    draw_heatmap(ax, make_matrix(col), title, cmap, log_scale)

plt.suptitle('Phase Plot: LI Model vs Re and # Trajectories\n(1-epoch, seed averaged)', fontsize=13, y=1.02)
plt.tight_layout()

out = f'{ROOT}/results/phase_plot_v3.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()
PYEOF

echo "===== Done! Plot: ${ROOT}/results/phase_plot_v3.png ====="
