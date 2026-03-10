"""
绘制 Phase Plot
读取 results/phase_metrics.csv，对 seed 取平均，
生成 3 张 phase plot（training loss、mean L2、t_0.95）

用法:
  python scripts/plot_phase.py \
      --csv results/phase_metrics.csv \
      --output_dir results/
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser()
parser.add_argument('--csv',        required=True, help='Path to phase_metrics CSV')
parser.add_argument('--output_dir', required=True, help='Directory for output PNGs')
parser.add_argument('--yparam',     default='ntraj',
                    help='Y-axis parameter column name (ntraj or tlen)')
parser.add_argument('--ylabel',     default=None,
                    help='Y-axis label override (e.g. "# Training Trajectories")')
parser.add_argument('--title',      default=None,
                    help='Suptitle override')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
YPARAM = args.yparam
YLABEL = args.ylabel or ('# Training Trajectories' if YPARAM == 'ntraj' else 'Trajectory Length (frames)')

df = pd.read_csv(args.csv)
print(f'Loaded {len(df)} rows from {args.csv}')
print(df)

# 对 seed 取均值（nanmean 以便 NaN 单元格跳过）
def nanmean(x): return np.nanmean(x) if not x.isna().all() else np.nan
def nanstd(x):  return np.nanstd(x)  if not x.isna().all() else np.nan

agg_dict = dict(
    train_loss_mean=('final_train_loss', nanmean),
    train_loss_std =('final_train_loss', nanstd),
    t095_mean      =('t_095',            nanmean),
    t095_std       =('t_095',            nanstd),
    l2_mean        =('mean_l2',          nanmean),
    l2_std         =('mean_l2',          nanstd),
    corr_mean      =('mean_corr',        nanmean),
    corr_std       =('mean_corr',        nanstd),
)
# 逐时间步指标（含新增的 t=100）
for col in ['l2_t10', 'l2_t30', 'l2_t50', 'l2_t100', 'corr_t10', 'corr_t30', 'corr_t50', 'corr_t100']:
    if col in df.columns:
        agg_dict[f'{col}_mean'] = (col, nanmean)
agg = df.groupby(['re', YPARAM]).agg(**agg_dict).reset_index()

RE_LIST     = sorted(df['re'].unique())
YPARAM_LIST = sorted(df[YPARAM].unique())

def make_matrix(col):
    """Build 2D array: rows=yparam (ascending), cols=re (ascending)"""
    mat = np.full((len(YPARAM_LIST), len(RE_LIST)), np.nan)
    for _, row in agg.iterrows():
        i = YPARAM_LIST.index(int(row[YPARAM]))
        j = RE_LIST.index(int(row['re']))
        mat[i, j] = row[col]
    return mat

metrics = [
    ('train_loss_mean',  'Final Train Loss',                   'viridis_r', True),
    ('t095_mean',        'Time until Corr < 0.95',             'RdYlGn',    False),
    ('l2_t50_mean',      'L2 Error @ t=50',                    'RdYlGn_r',  False),
    ('corr_t50_mean',    'Vorticity Corr @ t=50',              'RdYlGn',    False),
    ('l2_t100_mean',     'L2 Error @ t=100',                   'RdYlGn_r',  False),
    ('corr_t100_mean',   'Vorticity Corr @ t=100',             'RdYlGn',    False),
]

def draw_heatmap(ax, mat, col, title, cmap, log_scale):
    valid = mat[np.isfinite(mat) & (mat > 0)] if log_scale else mat[np.isfinite(mat)]
    if len(valid) == 0:
        ax.set_title(title + '\n(no data)')
        return None
    norm = mcolors.LogNorm(vmin=valid.min(), vmax=valid.max()) if log_scale \
           else mcolors.Normalize(vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    # 用灰色背景显示 NaN 格子
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, aspect='auto', cmap=cmap, norm=norm, origin='lower')
    ax.set_facecolor('#cccccc')  # NaN 格显示为灰色
    for i in range(len(YPARAM_LIST)):
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
    ax.set_yticks(range(len(YPARAM_LIST)))
    ax.set_yticklabels([f'{YPARAM}={n}' for n in YPARAM_LIST])
    ax.set_xlabel('Reynolds Number (Re)')
    ax.set_ylabel(YLABEL)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.85)
    return im

fig, axes = plt.subplots(2, 3, figsize=(21, 10))
for ax, (col, title, cmap, log_scale) in zip(axes.flatten(), metrics):
    draw_heatmap(ax, make_matrix(col), col, title, cmap, log_scale)

suptitle = args.title or f'Phase Plot: LI Model Performance vs Re and {YPARAM}\n(averaged over available seeds)'
plt.suptitle(suptitle, fontsize=13, y=1.02)
plt.tight_layout()

out_path = os.path.join(args.output_dir, 'phase_plot.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close(fig)

# 也保存各指标单独的大图
for col, title, cmap, log_scale in metrics:
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    mat = make_matrix(col)
    draw_heatmap(ax2, mat, col, title, cmap, log_scale)
    ax2.set_title(f'{title}\n(mean over seeds)')
    plt.tight_layout()
    fname = col.replace('_mean', '') + '.png'
    out2 = os.path.join(args.output_dir, fname)
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {out2}')

print('Done.')
