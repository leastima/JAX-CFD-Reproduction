"""
可视化 train / eval / long_eval 数据的涡量场与时间步分布
输出：
  scripts/viz_vorticity_fields.png   -- 各数据集代表性时间步的涡量场
  scripts/viz_timestep_histogram.png -- 各数据集时间步间隔 & 速度场统计
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA = 'content/kolmogorov_re_1000'
OUT = 'scripts'

FILES = {
    'train\n(32s×4880t, dt=0.007)':     f'{DATA}/train_2048x2048_64x64.nc',
    'eval\n(32s×488t, dt=0.07)':         f'{DATA}/eval_2048x2048_64x64.nc',
    'long_eval\n(16s×3477t, dt=0.07)':  f'{DATA}/long_eval_2048x2048_64x64.nc',
}

# ------------------------------------------------------------------ #
# 1. 涡量场可视化（每个数据集取 sample 0，展示 5 个时间步）
# ------------------------------------------------------------------ #
def vorticity(ds, sample=0, t_idx=0):
    u = ds.u.isel(sample=sample, time=t_idx).values
    v = ds.v.isel(sample=sample, time=t_idx).values
    dx = float(ds.x.values[1] - ds.x.values[0])
    # 简单有限差分
    dvdx = np.gradient(v, dx, axis=0)
    dudy = np.gradient(u, dx, axis=1)
    return dvdx - dudy

n_rows = len(FILES)
n_cols = 6  # 5 snapshots + 1 空间均值 time-series 缩略

fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5 + 0.5))
fig1.suptitle('Vorticity snapshots: train / eval / long_eval  (sample 0)', fontsize=13)

for row, (label, path) in enumerate(FILES.items()):
    ds = xr.open_dataset(path)
    n_t = ds.sizes['time']
    # 选 5 个均匀分布的时间步
    t_inds = np.linspace(0, min(n_t - 1, 487), 5, dtype=int)
    t_vals = ds.time.values

    vmax = 0
    vors = []
    for ti in t_inds:
        w = vorticity(ds, sample=0, t_idx=ti)
        vors.append(w)
        vmax = max(vmax, np.abs(w).max())

    for col, (ti, w) in enumerate(zip(t_inds, vors)):
        ax = axes1[row, col]
        im = ax.imshow(w.T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='equal')
        ax.set_title(f't={t_vals[ti]:.1f}', fontsize=8)
        ax.axis('off')

    # 最后一列：涡量 RMS 随时间
    ax = axes1[row, n_cols - 1]
    rms_vals = []
    step = max(1, n_t // 200)
    for ti in range(0, n_t, step):
        w = vorticity(ds, sample=0, t_idx=ti)
        rms_vals.append(np.sqrt(np.mean(w ** 2)))
    ax.plot(ds.time.values[::step], rms_vals, lw=1)
    ax.set_xlabel('time', fontsize=7)
    ax.set_ylabel('ω RMS', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title('ω RMS(t)', fontsize=8)

    # 行标签
    axes1[row, 0].set_ylabel(label, fontsize=8, rotation=0, labelpad=70, va='center')
    ds.close()

fig1.tight_layout()
fig1.savefig(os.path.join(OUT, 'viz_vorticity_fields.png'), dpi=130)
plt.close(fig1)
print('Saved viz_vorticity_fields.png')


# ------------------------------------------------------------------ #
# 2. 时间步分布 & 速度统计
# ------------------------------------------------------------------ #
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 7))
fig2.suptitle('Time-step distribution & velocity statistics', fontsize=13)

# row 0: 时间步 histogram (dt)
# row 1: u 速度 histogram
colors = ['tab:blue', 'tab:orange', 'tab:green']

for col, (label, path) in enumerate(FILES.items()):
    ds = xr.open_dataset(path)
    t = ds.time.values
    dt = np.diff(t)
    u_flat = ds.u.isel(sample=0).values.ravel()

    # dt histogram
    ax = axes2[0, col]
    ax.hist(dt, bins=50, color=colors[col], edgecolor='k', lw=0.3)
    ax.set_title(f'{label.split(chr(10))[0]}  dt', fontsize=9)
    ax.set_xlabel('Δt', fontsize=8)
    ax.set_ylabel('count', fontsize=8)
    ax.tick_params(labelsize=7)
    stats = f'mean={dt.mean():.5f}\nstd={dt.std():.1e}\nmin={dt.min():.5f}\nmax={dt.max():.5f}'
    ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=7,
            va='top', ha='right', bbox=dict(boxstyle='round', alpha=0.3))

    # u histogram (sample 0)
    ax = axes2[1, col]
    ax.hist(u_flat, bins=80, color=colors[col], edgecolor='k', lw=0.2, density=True)
    ax.set_title(f'{label.split(chr(10))[0]}  u-velocity (sample 0, all t)', fontsize=9)
    ax.set_xlabel('u', fontsize=8)
    ax.set_ylabel('density', fontsize=8)
    ax.tick_params(labelsize=7)
    stats2 = f'mean={u_flat.mean():.3f}\nstd={u_flat.std():.3f}'
    ax.text(0.97, 0.97, stats2, transform=ax.transAxes, fontsize=7,
            va='top', ha='right', bbox=dict(boxstyle='round', alpha=0.3))
    ds.close()

fig2.tight_layout()
fig2.savefig(os.path.join(OUT, 'viz_timestep_histogram.png'), dpi=130)
plt.close(fig2)
print('Saved viz_timestep_histogram.png')
