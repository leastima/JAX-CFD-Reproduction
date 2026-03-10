"""
仿照 ml_model_inference_demo.ipynb 可视化 re=1000 tlen=4000 模型：
- Vorticity fields (多时间点)
- Vorticity correlation vs time
- Energy spectrum (k^5 * E(k))
"""
import sys, os, functools, warnings
warnings.simplefilter('ignore')

JAX_CFD_ROOT = '/jumbo/yaoqingyang/yuxin/JAX-CFD'
sys.path.insert(0, os.path.join(JAX_CFD_ROOT, 'jax-cfd'))
sys.path.insert(0, os.path.join(JAX_CFD_ROOT, 'models'))

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.6'

import gin
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import xarray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
from flax.training import checkpoints

import jax_cfd.base as cfd
import jax_cfd.data as cfd_data
import jax_cfd.ml as ml

model_builder          = ml.model_builder
physics_specifications = ml.physics_specifications

# ── 配置 ─────────────────────────────────────────────────────────────
MODEL_DIR   = f'{JAX_CFD_ROOT}/models_tlen_200k/re1000_tlen4000_seed0'
EVAL_NC     = f'{JAX_CFD_ROOT}/content/kolmogorov_re1000/long_eval_2048x2048_64x64.nc'
TRAIN_NC    = f'{JAX_CFD_ROOT}/content/kolmogorov_re1000/train_2048x2048_64x64.nc'
OUT_DIR     = f'{JAX_CFD_ROOT}/results/tlen_200k/viz_re1000_tlen4000'
RE          = 1000
LENGTH      = 200
INNER_STEPS = 10   # long_eval save_dt=0.07012 = 10 × train save_dt → 每帧走10步
SAMPLE_ID   = 0

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. 加载数据 ───────────────────────────────────────────────────────
print('Loading eval data...')
reference_ds = xarray.open_dataset(EVAL_NC)
train_ds     = xarray.open_dataset(TRAIN_NC)
grid = cfd_data.xarray_utils.grid_from_attrs(reference_ds.attrs)

initial_conditions = tuple(
    reference_ds[v].isel(sample=slice(SAMPLE_ID, SAMPLE_ID+1),
                          time=slice(0, 1)).values
    for v in cfd_data.xarray_utils.XR_VELOCITY_NAMES[:grid.ndim]
)
target_ds = reference_ds.isel(
    sample=slice(SAMPLE_ID, SAMPLE_ID+1),
    time=slice(0, LENGTH)
)

# ── 2. 加载 checkpoint & 构建模型 ─────────────────────────────────────
print('Loading checkpoint...')
ckpt_state = checkpoints.restore_checkpoint(MODEL_DIR, target=None)
params = ckpt_state['params']

def strip_imports(s):
    return '\n'.join(l for l in s.splitlines() if not l.startswith('import'))

print('Setting up model...')
gin.clear_config()
gin.parse_config_files_and_bindings(
    [
        os.path.join(JAX_CFD_ROOT, 'models/configs/official_li_config.gin'),
        os.path.join(JAX_CFD_ROOT, 'models/configs/kolmogorov_forcing.gin'),
    ],
    [
        'fixed_scale.rescaled_one = 0.2',
        'my_forward_tower_factory.num_hidden_channels = 128',
        'my_forward_tower_factory.num_hidden_layers = 6',
        'MyFusedLearnedInterpolation.pattern = "simple"',
    ],
    finalize_config=False,
)
gin.parse_config(strip_imports(train_ds.attrs['physics_config_str']))
dt = float(train_ds.attrs['stable_time_step'])

physics_specs = physics_specifications.get_physics_specs()
model_cls = model_builder.get_model_cls(grid, dt, physics_specs)

def compute_trajectory_fwd(x):
    solver = model_cls()
    x = solver.encode(x)
    _, trajectory = solver.trajectory(
        x, LENGTH, INNER_STEPS,
        start_with_input=True, post_process_fn=solver.decode)
    return trajectory

model       = hk.without_apply_rng(hk.transform(compute_trajectory_fwd))
trajectory_fn = jax.vmap(functools.partial(model.apply, params))

# ── 3. 推理 ──────────────────────────────────────────────────────────
print('Running inference...')
prediction = trajectory_fn(initial_conditions)
prediction_ds = cfd_data.xarray_utils.velocity_trajectory_to_xarray(
    prediction, grid, samples=True)
prediction_ds.coords['x']    = target_ds.coords['x']
prediction_ds.coords['y']    = target_ds.coords['y']
prediction_ds.coords['time'] = target_ds.coords['time']

datasets = {
    'DNS (reference)':      target_ds,
    'LI model (tlen=4000)': prediction_ds,
}

# ── 4. 评估指标 ───────────────────────────────────────────────────────
print('Computing metrics...')
summary = xarray.concat([
    cfd_data.evaluation.compute_summary_dataset(ds, target_ds)
    for ds in datasets.values()
], dim='model')
summary.coords['model'] = list(datasets.keys())

correlation = summary.vorticity_correlation.compute()
spectrum    = summary.energy_spectrum_mean.mean('time').compute()

palette = ['tab:blue', 'tab:orange']

# ── 5. 图1: Vorticity correlation ─────────────────────────────────────
print('Plotting correlation...')
fig, ax = plt.subplots(figsize=(8, 5))
for color, mname in zip(palette, summary['model'].data):
    style = '-' if 'DNS' in mname else '--'
    correlation.sel(model=mname).plot.line(
        ax=ax, color=color, linestyle=style, label=mname, linewidth=2.5)
ax.axhline(y=0.95, color='gray', linestyle=':', linewidth=1.5, label='0.95 threshold')
ax.set_xlim(0, float(target_ds.coords['time'].values[-1]))
ax.set_ylim(0, 1.05)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Vorticity Correlation')
ax.set_title(f'Vorticity Correlation — Re={RE}, tlen=4000')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/vorticity_correlation.png', dpi=150)
plt.close()
print(f'  Saved: vorticity_correlation.png')

# ── 6. 图2: Energy spectrum ───────────────────────────────────────────
print('Plotting energy spectrum...')
fig, ax = plt.subplots(figsize=(8, 5))
for color, mname in zip(palette, summary['model'].data):
    style = '-' if 'DNS' in mname else '--'
    (spectrum.k ** 5 * spectrum).sel(model=mname).plot.line(
        ax=ax, color=color, linestyle=style, label=mname, linewidth=2.5)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(3.5, None)
ax.set_ylim(1e9, None)
ax.set_xlabel('Wavenumber k')
ax.set_ylabel('k⁵ · E(k)')
ax.set_title(f'Energy Spectrum — Re={RE}, tlen=4000')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/energy_spectrum.png', dpi=150)
plt.close()
print(f'  Saved: energy_spectrum.png')

# ── 7. 图3: Vorticity fields ──────────────────────────────────────────
print('Plotting vorticity fields...')
vorticities = xarray.concat(
    [cfd_data.xarray_utils.vorticity_2d(ds) for ds in datasets.values()],
    dim='model'
).to_dataset()
vorticities.coords['model'] = list(datasets.keys())

num_to_show  = 5
time_indices = np.linspace(0, LENGTH - 1, num_to_show, dtype=int)
cmap = seaborn.cm.icefire

fig, axes = plt.subplots(len(datasets), num_to_show,
                          figsize=(num_to_show * 3.2, len(datasets) * 3.2))
for row, mname in enumerate(datasets.keys()):
    vort = vorticities['vorticity'].sel(model=mname).isel(sample=0)
    vmax = float(np.abs(vort.values).max()) * 0.8
    for col, t_idx in enumerate(time_indices):
        ax = axes[row, col]
        frame  = vort.isel(time=t_idx).values
        t_val  = float(vort.coords['time'].isel(time=t_idx))
        ax.imshow(frame.T, origin='lower', cmap=cmap,
                  vmin=-vmax, vmax=vmax, aspect='equal')
        if row == 0:
            ax.set_title(f't = {t_val:.2f}s', fontsize=9)
        if col == 0:
            ax.set_ylabel(mname, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle(f'Vorticity Fields — Re={RE}, tlen=4000', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/vorticity_fields.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: vorticity_fields.png')

print(f'\n===== 完成 =====')
print(f'输出目录: {OUT_DIR}')
