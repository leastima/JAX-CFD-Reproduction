"""
仿照 ml_model_inference_demo.ipynb 的评估脚本
用我们自训练的 LI 模型做推理，生成相关性曲线、能谱、L2误差和涡度场图。

用法：
  CUDA_VISIBLE_DEVICES=4 python scripts/run_inference_eval.py \
      --model_dir model_fixed_1ep \
      --output_dir model_fixed_1ep \
      --eval_nc content/my_kolmogorov_re1000/long_eval_2048x2048_64x64.nc \
      --train_nc content/my_kolmogorov_re1000/train_2048x2048_64x64.nc \
      --length 200 \
      --inner_steps 10
"""
import argparse, functools, os, sys, warnings
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',  default='model_fixed_1ep',
                    help='Orbax checkpoint 所在目录（相对 JAX_CFD_ROOT 或绝对路径）')
parser.add_argument('--output_dir', default=None,
                    help='保存图片的目录，默认同 model_dir')
parser.add_argument('--eval_nc',    default=None,
                    help='评估参考数据集路径（ground truth）')
parser.add_argument('--train_nc',   default=None,
                    help='训练数据集路径（用于读取 physics_config 和 stable_time_step）')
parser.add_argument('--baselines_dir', default=None,
                    help='baseline nc 文件目录（默认用原始 content/kolmogorov_re_1000）')
parser.add_argument('--length',     type=int, default=200,
                    help='预测轨迹长度（参考数据的时间步数）')
parser.add_argument('--inner_steps',type=int, default=10,
                    help='每个参考时间步内模型的内部步数')
parser.add_argument('--sample_id',  type=int, default=0)
parser.add_argument('--time_id',    type=int, default=0)
parser.add_argument('--model_label',default=None,
                    help='图例中显示的模型名称，默认用目录名')
args = parser.parse_args()

# ===== 路径设置 =====
JAX_CFD_ROOT = '/jumbo/yaoqingyang/yuxin/JAX-CFD'
sys.path.insert(0, JAX_CFD_ROOT)
sys.path.insert(0, os.path.join(JAX_CFD_ROOT, 'models'))
sys.path.insert(0, os.path.join(JAX_CFD_ROOT, 'jax-cfd'))

def _abs(p):
    return p if os.path.isabs(p) else os.path.join(JAX_CFD_ROOT, p)

MODEL_DIR  = _abs(args.model_dir)
OUTPUT_DIR = _abs(args.output_dir or args.model_dir)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ORIG_BASE  = os.path.join(JAX_CFD_ROOT, 'content/kolmogorov_re_1000')
BASELINES_DIR = _abs(args.baselines_dir) if args.baselines_dir else ORIG_BASE

EVAL_NC   = _abs(args.eval_nc)   if args.eval_nc   else os.path.join(ORIG_BASE, 'long_eval_2048x2048_64x64.nc')
TRAIN_NC  = _abs(args.train_nc)  if args.train_nc  else os.path.join(ORIG_BASE, 'train_2048x2048_64x64.nc')
MODEL_LABEL = args.model_label or os.path.basename(MODEL_DIR.rstrip('/'))

print(f'Model dir  : {MODEL_DIR}')
print(f'Output dir : {OUTPUT_DIR}')
print(f'Eval NC    : {EVAL_NC}')
print(f'Train NC   : {TRAIN_NC}')
print(f'Baselines  : {BASELINES_DIR}')
print(f'length={args.length}, inner_steps={args.inner_steps}')

# ===== 导入 =====
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
import my_model_builder, my_equations, my_interpolations, my_towers
import my_advections

model_builder        = ml.model_builder
physics_specifications = ml.physics_specifications

print('JAX devices:', jax.devices())

# ===== 1. 加载参考数据集 =====
print('\n[1/5] Loading reference dataset...')
reference_ds = xarray.open_dataset(EVAL_NC)
grid = cfd_data.xarray_utils.grid_from_attrs(reference_ds.attrs)
print(f'  grid: {grid}')
print(f'  dims: {dict(reference_ds.dims)}')

sample_id   = args.sample_id
time_id     = args.time_id
length      = args.length
inner_steps = args.inner_steps

initial_conditions = tuple(
    reference_ds[v].isel(
        sample=slice(sample_id, sample_id + 1),
        time=slice(time_id,     time_id     + 1),
    ).values
    for v in cfd_data.xarray_utils.XR_VELOCITY_NAMES[:grid.ndim]
)
target_ds = reference_ds.isel(
    sample=slice(sample_id, sample_id + 1),
    time=slice(time_id, time_id + length),
)
print(f'  IC shapes : {[x.shape for x in initial_conditions]}')
print(f'  Target    : {dict(target_ds.dims)}')

# ===== 2. 加载 Checkpoint =====
print('\n[2/5] Loading checkpoint...')
ckpt_state = checkpoints.restore_checkpoint(MODEL_DIR, target=None)
params     = ckpt_state['params']
step       = int(ckpt_state.get('step', -1))
print(f'  step: {step}')

# ===== 3. 构建模型 =====
print('\n[3/5] Building model...')

def strip_imports(s):
    return '\n'.join(l for l in s.splitlines() if not l.startswith('import'))

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
train_ds = xarray.open_dataset(TRAIN_NC)
gin.parse_config(strip_imports(train_ds.attrs['physics_config_str']))

dt = float(train_ds.attrs['stable_time_step'])
print(f'  dt = {dt:.7f}')

physics_specs_obj = physics_specifications.get_physics_specs()
model_cls = model_builder.get_model_cls(grid, dt, physics_specs_obj)

def compute_trajectory_fwd(x):
    solver = model_cls()
    x = solver.encode(x)
    final, trajectory = solver.trajectory(
        x, length, inner_steps, start_with_input=True,
        post_process_fn=solver.decode)
    return trajectory

model       = hk.without_apply_rng(hk.transform(compute_trajectory_fwd))
trajectory_fn = jax.vmap(functools.partial(model.apply, params))

# ===== 4. 推理 =====
print('\n[4/5] Running inference...')
prediction    = trajectory_fn(initial_conditions)
prediction_ds = cfd_data.xarray_utils.velocity_trajectory_to_xarray(
    prediction, grid, samples=True)
prediction_ds.coords['x']    = target_ds.coords['x']
prediction_ds.coords['y']    = target_ds.coords['y']
prediction_ds.coords['time'] = target_ds.coords['time']
print(f'  prediction dims: {dict(prediction_ds.dims)}')

# ===== 5. 评估 & 可视化 =====
print('\n[5/5] Computing metrics and plotting...')

# ---- 加载 baseline（与 target 时间轴一致才有效；不存在则跳过）----
datasets = {}
for res in [64, 128, 256, 512, 1024, 2048]:
    nc_path = os.path.join(BASELINES_DIR, f'long_eval_{res}x{res}_64x64.nc')
    if os.path.exists(nc_path):
        try:
            bl_ds = xarray.open_dataset(nc_path).isel(
                sample=slice(sample_id, sample_id + 1),
                time=slice(time_id, time_id + length),
            )
            # 时间轴必须匹配 target（step 数相同），否则跳过
            if bl_ds.dims['time'] >= length:
                bl_ds = bl_ds.assign_coords(
                    time=target_ds.coords['time'])
                datasets[f'baseline_{res}x{res}'] = bl_ds
        except Exception as e:
            print(f'  skip baseline {res}x{res}: {e}')

datasets[MODEL_LABEL] = prediction_ds
print(f'  Datasets: {list(datasets.keys())}')

# ---- 计算 summary（vorticity_correlation + energy_spectrum）----
summary = xarray.concat([
    cfd_data.evaluation.compute_summary_dataset(ds, target_ds)
    for ds in datasets.values()
], dim='model')
summary.coords['model'] = list(datasets.keys())

correlation = summary.vorticity_correlation.compute()
spectrum    = summary.energy_spectrum_mean.mean('time').compute()

# ---- 计算 L2 error（逐时间步的涡度 RMS 误差，归一化）----
print('  Computing L2 vorticity error...')
vorticities = xarray.concat(
    [cfd_data.xarray_utils.vorticity_2d(ds) for ds in datasets.values()],
    dim='model',
).to_dataset()
vorticities.coords['model'] = list(datasets.keys())

l2_errors = {}
# vorticity_2d returns a DataArray directly (not Dataset)
ref_vort = cfd_data.xarray_utils.vorticity_2d(target_ds)
for mname, ds in datasets.items():
    pred_vort = cfd_data.xarray_utils.vorticity_2d(ds)
    diff = pred_vort - ref_vort
    rms_err = np.sqrt((diff ** 2).mean(dim=['x', 'y', 'sample']).values)
    rms_ref = np.sqrt((ref_vort ** 2).mean(dim=['x', 'y', 'sample']).values)
    l2_errors[mname] = rms_err / (rms_ref + 1e-12)

# ---- 调色板（与 notebook 一致）----
n_baselines = len(datasets) - 1
palette_bl  = seaborn.color_palette('YlGnBu', n_colors=max(n_baselines + 1, 2))[1:]
model_color = seaborn.xkcd_palette(['burnt orange'])
palette     = palette_bl[:n_baselines] + model_color

time_vals = correlation.coords['time'].values
t_max_corr = min(float(time_vals[-1]), 15.0)   # 最多展示 15 时间单位（notebook 默认）

# ============================================================
# 图 1：Correlation（左）+ Energy Spectrum × k^5（右）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# -- 左：Vorticity Correlation --
ax = axes[0]
for color, mname in zip(palette, summary['model'].data):
    style = '-'  if 'baseline' in mname else '--'
    lw    = 2    if 'baseline' in mname else 3
    correlation.sel(model=mname).plot.line(
        ax=ax, color=color, linestyle=style, label=mname, linewidth=lw)
ax.axhline(y=0.95, color='gray', linestyle=':', linewidth=1)
ax.set_xlim(0, t_max_corr)
ax.set_ylim(-0.05, 1.05)
ax.set_title(f'Vorticity Correlation  ({MODEL_LABEL})')
ax.set_xlabel('Time')
ax.set_ylabel('Correlation')
ax.legend(fontsize=8)

# -- 右：Energy Spectrum × k^5（与 notebook 完全一致）--
ax = axes[1]
for color, mname in zip(palette, summary['model'].data):
    style = '-'  if 'baseline' in mname else '--'
    lw    = 2    if 'baseline' in mname else 3
    (spectrum.k ** 5 * spectrum).sel(model=mname).plot.line(
        ax=ax, color=color, linestyle=style, label=mname, linewidth=lw)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(3.5, None)
ax.set_ylim(1e9, None)
ax.set_title('Energy Spectrum × k⁵')
ax.set_xlabel('k')
ax.legend(fontsize=8)

plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, 'notebook_eval_corr_spectrum.png')
plt.savefig(out1, dpi=120); plt.close()
print(f'  Saved: {out1}')

# ============================================================
# 图 2：L2 Vorticity Error over time
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for color, mname in zip(palette, datasets.keys()):
    style = '-'  if 'baseline' in mname else '--'
    lw    = 2    if 'baseline' in mname else 3
    ax.plot(time_vals, l2_errors[mname], color=color,
            linestyle=style, linewidth=lw, label=mname)
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='random (norm=1)')
ax.set_xlim(0, t_max_corr)
ax.set_ylim(0, None)
ax.set_xlabel('Time')
ax.set_ylabel('Normalized L2 Vorticity Error')
ax.set_title(f'L2 Error  ({MODEL_LABEL})')
ax.legend(fontsize=8)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'notebook_eval_l2_error.png')
plt.savefig(out2, dpi=120); plt.close()
print(f'  Saved: {out2}')

# ============================================================
# 图 3：涡度场快照（xarray facet，与 notebook 一致）
# ============================================================
last_step = min(length - 1, 199)
num_show  = 5
time_slice = slice(None, last_step + 1, max(1, last_step // num_show))

plot_ds = vorticities.isel(time=time_slice, sample=0)
g = plot_ds['vorticity'].plot.imshow(
    row='model', col='time',
    cmap=seaborn.cm.icefire, robust=True,
    size=2.5, aspect=1,
)
out3 = os.path.join(OUTPUT_DIR, 'notebook_eval_vorticity.png')
plt.savefig(out3, dpi=100, bbox_inches='tight'); plt.close()
print(f'  Saved: {out3}')

# ============================================================
# 数字摘要
# ============================================================
print('\n=== Summary ===')
for mname in summary['model'].data:
    c = correlation.sel(model=mname).values
    idx_below = np.where(c < 0.95)[0]
    t95 = time_vals[idx_below[0]] if len(idx_below) > 0 else time_vals[-1]
    l2_t0 = l2_errors[mname][0]
    print(f'  {mname:35s}  corr@t=0: {c[0]:.4f}  t_0.95: {t95:.2f}  L2@t=0: {l2_t0:.4f}')

print('\nDone.')
