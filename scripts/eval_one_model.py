"""
单模型评估脚本，输出三个指标到 CSV：
  - final_train_loss: 训练结束时的 loss
  - t_095:            vorticity correlation 降到 0.95 以下的时间
  - mean_l2:          评估期间的平均归一化 L2 涡度误差

用法:
  python scripts/eval_one_model.py \
      --model_dir models_phase/re1000_ntraj8_seed0 \
      --eval_nc   content/kolmogorov_re1000/long_eval_2048x2048_64x64.nc \
      --train_nc  content/kolmogorov_re1000/train_2048x2048_64x64.nc \
      --re 1000 --ntraj 8 --seed 0 \
      --output_csv results/phase_metrics.csv
"""
import argparse, csv, functools, os, sys, warnings
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',  required=True)
parser.add_argument('--eval_nc',    required=True)
parser.add_argument('--train_nc',   required=True)
parser.add_argument('--re',         type=int, required=True)
parser.add_argument('--ntraj',      type=int, required=True)
parser.add_argument('--seed',       type=int, required=True)
parser.add_argument('--output_csv', required=True)
parser.add_argument('--length',     type=int, default=200)
parser.add_argument('--inner_steps',type=int, default=10)
parser.add_argument('--data_seed',  type=int, default=0, help='data generation seed')
# 二选一：ntraj 实验用 --ntraj，tlen 实验用 --tlen
parser.add_argument('--tlen',       type=int, default=-1,
                    help='time_length (for tlen experiment); if set, overrides ntraj in CSV')
args = parser.parse_args()
# y 轴参数名和值（兼容两种实验）
if args.tlen > 0:
    yparam_name, yparam_val = 'tlen', args.tlen
else:
    yparam_name, yparam_val = 'ntraj', args.ntraj

JAX_CFD_ROOT = '/jumbo/yaoqingyang/yuxin/JAX-CFD'
sys.path.insert(0, JAX_CFD_ROOT)
sys.path.insert(0, os.path.join(JAX_CFD_ROOT, 'models'))
sys.path.insert(0, os.path.join(JAX_CFD_ROOT, 'jax-cfd'))

import gin, jax, jax.numpy as jnp, numpy as np, haiku as hk, xarray
from flax.training import checkpoints
import jax_cfd.base as cfd
import jax_cfd.data as cfd_data
import jax_cfd.ml as ml
import my_model_builder, my_equations, my_interpolations, my_towers, my_advections

model_builder        = ml.model_builder
physics_specifications = ml.physics_specifications

print(f'[Re={args.re} {yparam_name}={yparam_val} seed={args.seed}] evaluating...')

# ---- 1. 读取训练 loss ----
final_train_loss = float('nan')
# 先尝试 train_loss.csv
csv_path = os.path.join(args.model_dir, 'train_loss.csv')
if os.path.exists(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if rows:
        last = rows[-1]
        final_train_loss = float(last.get('train_loss', last.get('loss', float('nan'))))
else:
    # 从 log 文件读（与 model_dir 同级 logs/train_phase/ 里）
    log_path = os.path.join(
        JAX_CFD_ROOT, 'logs', 'train_phase',
        f're{args.re}_ntraj{args.ntraj}_seed{args.seed}.log')
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if 'train_loss' in line:
                    try:
                        import re as _re
                        m = _re.search(r"'train_loss':\s*([\d.e+\-]+)", line)
                        if m:
                            final_train_loss = float(m.group(1))
                    except:
                        pass

print(f'  final_train_loss = {final_train_loss:.6e}')

# ---- 1b. 若无 loss 记录，稍后从训练数据前向推断 ----
_need_compute_train_loss = np.isnan(final_train_loss)

# ---- 2. 加载数据和 checkpoint ----
reference_ds = xarray.open_dataset(args.eval_nc)
grid = cfd_data.xarray_utils.grid_from_attrs(reference_ds.attrs)

initial_conditions = tuple(
    reference_ds[v].isel(sample=slice(0, 1), time=slice(0, 1)).values
    for v in cfd_data.xarray_utils.XR_VELOCITY_NAMES[:grid.ndim]
)
target_ds = reference_ds.isel(sample=slice(0, 1), time=slice(0, args.length))

ckpt_state = checkpoints.restore_checkpoint(args.model_dir, target=None)
params = ckpt_state['params']

# ---- 3. 构建模型 ----
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
train_ds = xarray.open_dataset(args.train_nc)
gin.parse_config(strip_imports(train_ds.attrs['physics_config_str']))
dt = float(train_ds.attrs['stable_time_step'])

physics_specs_obj = physics_specifications.get_physics_specs()
model_cls = model_builder.get_model_cls(grid, dt, physics_specs_obj)

def compute_trajectory_fwd(x):
    solver = model_cls()
    x = solver.encode(x)
    _, trajectory = solver.trajectory(
        x, args.length, args.inner_steps,
        start_with_input=True, post_process_fn=solver.decode)
    return trajectory

model = hk.without_apply_rng(hk.transform(compute_trajectory_fwd))
trajectory_fn = jax.vmap(functools.partial(model.apply, params))

# ---- 4. 推理 ----
prediction = trajectory_fn(initial_conditions)
prediction_ds = cfd_data.xarray_utils.velocity_trajectory_to_xarray(
    prediction, grid, samples=True)
prediction_ds.coords['x']    = target_ds.coords['x']
prediction_ds.coords['y']    = target_ds.coords['y']
prediction_ds.coords['time'] = target_ds.coords['time']

# ---- 5. 计算指标 ----
# Vorticity correlation
summary = cfd_data.evaluation.compute_summary_dataset(prediction_ds, target_ds)
corr = summary.vorticity_correlation.compute().values  # shape (time,)
time_vals = target_ds.coords['time'].values

# t_0.95: 第一次低于 0.95 的时间
below = np.where(corr < 0.95)[0]
t_095 = float(time_vals[below[0]]) if len(below) > 0 else float(time_vals[-1])

# Mean L2 vorticity error（用 numpy 避免 xarray 坐标对齐产生 nan；截断发散步后取均值）
ref_vort  = cfd_data.xarray_utils.vorticity_2d(target_ds).values     # (sample,time,x,y)
pred_vort = cfd_data.xarray_utils.vorticity_2d(prediction_ds).values  # (sample,time,x,y)
diff    = pred_vort - ref_vort
rms_err = np.sqrt(np.nanmean(diff ** 2, axis=(0, 2, 3)))   # shape (time,)
rms_ref = np.sqrt(np.nanmean(ref_vort ** 2, axis=(0, 2, 3)))
l2      = rms_err / (rms_ref + 1e-12)
# 截断到 10（>10 意味着轨迹已发散；截断后均值仍能区分好坏模型）
l2_clipped = np.clip(l2, 0.0, 10.0)
mean_l2 = float(np.nanmean(l2_clipped))

# L2 at specific time indices (t=10, 30, 50 steps in trajectory)
def _l2_at(idx):
    if idx < len(l2_clipped):
        return float(l2_clipped[idx])
    return float('nan')
l2_t10 = _l2_at(10)
l2_t30 = _l2_at(30)
l2_t50 = _l2_at(50)

# Mean correlation over time
mean_corr = float(np.nanmean(corr))

# Correlation at specific time indices
def _corr_at(idx):
    if idx < len(corr):
        return float(corr[idx])
    return float('nan')
corr_t10  = _corr_at(10)
corr_t30  = _corr_at(30)
corr_t50  = _corr_at(50)
corr_t100 = _corr_at(100)
l2_t100   = _l2_at(100)

print(f'  t_0.95    = {t_095:.4f}')
print(f'  mean_l2   = {mean_l2:.6f}')
print(f'  l2@t10    = {l2_t10:.6f}')
print(f'  l2@t30    = {l2_t30:.6f}')
print(f'  l2@t50    = {l2_t50:.6f}')
print(f'  l2@t100   = {l2_t100:.6f}')
print(f'  mean_corr = {mean_corr:.6f}')
print(f'  corr@t10  = {corr_t10:.6f}')
print(f'  corr@t30  = {corr_t30:.6f}')
print(f'  corr@t50  = {corr_t50:.6f}')
print(f'  corr@t100 = {corr_t100:.6f}')

# ---- 5b. train_loss 缺失时保持 NaN，在 heatmap 中显示 N/A ----
# (autoregressive proxy 与实际 training loss 量级差 4 个数量级，不适合混入同一 heatmap)

# ---- 6. 写入 CSV ----
os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
write_header = not os.path.exists(args.output_csv)
with open(args.output_csv, 'a', newline='') as f:
    writer = csv.writer(f)
    data_seed = getattr(args, 'data_seed', 0)
    if write_header:
        writer.writerow(['re', yparam_name, 'seed', 'data_seed', 'final_train_loss',
                         't_095', 'mean_l2', 'mean_corr',
                         'l2_t10', 'l2_t30', 'l2_t50', 'l2_t100',
                         'corr_t10', 'corr_t30', 'corr_t50', 'corr_t100'])
    writer.writerow([args.re, yparam_val, args.seed, data_seed,
                     f'{final_train_loss:.6e}', f'{t_095:.6f}', f'{mean_l2:.6f}', f'{mean_corr:.6f}',
                     f'{l2_t10:.6f}', f'{l2_t30:.6f}', f'{l2_t50:.6f}', f'{l2_t100:.6f}',
                     f'{corr_t10:.6f}', f'{corr_t30:.6f}', f'{corr_t50:.6f}', f'{corr_t100:.6f}'])

print(f'  → written to {args.output_csv}')
