"""
生成 Kolmogorov flow 训练/评估数据集，支持任意 Re
格式与 train_2048x2048_64x64.nc / long_eval_2048x2048_64x64.nc 完全一致

用法示例：

# 生成 Re=1000 训练集 (32 samples, 30s, DNS 2048x2048 → 保存 64x64)
CUDA_VISIBLE_DEVICES=0 python scripts/generate_kolmogorov_data.py \
    --re 1000 \
    --output content/kolmogorov_re1000/train_2048x2048_64x64.nc \
    --num_samples 32 --dns_size 2048 --save_size 64 \
    --warmup_time 40.0 --simulation_time 30.0 --seed 0

# 生成 Re=2000 训练集
CUDA_VISIBLE_DEVICES=1 python scripts/generate_kolmogorov_data.py \
    --re 2000 \
    --output content/kolmogorov_re2000/train_2048x2048_64x64.nc \
    --num_samples 32 --dns_size 2048 --save_size 64 \
    --warmup_time 40.0 --simulation_time 30.0 --seed 0

物理参数 (可通过 --re 控制):
  viscosity = 1/Re, density=1.0
  forcing: kolmogorov wavenumber=4, scale=1.0, linear_coeff=-0.1
"""

import argparse, os, sys, time, warnings
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--output',          default='output.nc',      help='输出 .nc 文件路径')
parser.add_argument('--num_samples',     type=int,   default=32,   help='独立轨迹数量')
parser.add_argument('--dns_size',        type=int,   default=512,  help='DNS 网格分辨率 (NxN)')
parser.add_argument('--save_size',       type=int,   default=64,   help='保存网格分辨率 (NxN)')
parser.add_argument('--warmup_time',     type=float, default=40.0, help='热身时间 (时间单位)')
parser.add_argument('--simulation_time', type=float, default=30.0, help='记录时间 (时间单位)')
parser.add_argument('--seed',            type=int,   default=0,    help='随机种子')
parser.add_argument('--max_velocity',    type=float, default=7.0,  help='初始速度最大值')
parser.add_argument('--peak_wavenumber', type=int,   default=4,    help='初始条件峰值波数')
parser.add_argument('--cfl_safety',      type=float, default=0.5,  help='CFL 安全系数')
parser.add_argument('--re',                    type=float, default=1000, help='Reynolds 数 (viscosity = 1/Re)')
parser.add_argument('--chunk_steps',          type=int,   default=200,  help='每次 JIT 执行的外部步数 (控制内存)')
parser.add_argument('--time_subsample_factor', type=int,   default=10,   help='每隔多少外步保存一帧 (原始数据=10)')
args = parser.parse_args()

# ===== 物理参数 =====
VISCOSITY = 1.0 / args.re
DENSITY   = 1.0
FORCING_WAVENUMBER   = 4
FORCING_SCALE        = 1.0
FORCING_LINEAR_COEFF = -0.1

# ===== JAX + jax_cfd 导入 =====
JAX_CFD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(JAX_CFD_ROOT, 'jax-cfd'))

import jax, jax.numpy as jnp, numpy as np
import functools
import xarray
import jax_cfd.base as cfd
import jax_cfd.ml as cfd_ml

print(f'JAX devices: {jax.devices()}')
print(f'Re={args.re:.0f}  viscosity={VISCOSITY:.6f}')
print(f'DNS: {args.dns_size}x{args.dns_size} → save: {args.save_size}x{args.save_size}')
print(f'Samples: {args.num_samples}, warmup: {args.warmup_time}s, sim: {args.simulation_time}s')
print(f'Output: {args.output}')

# ===== 计算时间步长 =====
dns_grid  = cfd.grids.Grid((args.dns_size,  args.dns_size),
                            domain=((0, 2*jnp.pi), (0, 2*jnp.pi)))
save_grid = cfd.grids.Grid((args.save_size, args.save_size),
                            domain=((0, 2*jnp.pi), (0, 2*jnp.pi)))

dt_dns  = cfd.equations.stable_time_step(
    args.max_velocity, args.cfl_safety, VISCOSITY, dns_grid, implicit_diffusion=True)
dt_save = cfd.equations.stable_time_step(
    args.max_velocity, args.cfl_safety, VISCOSITY, save_grid, implicit_diffusion=True)

inner_steps = round(dt_save / dt_dns)   # DNS steps per output frame
print(f'dt_dns={dt_dns:.7f}  dt_save={dt_save:.7f}  inner_steps={inner_steps}')

dt_frame      = dt_save * args.time_subsample_factor   # physical time per saved frame
warmup_outer  = int(args.warmup_time  / dt_save)
simulate_outer = int(args.simulation_time / dt_frame)   # number of saved frames
total_frames  = simulate_outer
print(f'dt_frame={dt_frame:.7f}  time_subsample_factor={args.time_subsample_factor}')
print(f'Warmup: {warmup_outer} outer steps ({args.warmup_time:.0f}s)')
print(f'Simulate: {simulate_outer * args.time_subsample_factor} outer steps → {total_frames} saved frames')
print(f'Estimated time: ~{warmup_outer*inner_steps:,} + {simulate_outer*args.time_subsample_factor*inner_steps:,} DNS steps')

# ===== 物理设置 =====
forcing = cfd_ml.forcings.kolmogorov_forcing(
    dns_grid,
    scale=FORCING_SCALE,
    wavenumber=FORCING_WAVENUMBER,
    linear_coefficient=FORCING_LINEAR_COEFF,
)

step_fn = cfd.funcutils.repeated(
    cfd.equations.semi_implicit_navier_stokes(
        density=DENSITY,
        viscosity=VISCOSITY,
        dt=dt_dns,
        grid=dns_grid,
        forcing=forcing,
    ),
    steps=inner_steps,
)

# JIT 编译的 chunk rollout (batched over samples via vmap)
@functools.partial(jax.jit, static_argnums=(1,))
def run_chunk(v0, n_steps):
    """v0: (2, N, N) staggered velocity → returns (u_traj, v_traj): (n_steps, N, N)"""
    step_batched = jax.vmap(step_fn)
    _, traj = cfd.funcutils.trajectory(step_fn, n_steps)(v0)
    return traj   # tuple of GridVariable

# vmap 版本 (over samples)
def run_chunk_batched(v0_batch, n_steps):
    """v0_batch: tuple (u, v) each shape (S, N, N) staggered"""
    fn = functools.partial(cfd.funcutils.trajectory(step_fn, n_steps), )
    # 手动 vmap over sample dim
    traj_fn = jax.vmap(lambda v: cfd.funcutils.trajectory(step_fn, n_steps)(v)[1])
    traj_fn = jax.jit(traj_fn)
    return traj_fn(v0_batch)

# ===== 下采样工具 =====
_bc = cfd.boundaries.periodic_boundary_conditions(dns_grid.ndim)

def downsample_velocity(u, v):
    """从 dns_grid 下采样到 save_grid，使用正确的 staggered 偏移量。"""
    uv_grid_var = tuple(
        cfd.grids.GridVariable(cfd.grids.GridArray(arr, offset, dns_grid), _bc)
        for arr, offset in zip([u, v], dns_grid.cell_faces)
    )
    ds = cfd.resize.downsample_staggered_velocity(dns_grid, save_grid, uv_grid_var)
    return np.array(ds[0].data), np.array(ds[1].data)

# ===== 初始化 =====
rng = jax.random.PRNGKey(args.seed)
print(f'\n[1/3] Initializing {args.num_samples} random initial conditions...')

all_u0, all_v0 = [], []
for i in range(args.num_samples):
    rng, subrng = jax.random.split(rng)
    v0 = cfd.initial_conditions.filtered_velocity_field(
        subrng, dns_grid, args.max_velocity, args.peak_wavenumber)
    all_u0.append(np.array(v0[0].data))
    all_v0.append(np.array(v0[1].data))

all_u0 = np.stack(all_u0)   # (S, N, N)
all_v0 = np.stack(all_v0)

def make_gv(u_arr, v_arr):
    """Build GridVariable tuple from numpy arrays matching dns_grid cell_faces offsets."""
    bc = cfd.boundaries.periodic_boundary_conditions(dns_grid.ndim)
    uv = []
    for arr, offset in zip([u_arr, v_arr], dns_grid.cell_faces):
        ga = cfd.grids.GridArray(jnp.array(arr), offset, dns_grid)
        uv.append(cfd.grids.GridVariable(ga, bc))
    return tuple(uv)

# ===== Warmup (不保存) =====
print(f'\n[2/3] Warmup ({args.warmup_time:.0f} time units, chunk={args.chunk_steps} steps)...')
t0 = time.time()

u_cur = all_u0.copy()
v_cur = all_v0.copy()

warmup_done = 0
while warmup_done < warmup_outer:
    chunk = min(args.chunk_steps, warmup_outer - warmup_done)
    traj_list = []
    for s in range(args.num_samples):
        v0_s = make_gv(u_cur[s:s+1][0], v_cur[s:s+1][0])
        _, traj = jax.device_get(jax.jit(cfd.funcutils.trajectory(step_fn, chunk))(v0_s))
        u_cur[s] = np.array(traj[0].data[-1])
        v_cur[s] = np.array(traj[1].data[-1])
    warmup_done += chunk
    elapsed = time.time() - t0
    pct = warmup_done / warmup_outer * 100
    eta = elapsed / max(warmup_done, 1) * (warmup_outer - warmup_done)
    print(f'  warmup {warmup_done}/{warmup_outer} ({pct:.0f}%)  elapsed={elapsed:.0f}s  ETA={eta:.0f}s')

print(f'Warmup done in {time.time()-t0:.0f}s')

# ===== Simulation (保存轨迹) =====
print(f'\n[3/3] Recording {args.simulation_time:.0f} time units...')
t1 = time.time()

# 预分配输出数组
u_out = np.zeros((args.num_samples, total_frames, args.save_size, args.save_size), dtype=np.float32)
v_out = np.zeros((args.num_samples, total_frames, args.save_size, args.save_size), dtype=np.float32)

tsf = args.time_subsample_factor   # outer steps per saved frame
frames_done = 0
while frames_done < total_frames:
    chunk = min(args.chunk_steps, total_frames - frames_done)
    outer_chunk = chunk * tsf   # total outer steps to run for this chunk of frames
    for s in range(args.num_samples):
        v0_s = make_gv(u_cur[s], v_cur[s])
        _, traj = jax.device_get(jax.jit(cfd.funcutils.trajectory(step_fn, outer_chunk))(v0_s))
        # 每 tsf 个外步取一帧下采样保存
        for fi in range(chunk):
            ti = (fi + 1) * tsf - 1   # last outer step of each group
            ud, vd = downsample_velocity(
                np.array(traj[0].data[ti]), np.array(traj[1].data[ti]))
            u_out[s, frames_done + fi] = ud
            v_out[s, frames_done + fi] = vd
        u_cur[s] = np.array(traj[0].data[-1])
        v_cur[s] = np.array(traj[1].data[-1])

    frames_done += chunk
    elapsed = time.time() - t1
    pct = frames_done / total_frames * 100
    eta = elapsed / max(frames_done, 1) * (total_frames - frames_done)
    print(f'  sim {frames_done}/{total_frames} ({pct:.0f}%)  elapsed={elapsed:.0f}s  ETA={eta:.0f}s')

print(f'Simulation done in {time.time()-t1:.0f}s')

# ===== 保存 =====
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

time_coord = dt_frame * np.arange(total_frames)

x_vals = np.array(save_grid.axes()[0]).mean() * 2 / args.save_size * np.arange(args.save_size)
y_vals = np.array(save_grid.axes()[1]).mean() * 2 / args.save_size * np.arange(args.save_size)

# 构建 physics_config_str（与原始数据格式一致）
physics_config_str = f"""
# Macros:
# ==============================================================================
DENSITY = {DENSITY}
RE = {args.re:.0f}
FORCING_MODULE = @forcings.kolmogorov_forcing

# Parameters for get_physics_specs:
# ==============================================================================
get_physics_specs.physics_specs_cls = @NavierStokesPhysicsSpecs

# Parameters for kolmogorov_forcing:
# ==============================================================================
kolmogorov_forcing.linear_coefficient = {FORCING_LINEAR_COEFF}
kolmogorov_forcing.scale = {FORCING_SCALE}
kolmogorov_forcing.swap_xy = False
kolmogorov_forcing.wavenumber = {FORCING_WAVENUMBER}

# Parameters for NavierStokesPhysicsSpecs:
# ==============================================================================
NavierStokesPhysicsSpecs.density = %DENSITY
NavierStokesPhysicsSpecs.forcing_module = %FORCING_MODULE
NavierStokesPhysicsSpecs.viscosity = {VISCOSITY}
"""

ds = xarray.Dataset(
    {
        'u': (['sample', 'time', 'x', 'y'], u_out),
        'v': (['sample', 'time', 'x', 'y'], v_out),
    },
    coords={
        'sample': np.arange(args.num_samples),
        'time':   time_coord,
        'x':      x_vals,
        'y':      y_vals,
    },
    attrs={
        'seed':                       args.seed,
        'ndim':                       2,
        'reynolds_number':            args.re,
        'domain_size_multiple':       1,
        'warmup_grid_size':           args.dns_size,
        'simulation_grid_size':       args.dns_size,
        'save_grid_size':             args.save_size,
        'warmup_time':                args.warmup_time,
        'simulation_time':            args.simulation_time,
        'time_subsample_factor':      args.time_subsample_factor,
        'maximum_velocity':           args.max_velocity,
        'init_peak_wavenumber':       float(args.peak_wavenumber),
        'init_cfl_safety_factor':     args.cfl_safety,
        'stable_time_step':           float(dt_save),
        'physics_config_str':         physics_config_str,
        'tracing_max_duration_in_msec': 100.0,
    }
)

print(f'\nSaving to {args.output} ...')
# 使用 NETCDF3_64BIT 格式（与原始数据一致，避免 HPC Lustre 文件系统上 HDF5 锁问题）
ds.to_netcdf(args.output, format='NETCDF3_64BIT')
print(f'Done. Dataset: {dict(ds.dims)}')
print(f'  u shape: {u_out.shape},  time span: 0 ~ {time_coord[-1]:.2f}')
print(f'\nTotal time: {time.time()-t0:.0f}s')
