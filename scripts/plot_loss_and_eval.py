#!/usr/bin/env python3
"""
1) Plot training loss curve from output_dir/train_loss.csv -> output_dir/train_loss_curve.png
2) On eval data: load checkpoint, run model rollout, compute baseline (ground truth) vs model
   comparison, save correlation + spectrum + field comparison figures to output_dir.
Usage:
  python scripts/plot_loss_and_eval.py --output_dir=/path/to/model --eval_nc=/path/to/eval_*.nc \\
    [--gin_file=...] [--delta_time=...] [--length=200] [--sample_id=0] [--time_id=0]
"""
import argparse
import os
import sys
import warnings
warnings.simplefilter('ignore')

import functools
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray
import gin
import jax
import jax.numpy as jnp
import haiku as hk

# Add project root and models dir so gin can resolve my_model_builder etc.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'models'))
import jax_cfd.base as cfd
import jax_cfd.data as cfd_data
import jax_cfd.ml as ml
from jax_cfd.data import xarray_utils as xr_utils
from jax_cfd.data import evaluation as cfd_eval

# Import so gin can resolve my_* modules
import models  # noqa: F401
from my_model_builder import my_trajectory_from_step  # noqa: F401


def strip_imports(s):
    out = []
    for line in s.splitlines():
        if not line.strip().startswith('import'):
            out.append(line)
    return '\n'.join(out)


def plot_loss_curve(output_dir):
    csv_path = os.path.join(output_dir, 'train_loss.csv')
    if not os.path.exists(csv_path):
        print('No train_loss.csv found, skipping loss curve.')
        return
    df = pd.read_csv(csv_path)
    if df.empty or 'train_loss' not in df.columns:
        print('train_loss.csv empty or missing train_loss column, skipping.')
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df['step'].values, df['train_loss'].values, color='#1f77b4', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Train loss')
    ax.set_title('Training loss')
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(output_dir, 'train_loss_curve.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print('Saved', out_path)


def load_checkpoint(output_dir):
    """Load params from Flax checkpoint dir or LI_ckpt.pkl."""
    try:
        from flax.training import checkpoints
        latest = checkpoints.latest_checkpoint(output_dir)
        if latest is not None:
            state = checkpoints.restore_checkpoint(output_dir, target=None)
            if hasattr(state, 'params'):
                return state.params
            if isinstance(state, dict):
                return state.get('params', state.get(1))
            return state[1]
    except Exception:
        pass
    pkl_path = os.path.join(output_dir, 'LI_ckpt.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError('No checkpoint in %s and no %s' % (output_dir, pkl_path))
    with open(pkl_path, 'rb') as f:
        ckpt = pickle.load(f)
    if hasattr(ckpt, 'params'):
        return ckpt.params
    if isinstance(ckpt, (tuple, list)) and len(ckpt) >= 2:
        return ckpt[1]
    if isinstance(ckpt, dict):
        return ckpt.get('params', ckpt.get(1))
    return ckpt


def main():
    parser = argparse.ArgumentParser(description='Plot loss curve and eval baseline vs model')
    parser.add_argument('--output_dir', type=str, required=True, help='Model/output directory')
    parser.add_argument('--eval_nc', type=str, required=True, help='Eval NetCDF path')
    parser.add_argument('--gin_file', type=str, action='append', default=None,
                        help='Gin config (repeat for multiple)')
    parser.add_argument('--gin_param', type=str, action='append', default=None)
    parser.add_argument('--delta_time', type=float, default=None, help='Model dt (default from data)')
    parser.add_argument('--length', type=int, default=200, help='Trajectory length (steps)')
    parser.add_argument('--sample_id', type=int, default=0)
    parser.add_argument('--time_id', type=int, default=0)
    parser.add_argument('--inner_steps', type=int, default=None,
                        help='Inner steps per saved step (default from data)')
    parser.add_argument('--encode_steps', type=int, default=16,
                        help='Encode steps (must match training)')
    parser.add_argument('--skip_eval', action='store_true', help='Only plot loss')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1) Loss curve
    plot_loss_curve(output_dir)
    if args.skip_eval:
        return

    # 2) Eval: load data and grid
    eval_ds = xarray.open_dataset(args.eval_nc)
    grid = cfd_data.xarray_utils.grid_from_attrs(eval_ds.attrs)
    stable_time_step = float(eval_ds.attrs.get('stable_time_step', 0.007))
    dt = args.delta_time if args.delta_time is not None else stable_time_step
    inner_steps = args.inner_steps
    if inner_steps is None:
        inner_steps = max(1, round(stable_time_step / dt))

    # Gin config (must match training)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gin_files = args.gin_file or [
        os.path.join(base, 'models/configs/official_li_config.gin'),
        os.path.join(base, 'models/configs/kolmogorov_forcing.gin'),
    ]
    for p in gin_files:
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(base, p))
        gin.parse_config_file(p)
    for binding in args.gin_param or []:
        gin.parse_config(binding)
    params = load_checkpoint(output_dir)

    # Physics from data
    if 'physics_config_str' in eval_ds.attrs:
        gin.parse_config(strip_imports(eval_ds.attrs['physics_config_str']))

    physics_specs = ml.physics_specifications.get_physics_specs()
    model_cls = ml.model_builder.get_model_cls(grid, dt, physics_specs)

    # Initial condition: first encode_steps frames; target: same sample, full trajectory
    sample_id, time_id, length = args.sample_id, args.time_id, args.length
    encode_steps = args.encode_steps
    initial_conditions = tuple(
        eval_ds[vel].isel(
            sample=slice(sample_id, sample_id + 1),
            time=slice(time_id, time_id + encode_steps)
        ).values
        for vel in xr_utils.XR_VELOCITY_NAMES[:grid.ndim]
    )
    target_ds = eval_ds.isel(
        sample=slice(sample_id, sample_id + 1),
        time=slice(time_id, time_id + length)
    )

    # Model forward
    def compute_trajectory_fwd(x):
        solver = model_cls()
        x = solver.encode(x)
        _, trajectory = solver.trajectory(
            x, length, inner_steps, start_with_input=True, post_process_fn=solver.decode)
        return trajectory

    model = hk.without_apply_rng(hk.transform(compute_trajectory_fwd))
    trajectory_fn = jax.jit(model.apply)
    prediction = trajectory_fn(params, initial_conditions)
    if prediction[0].ndim == 3:
        prediction = tuple(p[np.newaxis, ...] for p in prediction)
    prediction_ds = cfd_data.xarray_utils.velocity_trajectory_to_xarray(
        prediction, grid, time=target_ds.time.values, samples=True)
    prediction_ds.coords['x'] = target_ds.coords['x']
    prediction_ds.coords['y'] = target_ds.coords['y']
    prediction_ds.coords['time'] = target_ds.coords['time']

    # Summary: model vs ground truth (correlation = model–GT correlation over time)
    summary = cfd_eval.compute_summary_dataset(prediction_ds, target_ds)

    # Plots
    # (a) Vorticity correlation over time (model vs GT, one curve)
    if 'vorticity_correlation' in summary:
        corr = summary.vorticity_correlation.compute()
        fig, ax = plt.subplots(figsize=(7, 5))
        t = corr.time.values if 'time' in corr.dims else np.arange(len(corr))
        ax.plot(t, np.asarray(corr.values).ravel(), '-', label='Model vs GT', color='C1')
        ax.axhline(y=0.95, color='gray', linestyle=':')
        ax.set_xlabel('Time')
        ax.set_ylabel('Vorticity correlation')
        ax.set_title('Model vs baseline (ground truth)')
        ax.legend()
        ax.set_xlim(0, min(15, np.max(t)) if len(t) else 15)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'eval_vorticity_correlation.png'), dpi=150)
        plt.close()
        print('Saved eval_vorticity_correlation.png')

    # (b) Energy spectrum: combined dataset for model vs baseline
    combined = xarray.concat([prediction_ds, target_ds], dim='model')
    combined.coords['model'] = ['eval_model', 'ground_truth']
    try:
        spec_combined = xr_utils.isotropic_energy_spectrum(combined, average_dims=('sample',))
        spec_mean = spec_combined.mean('time').compute()
        if 'k' in spec_mean.dims:
            fig, ax = plt.subplots(figsize=(7, 5))
            k = spec_mean.k.values
            for m in ['eval_model', 'ground_truth']:
                s = np.asarray(spec_mean.sel(model=m).values)
                label = 'Model' if m == 'eval_model' else 'Baseline (GT)'
                ax.loglog(k, np.maximum(s, 1e-20), label=label)
            ax.set_xlabel('Wavenumber k')
            ax.set_ylabel('Energy spectrum')
            ax.set_title('Energy spectrum: model vs baseline')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'eval_energy_spectrum.png'), dpi=150)
            plt.close()
            print('Saved eval_energy_spectrum.png')
    except Exception as e:
        print('Energy spectrum plot skipped:', e)

    # (c) Field comparison: a few time steps, u/v
    n_show = 5
    t_slice = slice(None, min(length, 200), max(1, min(length, 200) // n_show))
    times_show = target_ds.time.isel(time=np.arange(0, min(length, 200), max(1, min(length, 200) // n_show)))
    u_gt = target_ds.u.isel(sample=0).sel(time=times_show)
    u_pred = prediction_ds.u.isel(sample=0).sel(time=times_show)
    v_gt = target_ds.v.isel(sample=0).sel(time=times_show)
    v_pred = prediction_ds.v.isel(sample=0).sel(time=times_show)

    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))
    for i in range(n_show):
        if i < u_gt.sizes['time']:
            axes[0, i].imshow(u_gt.isel(time=i).values, cmap='RdBu_r', aspect='equal')
            axes[0, i].set_title('u GT t=%d' % i)
            axes[0, i].axis('off')
            axes[1, i].imshow(u_pred.isel(time=i).values, cmap='RdBu_r', aspect='equal')
            axes[1, i].set_title('u Model t=%d' % i)
            axes[1, i].axis('off')
    fig.suptitle('u: Ground truth (row1) vs Model (row2)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'eval_field_comparison_u.png'), dpi=150)
    plt.close()
    print('Saved eval_field_comparison_u.png')

    fig2, axes2 = plt.subplots(2, n_show, figsize=(2 * n_show, 4))
    for i in range(n_show):
        if i < v_gt.sizes['time']:
            axes2[0, i].imshow(v_gt.isel(time=i).values, cmap='RdBu_r', aspect='equal')
            axes2[0, i].set_title('v GT t=%d' % i)
            axes2[0, i].axis('off')
            axes2[1, i].imshow(v_pred.isel(time=i).values, cmap='RdBu_r', aspect='equal')
            axes2[1, i].set_title('v Model t=%d' % i)
            axes2[1, i].axis('off')
    fig2.suptitle('v: Ground truth (row1) vs Model (row2)')
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'eval_field_comparison_v.png'), dpi=150)
    plt.close()
    print('Saved eval_field_comparison_v.png')

    print('All plots saved under', output_dir)


if __name__ == '__main__':
    main()
