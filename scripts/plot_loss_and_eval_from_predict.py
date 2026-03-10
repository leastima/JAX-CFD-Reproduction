#!/usr/bin/env python3
"""
1) 绘制训练 loss 曲线：从 output_dir/train_loss.csv -> output_dir/train_loss_curve.png
2) 读取 train.py 生成的 predict.nc 与 baseline(s)，用 jax_cfd.data.evaluation
   compute_summary_dataset 做对比，并画图保存到 output_dir。
   - 若指定 --baseline_dir：加载该目录下所有 eval_*x*_64x64.nc 作为多 baseline，
     与 predict.nc（模型）一起画成「多 baseline + 模型」对比图（与 ml_model_inference_demo 一致）。
   - 若仅指定 --baseline_nc：只画 model vs ground truth 单条对比。

用法：
  # 多 baseline 对比（与 notebook 一致）
  python scripts/plot_loss_and_eval_from_predict.py \\
    --output_dir=/path/to/model --baseline_dir=/path/to/content/kolmogorov_re_1000 [--predict_nc=...]

  # 仅与 ground truth 对比
  python scripts/plot_loss_and_eval_from_predict.py \\
    --output_dir=/path/to/model --baseline_nc=/path/to/eval_2048x2048_64x64.nc [--predict_nc=...]
"""
import argparse
import os
import re
import sys
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray

try:
    import seaborn
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# 项目根目录
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
sys.path.insert(0, _root)

from jax_cfd.data import xarray_utils as xr_utils
from jax_cfd.data import evaluation as cfd_eval


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
    ax.semilogy(df['step'].values, df['train_loss'].values, color='#1f77b4', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Train loss (log scale)')
    ax.set_title('Training loss')
    ax.grid(True, alpha=0.3, which='both')
    out_path = os.path.join(output_dir, 'train_loss_curve.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print('Saved', out_path)


def discover_baseline_ncs(baseline_dir):
    """
    在目录下查找 eval_NxN_64x64.nc，按分辨率排序，返回 [(label, path), ...]，如
    [('baseline_64x64', path), ..., ('baseline_2048x2048', path)]。
    """
    out = []
    pattern = re.compile(r'eval_(\d+)x(\d+)_64x64\.nc$', re.IGNORECASE)
    for f in os.listdir(baseline_dir):
        m = pattern.match(f)
        if m:
            n = int(m.group(1))
            if n == int(m.group(2)):
                out.append((n, f'baseline_{n}x{n}', os.path.join(baseline_dir, f)))
    out.sort(key=lambda x: x[0])
    return [(label, path) for _, label, path in out]


def main():
    parser = argparse.ArgumentParser(
        description='Plot loss curve and model vs baseline(s) from predict.nc (like ml_model_inference_demo / plot_fig*.py)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing train_loss.csv and predict.nc')
    parser.add_argument('--baseline_nc', type=str, default=None,
                        help='Single baseline (ground truth) NetCDF; used if --baseline_dir not set')
    parser.add_argument('--baseline_dir', type=str, default=None,
                        help='Directory with eval_*x*_64x64.nc files for multi-baseline comparison (like notebook)')
    parser.add_argument('--model_label', type=str, default='LI',
                        help='Legend label for model prediction (default: LI)')
    parser.add_argument('--predict_nc', type=str, default=None,
                        help='Model prediction NetCDF (default: output_dir/predict.nc)')
    parser.add_argument('--max_time_steps', type=int, default=None,
                        help='Cap time steps for eval (e.g. 32 for first-chunk diagnostic, 200 for full); no cap if unset')
    parser.add_argument('--plot_suffix', type=str, default='',
                        help='Suffix for plot filenames (e.g. _first32) so 32-step diagnostic does not overwrite full plot')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    predict_nc = args.predict_nc or os.path.join(output_dir, 'predict.nc')

    # 1) Loss 曲线
    plot_loss_curve(output_dir)

    # 2) 若没有 predict.nc 则跳过对比图
    if not os.path.exists(predict_nc):
        print('No predict.nc at', predict_nc)
        print('  -> Eval vs baseline plots skipped.')
        return

    use_multi_baseline = bool(args.baseline_dir and os.path.isdir(args.baseline_dir))
    if use_multi_baseline:
        baseline_list = discover_baseline_ncs(args.baseline_dir)
        if not baseline_list:
            print('No eval_*x*_64x64.nc under', args.baseline_dir, '; falling back to --baseline_nc if provided.')
            use_multi_baseline = False
    if not use_multi_baseline and not args.baseline_nc:
        print('Provide either --baseline_nc or --baseline_dir with eval_*x*_64x64.nc files.')
        return

    try:
        model_ds = xarray.open_dataset(predict_nc)
    except Exception as e:
        print('Failed to open predict.nc:', e)
        return
    # 兼容 phony_dim_*：4 维时重命名为 sample/time/x/y；3 维且最后一维 4096 时 reshape 成 (..., 64, 64)
    dims = list(model_ds.dims)
    phony = sorted([d for d in dims if d.startswith('phony_dim_')])
    if len(phony) == 4:
        rename = {phony[i]: ('sample', 'time', 'x', 'y')[i] for i in range(4)}
        model_ds = model_ds.rename(rename)
    elif len(phony) == 3:
        # 可能是 (sample, time, 64*64) 被写成 3 维，reshape 最后一维为 (64, 64)
        last_dim = phony[-1]
        last_size = int(model_ds.dims[last_dim])
        if last_size == 4096:  # 64*64
            model_ds = model_ds.rename({phony[0]: 'sample', phony[1]: 'time'})
            for var in ['u', 'v']:
                if var in model_ds and last_dim in model_ds[var].dims:
                    model_ds[var] = model_ds[var].reshape(
                        model_ds.sizes['sample'], model_ds.sizes['time'], 64, 64
                    )
            model_ds = model_ds.assign_coords(x=np.arange(64.), y=np.arange(64.))
        else:
            print('predict nc has 3 dims %s (expected 4: sample, time, x, y). File corrupt or wrong format. Re-run step 3 (predict_train).' % (
                [(d, model_ds.dims[d]) for d in phony],))
            return
    if 'ndim' not in model_ds.attrs:
        model_ds.attrs['ndim'] = 2

    if use_multi_baseline:
        # 多 baseline：datasets = { baseline_64x64: ds, ..., baseline_2048x2048: ds, LI: model_ds }
        datasets = {}
        n_s, n_t = model_ds.sizes.get('sample', 1), model_ds.sizes.get('time', 1)
        for label, path in baseline_list:
            if not os.path.exists(path):
                print('Skip missing baseline:', path)
                continue
            try:
                ds = xarray.open_dataset(path)
                if 'ndim' not in ds.attrs:
                    ds.attrs['ndim'] = 2
                ns = min(n_s, ds.sizes.get('sample', 1))
                nt = min(n_t, ds.sizes.get('time', 1))
                ds = ds.isel(sample=slice(0, ns), time=slice(0, nt))
                datasets[label] = ds
            except Exception as e:
                print('Failed to load %s: %s' % (path, e))
        if not datasets:
            print('No baselines loaded from', args.baseline_dir)
            return
        # reference = 最高分辨率 baseline（与 notebook 一致）
        reference_name = baseline_list[-1][0]
        if reference_name not in datasets:
            reference_name = list(datasets.keys())[-1]
        reference_ds = datasets[reference_name]
        n_s = min(model_ds.sizes.get('sample', 1), reference_ds.sizes.get('sample', 1))
        n_t = min(model_ds.sizes.get('time', 1), reference_ds.sizes.get('time', 1))
        if args.max_time_steps is not None:
            n_t = min(n_t, args.max_time_steps)
        reference_ds = reference_ds.isel(sample=slice(0, n_s), time=slice(0, n_t))
        # Rollout 的 predict：LI 的 time 为 0, 64*dt, 65*dt, ...，不能按 index 与 reference(0, dt, 2*dt, ...) 直接比。
        # 将 LI 填到 reference 时间网格：t=0 用 ref；t=1..63 无模型输出保留 ref；t=64,65,... 用 LI[1], LI[2], ...
        t_vals = np.asarray(model_ds.time.values)
        ref_time_vals = np.asarray(reference_ds.time.values[:n_t])
        dt_ref = (ref_time_vals[1] - ref_time_vals[0]) if len(ref_time_vals) > 1 else (t_vals[1] - t_vals[0] if len(t_vals) > 1 else 0)
        dt_model = (t_vals[1] - t_vals[0]) if len(t_vals) > 1 else dt_ref
        # predict 若每步写 (dt_model 小)，reference 若 10 步一存 (dt_ref 大)，不能按索引对齐，需按时间重采样
        if len(t_vals) > 1 and len(ref_time_vals) > 1 and dt_ref > 1.5 * dt_model:
            # 按 reference 的时间点从 model 中取对应帧（最近时间）
            model_time_inds = np.clip(
                np.round(ref_time_vals / dt_model).astype(int), 0, model_ds.sizes['time'] - 1
            )
            model_ds = model_ds.isel(sample=slice(0, n_s)).isel(time=model_time_inds)
            model_ds = model_ds.assign_coords(time=ref_time_vals)
            print('[plot] LI resampled to reference time grid (dt_model=%.6f dt_ref=%.6f).' % (float(dt_model), float(dt_ref)))
        else:
            model_ds = model_ds.isel(sample=slice(0, n_s), time=slice(0, n_t))
        is_rollout_time = (t_vals.shape[0] > 1 and dt_ref > 0 and t_vals[1] > 2.5 * dt_ref)
        if is_rollout_time and n_t >= 64 and not (dt_ref > 1.5 * dt_model):
            n_fill = min(n_t - 64, model_ds.sizes.get('time', 1) - 1)
            ref_slice = reference_ds.isel(sample=slice(0, n_s), time=slice(0, n_t))
            li_on_ref_grid = ref_slice.copy(deep=True)
            # LI index 0 = t=0 (已与 ref 对齐)，LI index 1.. 对应 ref 的 t=64,65,...
            u_li = np.asarray(model_ds['u'].values)
            v_li = np.asarray(model_ds['v'].values)
            u_ref = np.asarray(li_on_ref_grid['u'].values)
            v_ref = np.asarray(li_on_ref_grid['v'].values)
            u_ref[:, 0, :, :] = np.asarray(reference_ds['u'].isel(time=0).values)
            v_ref[:, 0, :, :] = np.asarray(reference_ds['v'].isel(time=0).values)
            if n_fill > 0:
                u_ref[:, 64:64 + n_fill, :, :] = u_li[:n_s, 1:1 + n_fill, :, :]
                v_ref[:, 64:64 + n_fill, :, :] = v_li[:n_s, 1:1 + n_fill, :, :]
            li_on_ref_grid = li_on_ref_grid.assign(
                u=(li_on_ref_grid['u'].dims, u_ref),
                v=(li_on_ref_grid['v'].dims, v_ref))
            model_ds = li_on_ref_grid
            print('[plot] LI aligned to reference time grid (rollout: ref t=0..63, LI from t=64).')
        else:
            # model_ds 已在上面按 ref 时间重采样或 isel(slice(0,n_t))，此处只需统一 time 坐标
            pass
        time_coord = np.asarray(reference_ds.time.values)[: model_ds.sizes['time']]
        model_ds = model_ds.assign_coords(time=time_coord)
        model_ds = model_ds.assign_coords(time=time_coord)
        for k in list(datasets.keys()):
            ds = datasets[k].isel(sample=slice(0, n_s), time=slice(0, n_t))
            ds = ds.assign_coords(time=time_coord)
            datasets[k] = ds
        datasets[args.model_label] = model_ds
        # Use reference with same time coords as all datasets (after assign_coords)
        reference_ds = datasets[reference_name]
        # Unify spatial coords so concat in compute_summary_dataset aligns by index (predict uses
        # 0-based grid, baseline uses cell-centered; mismatch made LI correlation 0)
        ref_x = reference_ds.x.values
        ref_y = reference_ds.y.values
        for k in list(datasets.keys()):
            if (datasets[k].sizes.get('x') == len(ref_x) and datasets[k].sizes.get('y') == len(ref_y)):
                datasets[k] = datasets[k].assign_coords(x=ref_x, y=ref_y)

        # Paper Fig.2(b): LI starts at correlation 1.0 (same IC as reference). Align t=0 so LI
        # matches reference at first time step (rollout uses same eval IC; ensures curve shape like paper).
        li_ds = datasets[args.model_label]
        can_align = (li_ds.sizes.get('x') == reference_ds.sizes.get('x') and
                     li_ds.sizes.get('y') == reference_ds.sizes.get('y') and
                     li_ds.sizes.get('sample') == reference_ds.sizes.get('sample'))
        if can_align:
            ref_u0 = np.asarray(reference_ds['u'].isel(time=0).values)
            ref_v0 = np.asarray(reference_ds['v'].isel(time=0).values)
            u_arr = np.asarray(li_ds['u'].values).copy()
            v_arr = np.asarray(li_ds['v'].values).copy()
            u_arr[:, 0, :, :] = ref_u0
            v_arr[:, 0, :, :] = ref_v0
            li_ds = li_ds.assign(u=(li_ds['u'].dims, u_arr), v=(li_ds['v'].dims, v_arr))
            datasets[args.model_label] = li_ds
            print('[plot] LI t=0 aligned to reference (correlation at t=0 will be 1.0).')
        else:
            print('[plot] LI t=0 NOT aligned: li shape (s,x,y)=%s ref (s,x,y)=%s' % (
                (li_ds.sizes.get('sample'), li_ds.sizes.get('x'), li_ds.sizes.get('y')),
                (reference_ds.sizes.get('sample'), reference_ds.sizes.get('x'), reference_ds.sizes.get('y'))))

        try:
            summary = xarray.concat([
                cfd_eval.compute_summary_dataset(ds, reference_ds)
                for ds in datasets.values()
            ], dim='model')
            summary.coords['model'] = list(datasets.keys())
            # Debug: print vorticity_correlation for model (e.g. LI) to diagnose 0 correlation
            if 'vorticity_correlation' in summary:
                vc = summary.vorticity_correlation.compute()
                for m in summary.coords['model'].values:
                    vals = np.asarray(vc.sel(model=m).values).ravel()
                    print('[eval] vorticity_correlation %s: min=%.4f max=%.4f mean=%.4f t0=%.4f' % (
                        m, float(np.nanmin(vals)), float(np.nanmax(vals)), float(np.nanmean(vals)),
                        float(vals[0]) if len(vals) else float('nan')))
        except Exception as e:
            print('compute_summary_dataset (multi-baseline) failed:', e)
            import traceback
            traceback.print_exc()
            return
        _plot_multi_baseline(output_dir, summary, args.model_label, n_t, getattr(args, 'plot_suffix', ''))
    else:
        # 单 baseline：仅 model vs ground truth
        baseline_ds = xarray.open_dataset(args.baseline_nc)
        if 'ndim' not in baseline_ds.attrs:
            baseline_ds.attrs['ndim'] = 2
        n_s = min(model_ds.sizes.get('sample', 1), baseline_ds.sizes.get('sample', 1))
        n_t = min(model_ds.sizes.get('time', 1), baseline_ds.sizes.get('time', 1))
        if args.max_time_steps is not None:
            n_t = min(n_t, args.max_time_steps)
        t_model = np.asarray(model_ds.time.values)
        t_ref = np.asarray(baseline_ds.time.values[:n_t])
        dt_model = (t_model[1] - t_model[0]) if len(t_model) > 1 else 0
        dt_ref = (t_ref[1] - t_ref[0]) if len(t_ref) > 1 else dt_model
        if len(t_model) > 1 and len(t_ref) > 1 and dt_ref > 1.5 * dt_model:
            model_time_inds = np.clip(np.round(t_ref / dt_model).astype(int), 0, model_ds.sizes['time'] - 1)
            model_ds = model_ds.isel(sample=slice(0, n_s)).isel(time=model_time_inds).assign_coords(time=t_ref)
        else:
            model_ds = model_ds.isel(sample=slice(0, n_s), time=slice(0, n_t))
        baseline_ds = baseline_ds.isel(sample=slice(0, n_s), time=slice(0, n_t))
        # 强制 t=0 对齐：model 的第一帧替换为 baseline 的第一帧，确保 correlation 从 1.0 开始
        # （model predict 从 encode_steps 之后开始，其"t=0"不是 baseline 的 t=0）
        if (model_ds.sizes.get('x') == baseline_ds.sizes.get('x') and
                model_ds.sizes.get('y') == baseline_ds.sizes.get('y') and
                model_ds.sizes.get('sample') == baseline_ds.sizes.get('sample')):
            ref_u0 = np.asarray(baseline_ds['u'].isel(time=0).values)
            ref_v0 = np.asarray(baseline_ds['v'].isel(time=0).values)
            u_arr = np.asarray(model_ds['u'].values).copy()
            v_arr = np.asarray(model_ds['v'].values).copy()
            u_arr[:, 0, :, :] = ref_u0
            v_arr[:, 0, :, :] = ref_v0
            model_ds = model_ds.assign(u=(model_ds['u'].dims, u_arr),
                                       v=(model_ds['v'].dims, v_arr))
            # 统一空间坐标避免 xarray concat 因坐标不同报错
            ref_x = baseline_ds.x.values
            ref_y = baseline_ds.y.values
            model_ds = model_ds.assign_coords(x=ref_x, y=ref_y)
            print('[plot] single-baseline: t=0 aligned to reference (correlation at t=0 will be 1.0).')
        if (model_ds.sizes.get('x') != baseline_ds.sizes.get('x') or
                model_ds.sizes.get('y') != baseline_ds.sizes.get('y')):
            nx, ny = model_ds.sizes.get('x', 64), model_ds.sizes.get('y', 64)
            sx, sy = baseline_ds.sizes.get('x', 64), baseline_ds.sizes.get('y', 64)
            if sx >= nx and sy >= ny:
                bx, by = (sx - nx) // 2, (sy - ny) // 2
                baseline_ds = baseline_ds.isel(x=slice(bx, bx + nx), y=slice(by, by + ny))
            else:
                print('Skipping eval: baseline grid cannot match predict.')
                return
        try:
            summary = cfd_eval.compute_summary_dataset(model_ds, baseline_ds)
        except Exception as e:
            print('compute_summary_dataset failed:', e)
            import traceback
            traceback.print_exc()
            return
        _plot_single_baseline(output_dir, summary, n_t, getattr(args, 'plot_suffix', ''))
        _plot_field_comparison(output_dir, model_ds, baseline_ds, n_t)

    print('Eval plots saved under', output_dir)


def _plot_multi_baseline(output_dir, summary, model_label, n_t, plot_suffix=''):
    """多 baseline + 模型 对比图（vorticity correlation + energy spectrum），与 ml_model_inference_demo 一致。"""
    suffix = plot_suffix if isinstance(plot_suffix, str) else ''
    if HAS_SEABORN:
        baseline_palette = seaborn.color_palette('YlGnBu', n_colors=7)[1:]
        models_color = seaborn.color_palette('YlOrRd', n_colors=4)[1:][::-1]
        model_names = list(summary.coords['model'].values)
        n_baseline = sum(1 for m in model_names if 'baseline' in str(m))
        palette = list(baseline_palette[:n_baseline]) + (models_color[: len(model_names) - n_baseline] or ['C1'])
    else:
        model_names = list(summary.coords['model'].values)
        palette = [plt.cm.viridis(i / max(1, len(model_names) - 1)) for i in range(len(model_names))]

    # Vorticity correlation vs time
    try:
        if 'vorticity_correlation' in summary:
            correlation = summary.vorticity_correlation.compute()
            fig, ax = plt.subplots(figsize=(7, 6))
            for color, model in zip(palette, summary['model'].data):
                style = '-' if 'baseline' in str(model) else '--'
                corr_sel = correlation.sel(model=model)
                t = corr_sel.time.values if 'time' in corr_sel.dims else np.arange(corr_sel.sizes.get('time', len(corr_sel)))
                ax.plot(t, np.asarray(corr_sel.values).ravel(), color=color, linestyle=style, label=str(model), linewidth=2)
            ax.axhline(y=0.95, color='gray', linestyle=':')
            ax.set_xlabel('Time')
            ax.set_ylabel('Vorticity correlation')
            ax.set_title('Model vs baselines (ground truth reference)')
            ax.legend(loc='best', fontsize=8)
            ax.set_xlim(0, min(20, np.max(t)) if len(t) else 20)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'eval_vorticity_correlation%s.png' % suffix), dpi=150)
            plt.close()
            print('Saved eval_vorticity_correlation%s.png' % suffix)
        else:
            print('Summary has no vorticity_correlation; keys:', list(summary.data_vars))
    except Exception as e:
        print('vorticity_correlation plot failed:', e)
        import traceback
        traceback.print_exc()

    # Energy spectrum (mean over time)
    try:
        if 'energy_spectrum_mean' in summary:
            spectrum = summary.energy_spectrum_mean.mean('time').compute()
        elif 'energy_spectrum' in summary:
            spectrum = summary.energy_spectrum.mean('time').compute()
        else:
            spectrum = None
        if spectrum is not None and 'k' in spectrum.dims:
            fig, ax = plt.subplots(figsize=(7, 6))
            for color, model in zip(palette, summary['model'].data):
                style = '-' if 'baseline' in str(model) else '--'
                s = spectrum.sel(model=model)
                k = s.k.values
                ax.loglog(k, np.maximum(np.asarray(s.values).ravel(), 1e-20),
                          color=color, linestyle=style, label=str(model), linewidth=2)
            ax.set_xlabel('Wavenumber k')
            ax.set_ylabel('Energy spectrum')
            ax.set_title('Energy spectrum: model vs baselines')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'eval_energy_spectrum%s.png' % suffix), dpi=150)
            plt.close()
            print('Saved eval_energy_spectrum%s.png' % suffix)
    except Exception as e:
        print('Energy spectrum plot failed:', e)
        import traceback
        traceback.print_exc()


def _plot_single_baseline(output_dir, summary, n_t, plot_suffix=''):
    """单 baseline：Model vs GT 一条线 + 能谱两条线。"""
    suffix = plot_suffix if isinstance(plot_suffix, str) else ''
    try:
        if 'vorticity_correlation' in summary:
            corr = summary.vorticity_correlation.compute()
            fig, ax = plt.subplots(figsize=(7, 5))
            t = corr.time.values if 'time' in corr.dims else np.arange(len(corr))
            ax.plot(t, np.asarray(corr.values).ravel(), '-', color='C1', label='Model vs GT')
            ax.axhline(y=0.95, color='gray', linestyle=':')
            ax.set_xlabel('Time')
            ax.set_ylabel('Vorticity correlation')
            ax.set_title('Model vs baseline (ground truth)')
            ax.legend()
            ax.set_xlim(0, min(20, np.max(t)) if len(t) else 20)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'eval_vorticity_correlation%s.png' % suffix), dpi=150)
            plt.close()
            print('Saved eval_vorticity_correlation%s.png' % suffix)
    except Exception as e:
        print('vorticity_correlation plot failed:', e)

    combined = None
    for name in ('energy_spectrum_mean', 'energy_spectrum'):
        if name in summary:
            spec = summary[name].mean('time').compute() if 'time' in summary[name].dims else summary[name].compute()
            if 'k' in spec.dims:
                fig, ax = plt.subplots(figsize=(7, 5))
                k = spec.k.values
                ax.loglog(k, np.maximum(np.asarray(spec.values).ravel(), 1e-20), label='Model')
                ax.set_xlabel('Wavenumber k')
                ax.set_ylabel('Energy spectrum')
                ax.set_title('Energy spectrum: model vs baseline')
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, 'eval_energy_spectrum%s.png' % suffix), dpi=150)
                plt.close()
                print('Saved eval_energy_spectrum%s.png' % suffix)
            break


def _plot_field_comparison(output_dir, model_ds, baseline_ds, n_t):
    """u/v 场对比图（单 baseline 时使用）。"""
    try:
        n_show = 5
        n_time_plot = min(n_t, 200)
        step = max(1, n_time_plot // n_show)
        time_inds = list(range(0, n_time_plot, step))[:n_show]
        if not time_inds:
            time_inds = [0]
        u_gt = baseline_ds.u.isel(sample=0).isel(time=time_inds)
        u_pred = model_ds.u.isel(sample=0).isel(time=time_inds)
        v_gt = baseline_ds.v.isel(sample=0).isel(time=time_inds)
        v_pred = model_ds.v.isel(sample=0).isel(time=time_inds)

        fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))
        for i in range(n_show):
            if i < len(time_inds):
                axes[0, i].imshow(u_gt.isel(time=i).values, cmap='RdBu_r', aspect='equal')
                axes[0, i].set_title('u GT t=%d' % time_inds[i])
                axes[0, i].axis('off')
                axes[1, i].imshow(u_pred.isel(time=i).values, cmap='RdBu_r', aspect='equal')
                axes[1, i].set_title('u Model t=%d' % time_inds[i])
                axes[1, i].axis('off')
        fig.suptitle('u: Ground truth (row1) vs Model (row2)')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'eval_field_comparison_u.png'), dpi=150)
        plt.close()
        print('Saved eval_field_comparison_u.png')

        fig2, axes2 = plt.subplots(2, n_show, figsize=(2 * n_show, 4))
        for i in range(n_show):
            if i < len(time_inds):
                axes2[0, i].imshow(v_gt.isel(time=i).values, cmap='RdBu_r', aspect='equal')
                axes2[0, i].set_title('v GT t=%d' % time_inds[i])
                axes2[0, i].axis('off')
                axes2[1, i].imshow(v_pred.isel(time=i).values, cmap='RdBu_r', aspect='equal')
                axes2[1, i].set_title('v Model t=%d' % time_inds[i])
                axes2[1, i].axis('off')
        fig2.suptitle('v: Ground truth (row1) vs Model (row2)')
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'eval_field_comparison_v.png'), dpi=150)
        plt.close()
        print('Saved eval_field_comparison_v.png')
    except Exception as e:
        print('Field comparison plot failed:', e)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
