#!/usr/bin/env python3
"""
验证 (1) baseline/reference 与 predict 用的 eval 是否同源；
    (2) 作者提供的 nc 文件基本是否合理（shape、属性、t=0 与 predict 初值是否一致）。

用法:
  python scripts/verify_baseline_and_predict.py \\
    --predict_nc=model_paper_gpu67/predict.nc \\
    --baseline_nc=content/kolmogorov_re_1000/eval_2048x2048_64x64.nc \\
    [--eval_split 与训练/预测时用的 eval 路径一致时可选]
"""
import argparse
import numpy as np
import xarray


def main():
    p = argparse.ArgumentParser(description='Verify baseline vs predict and basic nc sanity.')
    p.add_argument('--predict_nc', type=str, required=True, help='Path to predict.nc')
    p.add_argument('--baseline_nc', type=str, required=True,
                    help='Path to baseline/reference nc (e.g. eval_2048x2048_64x64.nc)')
    p.add_argument('--eval_split', type=str, default=None,
                    help='Eval path used in training/predict (if same as baseline_nc, no need)')
    args = p.parse_args()

    pred = xarray.open_dataset(args.predict_nc)
    base = xarray.open_dataset(args.baseline_nc)

    print('=== 1) Shape & dims ===')
    print('predict.nc: sample=%s time=%s x=%s y=%s' % (
        pred.sizes.get('sample'), pred.sizes.get('time'),
        pred.sizes.get('x'), pred.sizes.get('y')))
    print('baseline:   sample=%s time=%s x=%s y=%s' % (
        base.sizes.get('sample'), base.sizes.get('time'),
        base.sizes.get('x'), base.sizes.get('y')))

    print('\n=== 2) Baseline attrs (author data sanity) ===')
    for k in ['ndim', 'stable_time_step', 'simulation_grid_size', 'save_grid_size']:
        if k in base.attrs:
            print('  %s = %s' % (k, base.attrs[k]))
    if 'time' in base.coords and len(base.time) > 0:
        dt = float(base.time[1] - base.time[0]) if len(base.time) > 1 else None
        print('  time[0]=%s dt=%s' % (float(base.time[0]), dt))

    print('\n=== 3) Same source? (predict t=0 vs baseline t=0) ===')
    n_s = min(pred.sizes.get('sample', 1), base.sizes.get('sample', 1))
    pred_u0 = np.asarray(pred['u'].isel(sample=slice(0, n_s), time=0).values)
    base_u0 = np.asarray(base['u'].isel(sample=slice(0, n_s), time=0).values)
    pred_v0 = np.asarray(pred['v'].isel(sample=slice(0, n_s), time=0).values)
    base_v0 = np.asarray(base['v'].isel(sample=slice(0, n_s), time=0).values)
    if pred_u0.shape != base_u0.shape:
        print('  Shape mismatch: pred t=0 %s vs base t=0 %s -> cannot compare.' % (pred_u0.shape, base_u0.shape))
    else:
        err_u = np.abs(pred_u0 - base_u0).mean()
        err_v = np.abs(pred_v0 - base_v0).mean()
        print('  Mean |pred - base| at t=0: u=%.2e v=%.2e' % (err_u, err_v))
        if err_u < 1e-5 and err_v < 1e-5:
            print('  -> t=0 一致，predict 与 baseline 同源或数值上等价。')
        else:
            print('  -> t=0 有差异：baseline 与做 rollout 时用的 eval 可能不是同一文件，或下采样/存储方式不同。')

    if args.eval_split and args.eval_split != args.baseline_nc:
        print('\n=== 4) Eval_split vs baseline_nc ===')
        try:
            ev = xarray.open_dataset(args.eval_split)
            print('  eval_split sample=%s time=%s' % (ev.sizes.get('sample'), ev.sizes.get('time')))
            ev_u0 = np.asarray(ev['u'].isel(sample=0, time=0).values)
            base_u0_one = np.asarray(base['u'].isel(sample=0, time=0).values)
            if ev_u0.shape == base_u0_one.shape:
                print('  |eval_split[0,0] - baseline[0,0]| u mean=%.2e' % np.abs(ev_u0 - base_u0_one).mean())
            ev.close()
        except Exception as e:
            print('  Failed to open eval_split: %s' % e)
    pred.close()
    base.close()
    print('\nDone.')


if __name__ == '__main__':
    main()
