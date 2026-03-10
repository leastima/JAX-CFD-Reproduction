#!/usr/bin/env python3
"""
对比训练集与评估集：维度（样本数、时间步数）、在评估步数内是否一致。

若 eval 是 train 的前缀（同一批轨迹、同一初值，eval 只取前 N 步），则在前 N 步内
train 与 eval 应逐点一致；否则两集是不同轨迹（不同初值/不同 seed），评估时用的
「初值」与训练集无直接对应关系。

用法:
  python scripts/compare_train_eval_data.py \\
    --train_nc=/path/to/train_2048x2048_64x64.nc \\
    --eval_nc=/path/to/eval_2048x2048_64x64.nc
"""
import argparse
import numpy as np
import xarray


def main():
    p = argparse.ArgumentParser(description='Compare train vs eval nc: shape and overlap.')
    p.add_argument('--train_nc', type=str, required=True, help='Path to train nc')
    p.add_argument('--eval_nc', type=str, required=True, help='Path to eval nc')
    p.add_argument('--atol', type=float, default=1e-6,
                    help='Absolute tolerance for float equality (default 1e-6)')
    args = p.parse_args()

    train = xarray.open_dataset(args.train_nc)
    eval_ds = xarray.open_dataset(args.eval_nc)

    n_train_sample = train.sizes.get('sample', 0)
    n_train_time = train.sizes.get('time', 0)
    n_eval_sample = eval_ds.sizes.get('sample', 0)
    n_eval_time = eval_ds.sizes.get('time', 0)

    print('=== 1) 维度对比 ===')
    print('训练集 train:  sample=%s   time=%s   (x,y)=(%s,%s)' % (
        n_train_sample, n_train_time,
        train.sizes.get('x'), train.sizes.get('y')))
    print('评估集 eval:   sample=%s   time=%s   (x,y)=(%s,%s)' % (
        n_eval_sample, n_eval_time,
        eval_ds.sizes.get('x'), eval_ds.sizes.get('y')))

    # 评估用到的步数：通常就是 eval 的整条长度
    steps_in_eval = n_eval_time
    print('\n评估时使用的时间步数（eval 长度）: %s' % steps_in_eval)

    # 在「评估步数」内，train 是否有足够长度
    if n_train_time < steps_in_eval:
        print('注意: 训练集时间步 (%s) < 评估步数 (%s)，无法在评估长度内逐段对比。' % (
            n_train_time, steps_in_eval))
    else:
        print('训练集时间步 >= 评估步数，可以在前 %s 步内做逐点对比。' % steps_in_eval)

    # 样本数是否一致
    n_common = min(n_train_sample, n_eval_sample)
    if n_common == 0:
        print('\n无法对比：至少一方样本数为 0。')
        train.close()
        eval_ds.close()
        return

    # 在「评估步数」内的重叠长度
    overlap_steps = min(n_train_time, n_eval_time, steps_in_eval)

    print('\n=== 2) 在评估步数内是否一致（前 %s 步） ===' % overlap_steps)
    if overlap_steps == 0:
        print('无重叠步数，跳过数值对比。')
        train.close()
        eval_ds.close()
        return

    train_u = np.asarray(train['u'].values)
    train_v = np.asarray(train['v'].values)
    eval_u = np.asarray(eval_ds['u'].values)
    eval_v = np.asarray(eval_ds['v'].values)

    # 逐 sample 比较前 overlap_steps 步
    same = True
    max_diff_u = 0.0
    max_diff_v = 0.0
    for s in range(n_common):
        # train[s, :overlap_steps], eval[s, :overlap_steps]
        tu = train_u[s, :overlap_steps]
        tv = train_v[s, :overlap_steps]
        eu = eval_u[s, :overlap_steps]
        ev = eval_v[s, :overlap_steps]
        if tu.shape != eu.shape:
            print('  sample %s: shape 不一致 train %s eval %s' % (s, tu.shape, eu.shape))
            same = False
            continue
        du = np.abs(tu - eu)
        dv = np.abs(tv - ev)
        max_diff_u = max(max_diff_u, float(du.max()))
        max_diff_v = max(max_diff_v, float(dv.max()))
        if not (np.allclose(tu, eu, atol=args.atol) and np.allclose(tv, ev, atol=args.atol)):
            same = False

    if same:
        print('  结论: 在评估步数内，训练集与评估集 **一致**（eval 应为 train 的前缀或同一数据）。')
        print('  max |train - eval|: u=%.2e  v=%.2e' % (max_diff_u, max_diff_v))
    else:
        print('  结论: 在评估步数内，训练集与评估集 **不一致**（两集来自不同轨迹/不同初值）。')
        print('  max |train - eval|: u=%.2e  v=%.2e' % (max_diff_u, max_diff_v))

    # 可选：打印 time 坐标前几项是否一致
    if 'time' in train.coords and 'time' in eval_ds.coords:
        t_train = np.asarray(train.coords['time'].values)
        t_eval = np.asarray(eval_ds.coords['time'].values)
        n_t = min(len(t_train), len(t_eval), 5)
        if n_t > 0:
            time_match = np.allclose(t_train[:n_t], t_eval[:n_t], atol=1e-9)
            dt_train = float(t_train[1] - t_train[0]) if len(t_train) > 1 else None
            dt_eval = float(t_eval[1] - t_eval[0]) if len(t_eval) > 1 else None
            print('\n=== 3) time 坐标（前 %s 个） ===' % n_t)
            print('  train time[:%s] = %s' % (n_t, t_train[:n_t]))
            print('  eval  time[:%s] = %s' % (n_t, t_eval[:n_t]))
            print('  train dt=%.6e  eval dt=%.6e' % (dt_train or 0, dt_eval or 0))
            print('  一致: %s' % time_match)

    train.close()
    eval_ds.close()
    print('\nDone.')


if __name__ == '__main__':
    main()
