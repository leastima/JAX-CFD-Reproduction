import xarray as xr
import numpy as np

encode_steps = 16
pred = xr.open_dataset('model/predict.nc')
ref  = xr.open_dataset('content/kolmogorov_re_1000/long_eval_2048x2048_64x64.nc')

dt_model = float(pred.time.values[1] - pred.time.values[0])
dt_ref   = float(ref.time.values[1] - ref.time.values[0])
step = int(round(dt_ref / dt_model))
print('dt_model=%.5f  dt_ref=%.5f  step=%d' % (dt_model, dt_ref, step))
print('encode offset = %d frames * %.4f = %.3f time units' % (encode_steps, dt_ref, encode_steps * dt_ref))
print()
print('%6s  %12s  %14s' % ('t', 'wrong(ref[i])', 'correct(ref[i+16])'))

for i in range(20):
    a = pred.u.isel(sample=0, time=i * step).values.ravel()
    w = ref.u.isel(sample=0, time=i).values.ravel()
    c = ref.u.isel(sample=0, time=i + encode_steps).values.ravel()
    norm = lambda x: float(np.linalg.norm(x))
    cw = float(np.dot(a, w)) / (norm(a) * norm(w) + 1e-12)
    cc = float(np.dot(a, c)) / (norm(a) * norm(c) + 1e-12)
    print('t=%5.2f  wrong=%7.4f  correct=%7.4f' % (i * dt_ref, cw, cc))
