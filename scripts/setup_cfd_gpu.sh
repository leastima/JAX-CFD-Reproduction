#!/usr/bin/env bash
# 创建 cfd-gpu 环境：针对当前 CUDA（13.0）使用 jax[cuda13]，在 GPU 上跑 smoke 训练/评估。
# 用法：在 JAX-CFD 项目根目录执行  bash scripts/setup_cfd_gpu.sh
#
# 若 pip install jax[cuda13] 报错 "Cannot uninstall ... no RECORD file"：
#   1) 找到 env 的 site-packages，删除无 RECORD 的 .dist-info 目录；
#   2) 或新建干净 env 再执行本脚本，避免在已损坏的 env 上 --force-reinstall。
set -e
set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 当前机器 CUDA 13 → 用 jax[cuda13] + Python 3.11；若无 cuda13 轮子可改为 CUDA_JAX=cuda12 和 PYVER=3.10
CUDA_JAX="${CUDA_JAX:-cuda13}"
PYVER="${PYVER:-3.11}"

echo "=== Creating conda env cfd-gpu (python=$PYVER, jax[$CUDA_JAX]) ==="
conda create -n cfd-gpu python="$PYVER" -y
conda run -n cfd-gpu pip install --upgrade pip
conda run -n cfd-gpu pip install "jax[$CUDA_JAX]"
# PyPI ml_dtypes wheel 可能缺少 float8_e3m4，导致 JAX 报错；从源码构建可修复
conda run -n cfd-gpu pip install pybind11
conda run -n cfd-gpu pip uninstall -y ml_dtypes 2>/dev/null || true
conda run -n cfd-gpu pip install ml_dtypes --no-binary ml_dtypes
conda run -n cfd-gpu pip install -r requirements-cfd-gpu.txt
# DataLoader 使用 torch
conda run -n cfd-gpu pip install torch
conda run -n cfd-gpu pip install -e ./jax-cfd

echo "=== Installing CuDNN >= 9.12 (required by JAX GPU; system 9.10 causes DNN init failure) ==="
conda install -n cfd-gpu -y -c nvidia "cudnn>=9.12" || echo "WARN: CuDNN install failed. After activate cfd-gpu run: conda install -y -c nvidia cudnn"

echo "=== Verifying JAX sees GPU ==="
conda run -n cfd-gpu python -c "
import jax
print('jax backend:', jax.default_backend())
print('devices:', jax.devices())
assert jax.default_backend() == 'gpu', 'Expected GPU backend'
print('OK: cfd-gpu env is ready for GPU training.')
"

echo "=== Done. Activate with: conda activate cfd-gpu ==="
echo "Then run smoke:  bash scripts/run_train_smoke.sh"
