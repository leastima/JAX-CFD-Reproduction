#!/usr/bin/env bash
# ============================================================
# JAX-CFD 环境配置脚本
# 用法: bash setup_env.sh [cuda_version]
#   cuda_version: 12 or 13 (default: 13)
# ============================================================
set -e

CUDA_VER="${1:-13}"
ENV_NAME="cfd-gpu"
PYTHON_VERSION="3.11"

echo "=== Step 1: 创建 conda 环境 (Python ${PYTHON_VERSION}) ==="
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "=== Step 2: 安装 JAX (CUDA ${CUDA_VER}) ==="
pip install "jax[cuda${CUDA_VER}]"

echo "=== Step 3: 安装其他依赖 ==="
pip install \
    dm-haiku gin-config flax optax orbax-checkpoint \
    numpy scipy xarray netCDF4 dask pandas tree-math dm-tree \
    matplotlib Pillow seaborn jupyter \
    absl-py einops jmp chex

echo "=== Step 4: 克隆并安装 jax-cfd 子库 ==="
if [ ! -d "jax-cfd" ]; then
    git clone https://github.com/google/jax-cfd.git
fi
pip install -e ./jax-cfd

echo "=== Step 5: 验证 JAX GPU ==="
python3 -c "
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
if str(jax.devices()[0]).startswith('CUDA') or 'gpu' in str(jax.devices()[0]).lower():
    print('[OK] GPU detected')
else:
    print('[WARN] No GPU detected, running on CPU')
"

echo ""
echo "=== 完成！==="
echo "以后每次运行前先: conda activate ${ENV_NAME}"
echo "然后设置 PYTHONPATH:"
echo "  export PYTHONPATH=\$(pwd)/jax-cfd:\$(pwd)/models:\$(pwd)"
