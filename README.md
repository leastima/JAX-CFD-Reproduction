# JAX-CFD
Machine Learning-accelerated Computational Fluid Dynamics (CFD)

---

## 快速部署（新服务器）

```bash
# 1. 克隆仓库
git clone <YOUR_REPO_URL> JAX-CFD
cd JAX-CFD

# 2. 一键建环境（需要先安装 miniconda）
bash setup_env.sh 12   # CUDA 12  (A100/V100)
# 或
bash setup_env.sh 13   # CUDA 13  (H100/RTX 4000)

# 3. 激活环境，设置 PYTHONPATH
conda activate cfd-gpu
export PYTHONPATH=$(pwd)/jax-cfd:$(pwd)/models:$(pwd)

# 4. 生成训练数据（示例：Re=1000）
python scripts/generate_kolmogorov_data.py \
    --re 1000 \
    --output content/kolmogorov_re1000/train.nc \
    --num_samples 256 \
    --save_size 64 \
    --seed 0

# 5. 启动大规模实验（tlen phase plot 实验）
bash scripts/train_tlen_v2.sh
```

> 详细说明见下方各节。

---

# Datasets
The training data and evaluation data are in `/global/cfs/cdirs/m3898/zhiqings/cfd/content`

The trained models are in `/global/cfs/cdirs/m3898/zhiqings/cfd/models`

# Installation

**推荐方式（GPU 服务器）**：

```bash
bash setup_env.sh [12|13]   # 传入 CUDA 大版本号
```

该脚本会自动完成：conda 建环境 → 安装 JAX(GPU) → 安装所有依赖 → clone & 安装 jax-cfd 子库。

**jax-cfd 子项目**：若已 clone 到 `JAX-CFD/jax-cfd`，在 JAX-CFD 根目录执行 `pip install -e "./jax-cfd[complete]"`；否则 `git clone https://github.com/google/jax-cfd.git jax-cfd && pip install -e "./jax-cfd[complete]"`

# Run

环境激活后，在**项目根目录**执行（`PYTHONPATH` 必须包含当前目录）。

## 1. 数据

- **无法访问原路径时（推荐）**：用本仓库 + jax-cfd 在本地生成 train/eval 用的 .nc，不依赖 `/global/cfs/...`：
  1. 装好环境后安装 jax-cfd 子项目：`pip install -e "./jax-cfd[complete]"`（在 JAX-CFD 根目录执行）。
  2. 生成数据（会写出 `content/kolmogorov_re_1000/train_2048x2048_64x64.nc` 和 `long_eval_2048x2048_64x64.nc`）：
     ```bash
     cd /path/to/JAX-CFD
     conda activate cfd && export PYTHONPATH=$PWD
     python data/generate_train_eval_nc.py --output_dir ./content/kolmogorov_re_1000 --train_samples 4 --eval_samples 2 --train_steps 256 --eval_steps 512
     ```
  3. 训练时把数据目录指到当前项目根目录即可，例如：`STORAGE_PATH=$PWD`（因为数据在 `./content/kolmogorov_re_1000/` 下）。
- **直接用已有数据**：若你能访问原数据路径，设 `STORAGE_PATH=/global/cfs/cdirs/m3898/zhiqings/cfd`，训练/评估用的就是该路径下的 `content/kolmogorov_re_1000/train_2048x2048_64x64.nc` 和 `long_eval_2048x2048_64x64.nc`。
- **用自己的数据**：把 `STORAGE_PATH` 设成你的目录，保证下面有与 `content/kolmogorov_re_1000/` 同结构的 NetCDF（含 `sample`、`time`、`x`、`y` 及变量 `u`、`v`）。更多样本/步数可调 `generate_train_eval_nc.py` 的 `--train_samples`、`--eval_samples`、`--train_steps`、`--eval_steps`。

## 2. 训练 + 评估

单机、单卡示例（把 `STORAGE_PATH` 和 `MODEL_NAME` 换成你的路径和模型名）：

```bash
cd /jumbo/yaoqingyang/yuxin/JAX-CFD
conda activate cfd
export PYTHONPATH=$PWD

STORAGE_PATH=/global/cfs/cdirs/m3898/zhiqings/cfd   # 或你的数据根目录
MODEL_NAME=my_model
TRAINDATA=content/kolmogorov_re_1000/train_2048x2048_64x64.nc
EVALDATA=content/kolmogorov_re_1000/long_eval_2048x2048_64x64.nc

python -u models/train.py \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --model_encode_steps=16 \
  --model_decode_steps=160 \
  --model_predict_steps=16 \
  --train_device_batch_size=4 \
  --eval_batch_size=48 \
  --delta_time=0.007012483601762931 \
  --train_epochs=0.2 \
  --train_log_every=10 \
  --do_eval \
  --gin_file="models/configs/official_li_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin"
```

训练时加 `--do_eval` 会在训练结束后跑一次评估；更多超参和用法见 `scripts/run_interpolation.sh`、`scripts/run_dns.sh` 等。

## 3. 只做评估（不训练）

已有 checkpoint 时，只评估：

```bash
python -u models/train.py \
  --train_split="$STORAGE_PATH/$TRAINDATA" \
  --eval_split="$STORAGE_PATH/$EVALDATA" \
  --output_dir="$STORAGE_PATH/models/$MODEL_NAME" \
  --no_train \
  --do_eval \
  --resume_checkpoint \
  --gin_file="models/configs/official_li_config.gin" \
  --gin_file="models/configs/kolmogorov_forcing.gin" \
  ... # 其他 model_encode_steps 等与训练时一致
```

## 4. 可视化

画图脚本在根目录：`plot_fig1.py`、`plot_fig2.py`、`plot_fig3.py`；更多复现脚本在 `scripts/` 下。
