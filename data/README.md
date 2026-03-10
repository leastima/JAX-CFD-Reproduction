# 数据（与 ML4HPC 原仓一致）

本目录与 [ML4HPC/JAX-CFD](https://github.com/ML4HPC/JAX-CFD) 的 `data/` 一致。

## 训练与评估数据从哪里来

ML4HPC 原仓 **不包含** 生成「训练 / 评估用 NetCDF」的脚本。README 里只写了**绝对路径**（没有提 NERSC 或具体怎么访问）：

- **训练与评估数据**：`/global/cfs/cdirs/m3898/zhiqings/cfd/content`
- **已训练模型**：`/global/cfs/cdirs/m3898/zhiqings/cfd/models`

典型文件名（在 `content/kolmogorov_re_1000/` 下）：`train_2048x2048_64x64.nc`、`long_eval_2048x2048_64x64.nc` 等。

即：**训练/评估用的 .nc 在原仓的设想里就是该路径下已有数据**；若你无法访问该路径，需自备同结构的 .nc 并通过 `STORAGE_PATH` 指向你的目录。

## 该路径是什么、怎么访问

- **路径含义**：`/global/cfs/cdirs/<project>/...` 是 **NERSC（美国国家能源研究科学计算中心）** 的 **Community File System (CFS)** 标准形式；`m3898` 为项目编号，`zhiqings/cfd` 为项目成员/仓库维护者下的目录。详见 [NERSC CFS 文档](https://docs.nersc.gov/filesystems/community)。
- **如何访问**：
  - **在 NERSC 上有账号且属于项目 m3898**：登录 NERSC 机器后可直接访问该路径（或通过 `$CFS` 等环境变量）。
  - **不在 NERSC / 无该项目权限**：原仓未提供公开下载链接。可尝试联系 ML4HPC/JAX-CFD 或数据维护者（路径中的 zhiqings）询问是否可分享数据或权限；否则需自己在本地用 jax-cfd 生成与 `models/dataset.py` 兼容的 .nc（见下文）。

## 本目录脚本的用途

| 脚本 | 用途 |
|------|------|
| **generate_data.py** | 跑 Kolmogorov 2D DNS（warmup + 轨迹），**仅用于 demo**：加 `--demo` 时把不同分辨率的涡量图保存到 `../figs/`，**不写** train/eval 的 .nc。 |
| **re1000_demo.sh** 等 | 在 `data/` 下执行，调用 `generate_data.py ... --demo`，用于出图。 |
| **generate_re1000_train.sh** | 仅设置环境变量（如 `STORAGE_PATH`），不生成 .nc。 |
| **check_shape_train.py** | 从脚本内写死的 `data_path`（即上述 `/global/cfs/...`）读 `content/kolmogorov_re_1000/` 下的 train、long_eval 等 .nc，把 info 写到 `../logs/log_train.txt`。若你数据在别处，需改脚本里的 `data_path`。 |
| **check_shape_eval.py** | 从 `data_path` 的 `content/kolmogorov_re_1000_fig1/` 读 baseline/learned .nc，写 `../logs/log_eval.txt` 并出图到 `../samples/`。同样可改 `data_path` 指向本地目录。 |

## 若无法访问 NERSC 路径

- 需要自己生成与 `models/dataset.py` 兼容的 NetCDF（含 `sample`、`time`、`x`、`y` 及变量 `u`、`v`）。
- 可参考 jax-cfd 子项目中的 notebook（如 `ml_accelerated_cfd_data_analysis.ipynb`、`ml_model_inference_demo.ipynb`）或自写脚本，用 `jax_cfd.base` + `jax_cfd.ml.forcings.kolmogorov_forcing` 跑 DNS 并写入与上述结构一致的 .nc。
