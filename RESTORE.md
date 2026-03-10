# 还原到可运行流程（仿 ML4HPC/JAX-CFD）

参考 [ML4HPC/JAX-CFD](https://github.com/ML4HPC/JAX-CFD)：用已有数据 → 训练 → 画图。

## 一键跑通（smoke）

先激活 conda 环境（与 `run_train_smoke.sh` 相同，例如 `cfd-gpu`），再执行：

```bash
conda activate cfd-gpu   # 或你的 env 名
cd /jumbo/yaoqingyang/yuxin/JAX-CFD
bash scripts/run_restore_smoke.sh
```

- **数据**：使用 `content/kolmogorov_re_1000/train_2048x2048_64x64.nc` 与 `eval_2048x2048_64x64.nc`（与 `run_train_smoke.sh` / `run_train_paper.sh` 一致）。若目录或文件不存在，脚本会报错并提示。
- **训练**：极简配置（encode=decode=predict=2，约 0.002 epoch），不启用 `only_trajectory_start`，不生成数据、不跑 DNS baseline。
- **画图**：`scripts/plot_loss_and_eval_from_predict.py` 画 loss 与 eval 对比图。

输出目录：`model_restore_smoke/`（loss 曲线、predict.nc、eval_*.png）。

## 与参考仓库的对应关系

| 参考 README | 本仓库 |
|------------|--------|
| 数据在 `/global/.../content` | 数据在 `content/kolmogorov_re_1000/` |
| 脚本在 `scripts/` | `scripts/run_restore_smoke.sh`、`run_train_smoke.sh`、`run_train_paper.sh` |
| 可视化 `plot_fig*.py` | `scripts/plot_loss_and_eval_from_predict.py` |

## 本仓库保留的修改（为支持 batch 运行）

以下修改是为了在 **batch > 1** 时能跑通，已保留、未回退：

- **`models/my_equations.py`**：对 state 的 leading 维做 vmap，使物理步（BC/fd）只看到 2D 网格，避免 `jnp.pad` 维数错误。
- **`models/my_interpolations.py`**：`extract_patches` 只对空间维做 roll，并对 batch 维 vmap，避免 `_extract_patches_roll` 的 shift/axis 维数不一致。

若只用单样本（batch=1、单 device），理论上可不用上述 vmap，但当前脚本与配置为多 device/batch 设计，因此保留这些修改。

---

## 原始仓库（jax-cfd）是怎么做的

**结论：原始 jax-cfd 假定「物理步和插值里没有 batch 维」，batch 由调用方用 vmap 消化。**

### 1. 方程（`jax_cfd/ml/equations.py`）

- `modular_navier_stokes_model` 里的 `navier_stokes_step_fn(state)` 直接做 `v = state`，然后 `convection_module(..., v=v)`、`step_fn(v)`。
- 没有任何对 batch 维的处理，**假定** `state` 是 `GridVariable` 的元组，每个 `.data` 都是 **2D (x, y)**。
- 边界、有限差分（含 `jnp.pad`）都是按 2D 写的，多一维就会报错。

### 2. 插值 / extract_patches（`jax_cfd/ml/layers_util.py`）

- `_extract_patches_roll(x, patch_shape)` 里：`roll_axes = range(x.ndim)`，`rolls` 由 `patch_shape` 生成，长度等于**空间维数**（2D 则为 2）。
- 设计假定 `x` 的 shape 是 **(空间维 + 可选的 channel)**，例如 `(H, W)` 或 `(H, W, C)`，**没有 batch 维**。
- 若传入 `(batch, H, W, C)`，就会变成「2 个 shift vs 3 个 axis」的维数不一致。

### 3. 训练里 batch 应该怎么处理（`jax_cfd/ml/train_utils.py`、`model_utils.py`）

- `train_utils` 注释写明：下面这些函数都处理 **batched** 的输入。
- `decoded_trajectory_with_inputs` 返回的 `trajectory_fn` **没有**在内部做 vmap；接口是「一次接受一个 trajectory」。
- 因此 **谁调 trajectory，谁就要负责对 batch 做 vmap**：对 batch 维做 `vmap(trajectory_fn, in_axes=0)`，这样每次只传一个样本 `(time, x, y)`，encoder/advance 里就始终是 2D state，和上面 1、2 的假定一致。

### 4. 本仓库和原始的差异

- 我们的 `train.py` 里是：`trajectory = jax.vmap(..., axis_name='i')`，然后 `trajectory(inputs)`。
- DataLoader 给的 `inputs` 是 **按 device 分片后的**：shape 为 `(devices, batch_per_device, time, x, y)`，例如 `(4, 32, 96, 64, 64)`。
- 这里 `vmap(..., axis_name='i')` 默认是沿 **axis 0**（即 **devices**）做 vmap，不是沿 **batch_per_device**。所以每个「设备」上的 trajectory 拿到的仍是 `(32, 96, 64, 64)`，即 **batch=32** 一起进 encoder/advance，state 变成 `(32, 64, 64)`，物理和插值就看到了 3D，和原始 2D 假定冲突。
- 原始做法等价于：要么 **batch_per_device=1**，要么在 trajectory 外再对 **batch 维**做一层 vmap，让每次只进一个样本，这样 state 始终 2D，就不需要改 equations/interpolations。

### 5. 两种对齐方式

| 做法 | 说明 |
|------|------|
| **A. 改训练：对 batch 维 vmap** | 在现有 `vmap(..., axis_name='i')` 之外，再对 `batch_per_device` 维做 vmap，使 trajectory 每次只收 `(time, x, y)`，state 保持 2D；则可撤掉我们在 `my_equations` / `my_interpolations` 里加的 vmap，完全按原始库的 2D 假定跑。 |
| **B. 保持当前设计（本仓库做法）** | 不增加对 batch 的 vmap，继续让 trajectory 一次吃整块 `(batch_per_device, time, x, y)`；在 `my_equations` 和 `my_interpolations` 里对 leading 维做 vmap，让物理和 extract_patches 内部只看到 2D。 |

当前仓库采用的是 **B**，所以保留了 `my_equations` 和 `my_interpolations` 的修改。

## 相对原始 jax-cfd / 本仓库的修改清单

（仅列与「能跑通」相关的、会改行为的改动；新增脚本/配置不列。）

| 位置 | 修改内容 |
|------|----------|
| **models/train.py** | 用 `init_batch = next(train_dataset)` 做 init；`--dataset_num_workers`；DNS 时走 `decoded_trajectory_with_inputs` 无 `is_training`；loss/eval/predict 的 time 切片；predict 写 nc 时 time 坐标；可选 `--no_train` / `--no_dropout`。 |
| **models/my_model_builder.py** | `MyModularStepModel`、`my_trajectory_from_step`、`my_decoded_trajectory_with_inputs`；`with_split_input` / `my_with_input_included` 用 `time_axis=1`。 |
| **models/my_equations.py** | `my_modular_navier_stokes_model`：对 state 的 leading 维 vmap，使物理步只看到 2D；`u.data[-1]` 取「最后时间」再进 physics。 |
| **models/my_interpolations.py** | `MyFusedLearnedInterpolation`：init 时对 6D/5D/4D/3D 的 `inputs` 统一成 (x,y,-1)；`MySpatialDerivativeFromLogits.extract_patches` 只对空间维 roll 并对 batch vmap。 |
| **models/my_encoders.py** | `my_aligned_array_encoder`（与 decoder 对齐的接口）。 |
| **models/my_decoders.py** | `my_aligned_array_decoder`。 |
| **models/my_towers.py** | `my_forward_tower_factory` 等（gin 可配置）。 |
| **models/my_advections.py** | 可选（若 gin 用）。 |
| **data/generate_train_eval_nc.py** | 自研数据生成；nc 的 attrs/coords/offset 与 notebook 一致。 |
| **scripts/plot_loss_and_eval_from_predict.py** | 画 loss + eval；LI 的 rollout 时间对齐、多 baseline 对比。 |

原始 jax-cfd 未改：`jax-cfd/jax_cfd/ml/equations.py`、`layers_util.py`、`model_utils.py`、`interpolations.py` 等仍为上游实现；我们通过「my_*」层包一层或替换 gin 绑定来适配 batch/多卡。

## 其他脚本

- **`run_train_smoke.sh`**：与 restore 类似，同数据路径，多一点调试日志；通过即说明流程正常。
- **`run_train_paper.sh`**：论文规模训练（64 步、90 epoch 等），数据路径同上。
- **`run_li_only_no_baseline.sh`**：用生成数据只训 LI、不跑 baseline（需先有或生成 `long_eval_2048x2048_64x64.nc`）。
