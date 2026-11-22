# 环境锁定（Kaggle 双 T4）

实测来源：`results/blinkmimic_t4_ddp/run_20261122_081356/`，2026-11-22。

| 项 | 实际值 |
|---|---|
| GPU | 2 × Tesla T4，15360 MiB/卡 |
| 驱动 | 580.178.04 |
| Python | 3.13.16（`uv` 创建的 `.venv`） |
| PyTorch | 2.9.0+cu128 |
| CUDA（torch） | 12.8 |
| CUDA Toolkit（Warp 报告） | 12.9 |
| NCCL | 2.27.5+cuda12.9 |
| mjlab | 1.6.0 |
| MuJoCo | 3.11.0 |
| MuJoCo Warp | 3.11.0 |
| Warp | 1.16.0 |
| 上游仓库 | `fuyichen0725/blinkmimic` |
| 上游 commit | `d264658d6c9ce61fb52f9f0f8985a369f3ccc409` |
| 安装命令 | `uv sync --frozen`（143 个包） |

## 本项目补丁

- 文件：`scripts/patches/blinkmimic-t4-ddp-fp16.patch`
- 内容：`--amp fp16` + 静态 loss scale、torchrun 双卡数据并行、梯度 all-reduce、norm buffer 同步、rank 0 日志/checkpoint
- 应用方式：`git apply` 到上述固定 commit

## 已知环境限制

- T4 是 sm_75，无 bf16/TF32 硬件支持；使用 `--amp fp16`。
- Kaggle 双 T4 无 CUDA peer access，训练使用 `NCCL_P2P_DISABLE=1`。
- 正式首跑关闭 `torch.compile`（`DISABLE_COMPILE=True`），避开 Triton/sm_75 风险；性能口径需单独开启对照。
- `uv sync --frozen` 会以 Python 3.13 建环境；上游 `requires-python` 允许 `<3.14`。

## 复现顺序

```bash
git clone https://github.com/fuyichen0725/blinkmimic.git third_party/blinkmimic
git -C third_party/blinkmimic checkout --detach d264658d6c9ce61fb52f9f0f8985a369f3ccc409
git -C third_party/blinkmimic apply ../../scripts/patches/blinkmimic-t4-ddp-fp16.patch
cd third_party/blinkmimic && uv sync --frozen
```
