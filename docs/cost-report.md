# 训练成本报告

所有数字来自 Kaggle 实跑日志，不引用论文或他人的 4090/H100 数字。

## Smoke：双卡 DDP + fp16，100 iterations

来源：`results/blinkmimic_t4_ddp/run_20261122_081356/`

| 指标 | 实测 |
|---|---|
| 配置 | 每卡 num_envs 512，pi_batch 512，wm_batch 512，horizon 32 |
| 精度 | fp16，静态 loss scale 256 |
| torch.compile | 关闭 |
| 迭代数 | 100 |
| 训练 wall-clock | 129.0 s |
| 吞吐 | 0.93 it/s |
| 每卡显存峰值 | 2455 MiB / 15360 MiB |
| 全局环境步 | 321,536 |
| nonfinite | 0 |
| GPU-hours | 0.072（2 卡 × 129 s） |
| 结果 | exit_code=0 |

### 训练日志摘要

```text
it 10  wm 5.3896  e8 14.6217  pi -0.486  surv 0.206  ep_ret 0.264
it 50  wm 0.9718  e8  2.0426  pi -0.829  surv 0.434  ep_ret 0.280
it 100 wm 0.6053  e8  1.1666  pi -1.710  surv 0.723  ep_ret 0.087
```

世界模型误差下降，但真实仿真回报在 100 步内下降。结论仅限“管线可运行”，不构成策略收敛证据。

## 正式训练 baseline（进行中）

观测到资源占用（`run_20261122_083958`，150 iterations）：

| 指标 | 实测 |
|---|---|
| 每卡显存 | 5.7 GiB / 15 GiB |
| GPU 利用率 | 61% / 100% |
| CPU | 200% |
| RAM | 14.9 / 30 GiB |
| 吞吐 | 0.88 it/s |

结论：显存和 CPU 都有余量，GPU 利用率是主要瓶颈；仿真不是瓶颈（512→2048 环境吞吐几乎不变）。

## high_util 档（待跑）

计划配置：

| 项 | 值 |
|---|---|
| 数据 | 40 段 LAFAN1 G1 重定向 |
| 每卡 num_envs | 4096 |
| 每卡 pi_batch / wm_batch | 2048 / 2048 |
| policy 宽度 | 2048-1024-512-256 |
| horizon | 32 |
| iterations | 3500 |
| 精度 | fp16，loss scale 256 |
| torch.compile | 开启 + compile_more |
| checkpoint | 每 500 iter 或 10 分钟，完整状态 |
| 备份 | Hugging Face 私有 model 仓库 |

正式数字在跑完后补入，不使用估算值。
