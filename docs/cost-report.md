# 训练与评测成本报告

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

训练日志摘要：

```text
it 10  wm 5.3896  e8 14.6217  pi -0.486  surv 0.206  ep_ret 0.264
it 50  wm 0.9718  e8  2.0426  pi -0.829  surv 0.434  ep_ret 0.280
it 100 wm 0.6053  e8  1.1666  pi -1.710  surv 0.723  ep_ret 0.087
```

该次只证明管线可运行，不构成收敛证据。

## 正式训练 baseline（已完成）

来源：`results/blinkmimic_t4_ddp/run_20261122_083958/`，退出码 0。

| 指标 | 实测 |
|---|---|
| 数据 | 40 段 LAFAN1 G1 重定向 |
| 每卡 envs / pi_batch / wm_batch | 2048 / 1024 / 1024 |
| policy 宽度 | 1024-512-256 |
| torch.compile | 关闭 |
| iterations | 3500 |
| wall-clock | 3980.7 s = 1.106 h |
| GPU-hours | 2.212（2 卡） |
| 每卡显存峰值 | 5879 MiB / 15360 MiB |
| 吞吐 | 0.906 it/s |
| nonfinite | 0 |

最终训练日志（iteration 3500）：

```text
wm_loss 0.1233  wm_err8 0.2057  pi_loss -9.120
imag_ret 0.903  imag_survival 0.999
sim_r 0.804  ep_len 487.3  ep_ret 0.792  term 0.0046
```

说明：`ep_ret` 是训练期间的回报，不等于正式评测指标。该 run 的 40 段评测结果见 [coverage-and-failure-analysis.md](coverage-and-failure-analysis.md)。

## 正式训练高利用率配置（已完成）

来源：`results/blinkmimic_t4_ddp/run_20261122_105913/`，退出码 0。

| 指标 | 实测 |
|---|---|
| 数据 | 40 段 LAFAN1 G1 重定向 |
| 每卡 envs / pi_batch / wm_batch | 4096 / 3072 / 2048 |
| policy 宽度 | 2048-1024-512-256 |
| torch.compile | 开启，`compile_more` 关闭 |
| iterations | 3500 |
| wall-clock | 4347.3 s = 1.208 h |
| GPU-hours | 2.415（2 卡） |
| 每卡显存峰值 | 12589 MiB / 15360 MiB |
| 吞吐 | 约 0.874 it/s |
| nonfinite | 0 |

最终训练日志（iteration 3500）：

```text
wm_loss 0.1037  wm_err8 0.1719  pi_loss -9.417
imag_ret 0.930  imag_survival 0.9995
sim_r 0.822  ep_len 344.7  ep_ret 0.625  term 0.00087
```

## 正式评测成本

两个 checkpoint 在同一次 Kaggle 会话中并行评测，GPU 0 运行旧模型，GPU 1 运行新模型。指标评测使用 40 段动作；轨迹录制随后并行完成。

| 项目 | 旧模型 | 新模型 |
|---|---:|---:|
| 指标评测 wall-clock | 758.5 s | 747.6 s |
| 轨迹录制 wall-clock | 449.6 s | 450.0 s |
| 指标失败动作 | 0 / 40 | 0 / 40 |
| 评测设备 | T4 GPU 0 | T4 GPU 1 |

评测产生的 JSON、Markdown、NPZ 轨迹和 4 段预览视频已上传到 HF 对应 run 的 `eval/` 目录。之后上传的 `eval_video_all40.mp4` 是每段固定 3 秒的合辑，可能截断长动作，不作为完整动作视频交付。渲染代码已修正为按 `clip_len` 逐个完整渲染，每个动作独立 MP4，并通过 `ffprobe` 校验；修正版需在 Kaggle 重新运行，渲染耗时待该次运行日志记录。

### 资源观测（150 iterations 时）

| 指标 | 实测 |
|---|---|
| 每卡显存 | 5.7 GiB / 15 GiB |
| GPU 利用率 | 61% / 100% |
| CPU | 200% |
| RAM | 14.9 / 30 GiB |

结论：显存和 CPU 都有余量，GPU 利用率是主要瓶颈；仿真不是瓶颈（512→2048 环境吞吐几乎不变）。

## high_util 档（历史配置，已由正式高利用率 run 覆盖）

| 项 | 值 |
|---|---|
| 每卡 num_envs | 4096 |
| 每卡 pi_batch / wm_batch | 2048 / 2048 |
| policy 宽度 | 2048-1024-512-256 |
| horizon | 32 |
| iterations | 3500 |
| 精度 | fp16，loss scale 256 |
| torch.compile | 开启 + compile_more |

## max 档（未执行）

| 项 | 值 |
|---|---|
| 每卡 num_envs | 8192 |
| 每卡 pi_batch / wm_batch | 3072 / 2048 |
| policy 宽度 | 2048-1024-512-256 |
| CPU | 8 核时每 rank 4 线程 |
| iterations | 3500 |
| 精度 | fp16，loss scale 256 |
| torch.compile | 开启 + compile_more |

max 档的 DDP 通信已优化：梯度按模块打平成一次 all-reduce，norm/EMA buffer 按 dtype 打包广播，stop flag 每 10 步检查。

正式数字在跑完后补入，不使用估算值。
