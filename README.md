# 项目二：双卡 T4 约束下的人形机器人多动作全身跟踪

纯仿真项目。Unitree G1，基于 BlinkMimic（mjlab / MuJoCo Warp），双卡 T4 数据并行 + fp16。正式结果只来自 Kaggle 双 T4 实跑。

## 状态

- 2026-11-22：双卡 DDP + fp16 冒烟通过（100 iterations，0.072 GPU-h，2.4 GB/卡）
- 2026-11-22：40 段 LAFAN1 baseline 完成（3500 iterations，`exit_code=0`，2.21 GPU-h，5.9 GB/卡）
- 2026-11-22：高利用率配置完成（3500 iterations，`exit_code=0`，2.42 GPU-h，12.6 GB/卡）
- 2026-11-22：两个 checkpoint 完成 40 段正式指标评测，均为 `0/40` 失败；新模型 `norm_return=0.8762`
- 2026-11-22：两个 checkpoint 的指标 JSON、轨迹 NPZ 和 4 段预览视频已上传 HF 私有仓库
- 当前收尾：完整 40 段视频需运行 `scripts/render_all40_t4.sh`；不影响已完成的 40 段指标评测
- 仅仿真验证，未上真机

## 目录

```text
docs/            环境锁定、数据清单、成本报告、正式流程
configs/         正式训练参数
scripts/         Kaggle 启动脚本、T4 补丁、备份工具
third_party/     上游仓库（不提交）
data/            动作数据（不提交）
results/         日志与 checkpoint（不提交）
kaggle_blinkmimic_t4_ddp.ipynb   可直接 Run All 的训练 Notebook
kaggle_eval_t4.ipynb             训练后指标、轨迹与预览视频
docs/coverage-and-failure-analysis.md 40 段覆盖率、指标与失败分析
```

## 环境

见 [docs/env-lock.md](docs/env-lock.md)。核心版本：

- 2 × Tesla T4 16 GB，Kaggle
- Python 3.13.16，PyTorch 2.9.0+cu128
- mjlab 1.6.0，MuJoCo 3.11.0，Warp 1.16.0
- 上游 `fuyichen0725/blinkmimic` commit `d264658`

## 相对上游的修改

全部通过补丁完成，不改上游算法：

- `--amp fp16`：T4 无 bf16，新增 fp16 + 静态 loss scale
- 双卡数据并行：torchrun、NCCL 梯度平均、初始权重广播、norm buffer 同步
- rank 0 写日志和 checkpoint
- 不支持 DDP + `--cuda_graph` / `--eprobe_share`

补丁：[scripts/patches/blinkmimic-t4-ddp-fp16.patch](scripts/patches/blinkmimic-t4-ddp-fp16.patch)

## 数据

LAFAN1 G1 重定向，30 fps CSV。许可 CC BY-NC-ND 4.0，仅个人/非商业研究，不公开再分发。详见 [docs/data-manifest.md](docs/data-manifest.md)。

## 运行

Kaggle 上传 `kaggle_blinkmimic_t4_ddp.ipynb`，添加 `HF_TOKEN` Secret，然后 Run All。

训练完成后上传 `kaggle_eval_t4.ipynb`，同样 Run All，产出：

- `eval_final.md` / `eval_final.json`：归一化回报、失败动作、身体位置/姿态、根漂移、关节误差、脚滑、接触一致率、穿地
- `eval_traj.npz`：40 段动作的确定性 rollout
- `eval_video.mp4`：走、跑、转身、舞蹈四段预览视频
- 以上文件自动上传到 Hugging Face 私有仓库 `runs/run_<id>/eval/`

命令行：

```bash
bash scripts/train_blinkmimic_t4.sh <log_dir>
```

完整 40 段指标、覆盖率和失败分析见 [docs/coverage-and-failure-analysis.md](docs/coverage-and-failure-analysis.md)。

## 实测结果（冒烟）

| 项 | 值 |
|---|---|
| 迭代数 | 100 |
| wall-clock | 129 s |
| 吞吐 | 0.93 it/s（compile 关闭） |
| 显存峰值 | 2455 MiB/卡 |
| GPU-hours | 0.072 |
| nonfinite | 0 |

世界模型误差下降，但真实仿真回报在 100 步内下降。该结果只证明管线可运行，不构成策略收敛或覆盖率结论。

## 限制

- 仅仿真
- 冒烟只用 2 段舞蹈动作，正式训练使用 40 段
- 首跑关闭 torch.compile
- 不声称复现论文 H100/4090 的训练时间

## 上游与许可

- 方法：SuperMimic 系列；代码：BlinkMimic（MIT）
- 上游 commit 固定，补丁独立记录
- LAFAN1 动作数据不随仓库分发
