# 正式训练与备份流程

## 1. 数据

- `DATA_MODE='full40'`：运行 `tools/download_data.sh` 获取 40 段 LAFAN1 G1 重定向
- `DATA_MODE='smoke2'`：只用两段 dance 做管线检查
- 数据只放 Kaggle 私有 Dataset，不提交 GitHub，不上传公开 Hugging Face

## 2. 正式配置

### baseline（已验证）

```text
每卡 num_envs=2048, pi_batch=1024, wm_batch=1024
policy 宽度 1024-512-256
iterations=3500, amp=fp16, ckpt_every=500
torch.compile 关闭
```

### high_util（历史候选配置）

```text
每卡 num_envs=4096, pi_batch=2048, wm_batch=2048
policy 宽度 2048-1024-512-256
torch.compile + compile_more 开启
```

### stable_high（已完成正式高利用率运行）

```text
每卡 num_envs=4096, pi_batch=3072, wm_batch=2048
policy 宽度 2048-1024-512-256
FP16，loss scale 256，iterations=3500
torch.compile 开启，compile_more 关闭
```

实际训练每卡峰值显存 12589 / 15360 MiB；Kaggle Notebook 的 CPU 检测结果为 4 核。

### max（未执行）

```text
每卡 num_envs=8192, pi_batch=3072, wm_batch=2048
policy 宽度 2048-1024-512-256
torch.compile + compile_more 开启
```

max 配置没有作为正式训练结果，不应与已完成的 `stable_high` run 混写。

降档顺序：pi_batch 3072→2048；仍 OOM 则 envs 8192→4096；编译报错则关闭 compile。

### DDP 通信优化

- 梯度按模块打平成一次 all-reduce，不再逐 tensor 调用
- norm/EMA buffer 按 dtype 打平广播
- stop flag 每 10 个 iteration 检查一次，减少 host sync

## 3. 启动

```bash
bash scripts/train_blinkmimic_t4.sh <log_dir>
```

或在 Kaggle 里直接 Run All `notebooks/kaggle_blinkmimic_t4_ddp.ipynb`。

## 4. checkpoint

- 完整 checkpoint：`ckpt_<iteration>.pt` 和 `last.pt`
- 包含：world model、policy、critic、三个优化器、EMA、reset 统计
- 保存位置：`/kaggle/working/proj2-humanoid-mimic/results/blinkmimic_t4_ddp/run_<时间戳>/`
- 每 500 iteration 或 10 分钟保存一次

## 5. 备份

- Notebook 内置后台线程，把新产生的 checkpoint 和日志上传到 Hugging Face model 仓库
- 仓库名：`<hf_user>/proj2-blinkmimic-t4`，首次运行自动创建
- 可见性由仓库设置决定；公开发布前必须检查数据集和 checkpoint 的再分发许可
- token 从 Kaggle Secrets 读取，不写进代码
- Kaggle 输出同时保留一份，作为第二备份

## 6. 断点续训

1. 在 Notebook 配置里设置 `RESUME_RUN_ID` 为要恢复的时间戳
2. Notebook 从 Hugging Face 下载该 run 的 `last.pt`
3. torchrun 启动时附加 `--resume <last.pt>`
4. 日志和 checkpoint 继续写入同一 run 目录

## 7. 已知限制

- 仅仿真，不上真机
- 项目结果仅为仿真验证，不包含真机结果
- 40 段动作指标已完成；预览视频为 4 段，完整 40 段合辑需单独渲染
