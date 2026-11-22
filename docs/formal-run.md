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

### high_util（提利用率）

```text
每卡 num_envs=4096, pi_batch=2048, wm_batch=2048
policy 宽度 2048-1024-512-256
iterations=3500, amp=fp16, ckpt_every=500
torch.compile + compile_more 开启
```

全局等价：8192 环境、4096 policy batch、4096 world-model batch。

### DDP 通信优化

- 梯度按模块打平成一次 all-reduce，不再逐 tensor 调用
- norm/EMA buffer 按 dtype 打平广播
- stop flag 每 10 个 iteration 检查一次，减少 host sync

## 3. 启动

```bash
bash scripts/train_blinkmimic_t4.sh <log_dir>
```

或在 Kaggle 里直接 Run All `kaggle_blinkmimic_t4_ddp.ipynb`。

## 4. checkpoint

- 完整 checkpoint：`ckpt_<iteration>.pt` 和 `last.pt`
- 包含：world model、policy、critic、三个优化器、EMA、reset 统计
- 保存位置：`/kaggle/working/proj2-humanoid-mimic/results/blinkmimic_t4_ddp/run_<时间戳>/`
- 每 500 iteration 或 10 分钟保存一次

## 5. 备份

- Notebook 内置后台线程，把新产生的 checkpoint 和日志上传到 Hugging Face 私有 model 仓库
- 仓库名：`<hf_user>/proj2-blinkmimic-t4`，首次运行自动创建
- token 从 Kaggle Secrets 读取，不写进代码
- Kaggle 输出同时保留一份，作为第二备份

## 6. 断点续训

1. 在 Notebook 配置里设置 `RESUME_RUN_ID` 为要恢复的时间戳
2. Notebook 从 Hugging Face 下载该 run 的 `last.pt`
3. torchrun 启动时附加 `--resume <last.pt>`
4. 日志和 checkpoint 继续写入同一 run 目录

## 7. 已知限制

- 仅仿真，不上真机
- 首跑关闭 torch.compile，吞吐不是最终性能口径
- 40 段数据需先确认许可证和本地可用性
