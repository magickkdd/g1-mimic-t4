# 40 段动作覆盖率与失败分析

## 评测范围

| 项目 | 结果 |
|---|---|
| 评测数据 | 40 段 LAFAN1 G1 重定向 CSV |
| 评测协议 | 每段动作一次确定性 rollout，`eval_metrics.py` |
| 旧模型 | `run_20261122_083958` |
| 新模型 | `run_20261122_105913` |
| 评测设备 | Kaggle 2 × Tesla T4 16 GB |
| 评测 CPU | 实际检测 4 核，每张卡 2 线程 |

## 覆盖率表

| 模型 | 总动作数 | 成功动作数 | 失败动作数 | 覆盖率 |
|---|---:|---:|---:|---:|
| 旧模型 `083958` | 40 | 40 | 0 | 100% |
| 新模型 `105913` | 40 | 40 | 0 | 100% |

`n_failed=0` 表示评测器没有将任何动作判定为失败。该结论是按评测器的失败判定统计，不等同于每一帧误差为零。

## 汇总指标

| 指标 | 旧模型 | 新模型 | 新模型相对变化 |
|---|---:|---:|---:|
| normalized return | 0.8542 | 0.8762 | +2.6% |
| body position error (cm) | 3.6764 | 3.3260 | -9.5% |
| body orientation error (deg) | 7.9559 | 7.1339 | -10.3% |
| root position error (cm) | 70.0645 | 55.8304 | -20.3% |
| joint error (deg) | 4.6720 | 4.2468 | -9.1% |
| foot slip (cm/s) | 6.8466 | 6.4244 | -6.2% |
| reference foot slip (cm/s) | 10.1322 | 10.1322 | 0.0% |
| contact agreement | 0.7569 | 0.7713 | +1.9% |
| penetration proxy (mm) | 0.0612 | 0.0530 | -13.5% |

## 失败分析

本次评测没有失败动作，因此没有可报告的失败动作清单。新模型相较旧模型的主要改善集中在根位置漂移、身体姿态误差和关节误差；接触一致率也有所提高。脚滑仍然存在，且参考动作自身的脚滑基线为 `10.1322 cm/s`，因此该指标应结合参考值解释，不能简单归因于策略失败。

## 产物状态

指标和确定性轨迹已经生成并上传到 Hugging Face 模型仓库：

```text
runs/run_20261122_083958/eval/eval_final.json
runs/run_20261122_083958/eval/eval_final.md
runs/run_20261122_083958/eval/eval_traj.npz
runs/run_20261122_105913/eval/eval_final.json
runs/run_20261122_105913/eval/eval_final.md
runs/run_20261122_105913/eval/eval_traj.npz
```

此外，旧模型与新模型的完整 40 段合辑均已生成并上传：

```text
runs/run_20261122_083958/eval/eval_video_all40.mp4  3.80 MB
runs/run_20261122_105913/eval/eval_video_all40.mp4  3.78 MB
```

用户提供的上传日志显示两项文件传输和 Hugging Face 校验均达到 100%。4 段预览与完整 40 段合辑是不同产物。

## 证据来源

- Kaggle Notebook：`evals (2).ipynb`
- 旧模型评测：`eval_final.json`，`n_failed=0`
- 新模型评测：`eval_final.json`，`n_failed=0`
- 训练日志：两个 run 的 `train_log.csv`
