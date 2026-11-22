# 动作数据清单

## 来源

| 项 | 内容 |
|---|---|
| 数据集 | LAFAN1，Unitree G1 重定向版本 |
| 上游 | Hugging Face `lvhaidong/LAFAN1_Retargeting_Dataset` |
| 许可 | LAFAN1 原数据 CC BY-NC-ND 4.0；G1 重定向与机器人描述来自 Unitree（BSD-3-Clause） |
| 公式格式 | CSV，30 fps，每行 3 位姿 + 4 四元数（xyzw）+ 29 关节 |
| 仓库读取方式 | `data/lafan1_g1/*.csv`，由 `blinkmimic` 内部重采样到 50 Hz |
| 再分发 | 不公开再分发；只放 Kaggle 私有 Dataset |

## 冒烟数据（已验证）

| 文件名 | 大小 |
|---|---|
| `dance1_subject1.csv` | 1317 KB |
| `dance1_subject2.csv` | 1320 KB |

- 共 2 段，13148 帧（50 Hz 后），约 0.073 h
- 用于 2026-11-22 的 100 步双卡冒烟
- 两段舞蹈动作动态较强，仅用于管线验证，不作为覆盖率基线

## 正式数据（待下载/核对）

- 使用仓库 `tools/download_data.sh` 下载 40 段 G1 重定向 LAFAN1（约 2.45 h）
- 覆盖走、跑、转身、蹲起等动作
- 下载后必须核对：
  - 实际文件数与清单
  - 每段帧率与帧数
  - 是否存在穿地、关节超限
- 不把 CSV 上传到 GitHub 或公开 Hugging Face 仓库

## 过滤与重定向

- 现成 G1 重定向 CSV，不自己写重定向
- 可行性过滤在正式评测阶段按 `test_env` 回放和 `evaluate.py` 失败项处理
