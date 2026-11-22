# G1-Mimic-T4 (`g1-mimic-t4`)

**Multi-motion whole-body tracking for Unitree G1 on dual NVIDIA T4 GPUs.** This project adapts the BlinkMimic training stack to Kaggle's dual-T4 environment with PyTorch DDP and FP16, then evaluates two 3,500-iteration checkpoints on 40 LAFAN1 G1-retargeted motions.

> **Simulation only.** Results are from MuJoCo/mjlab simulation on Kaggle T4 GPUs. No real-robot deployment or sim-to-real transfer is claimed.

## Results

| Evaluation on 40 motions | Baseline | Higher-capacity run |
|---|---:|---:|
| Run | `20261122_083958` | `20261122_105913` |
| Training updates | 3,500 | 3,500 |
| Per-GPU environments | 2,048 | 4,096 |
| Policy / WM batch | 1,024 / 1,024 | 3,072 / 2,048 |
| Policy hidden widths | 1024-512-256 | 2048-1024-512-256 |
| Peak VRAM per T4 | 5,879 MiB | 12,589 MiB |
| Training wall time | 3,980.7 s | 4,347.3 s |
| Normalized return ↑ | 0.8542 | **0.8762** |
| Failed motions ↓ | 0 / 40 | **0 / 40** |
| Body position error (cm) ↓ | 3.6764 | **3.3260** |
| Body orientation error (deg) ↓ | 7.9559 | **7.1339** |
| Root position error (cm) ↓ | 70.0645 | **55.8304** |
| Joint error (deg) ↓ | 4.6720 | **4.2468** |
| Foot slip (cm/s) ↓ | 6.8466 | **6.4244** |
| Contact agreement ↑ | 75.69% | **77.13%** |
| Penetration proxy (mm) ↓ | 0.0612 | **0.0530** |

Both checkpoints completed deterministic evaluation on all 40 motions with zero evaluator-flagged failures. Metrics are aggregate tracking results; `0/40` does not mean zero frame-level error. See [coverage and failure analysis](docs/coverage-and-failure-analysis.md) and [training/evaluation cost report](docs/cost-report.md).

## Model

Checkpoints and evaluation artifacts are organized by run in the Hugging Face model repository:

**[A2bao/proj2-blinkmimic-t4](https://huggingface.co/A2bao/proj2-blinkmimic-t4)** · [Model card](model-card.md)

This model repository is public and browsable anonymously. It contains both `last.pt` checkpoints and the full evaluation artifacts for each run.

| Run | Checkpoint | Evaluation artifacts |
|---|---|---|
| `20261122_083958` | `runs/run_20261122_083958/last.pt` | [`eval/`](https://huggingface.co/A2bao/proj2-blinkmimic-t4/tree/main/runs/run_20261122_083958/eval) |
| `20261122_105913` | `runs/run_20261122_105913/last.pt` | [`eval/`](https://huggingface.co/A2bao/proj2-blinkmimic-t4/tree/main/runs/run_20261122_105913/eval) |

Each evaluation directory contains metric JSON/Markdown, the deterministic rollout archive, logs, a four-motion preview video, and a full 40-motion aggregate video (`eval_video_all40.mp4`). The two aggregate videos were uploaded successfully; the user-provided upload logs show 100% transfer and validation for both files (3.80 MB and 3.78 MB).

## Method and hardware

- Robot: Unitree G1, 29 actuated joints
- Simulation: mjlab, MuJoCo, MuJoCo Warp
- Training code: [BlinkMimic](https://github.com/fuyichen0725/blinkmimic), pinned at `d264658d6c9ce61fb52f9f0f8985a369f3ccc409`
- Hardware: Kaggle, 2 × NVIDIA Tesla T4 (15,360 MiB reported memory per GPU)
- Parallelism and precision: PyTorch DDP, FP16, static loss scale 256
- Motion set: 40 LAFAN1 G1-retargeted CSV motions, evaluated deterministically
- Higher-capacity run used `torch.compile`; `compile_more` was disabled

This is a reproduction/adaptation of the BlinkMimic stack for the stated hardware, not a new motion-control algorithm and not a claim of reproducing paper-reported H100/RTX 4090 throughput.

## Reproduce

1. Open [`notebooks/kaggle_blinkmimic_t4_ddp.ipynb`](notebooks/kaggle_blinkmimic_t4_ddp.ipynb) in Kaggle with 2 × T4 GPUs.
2. Add the Hugging Face token as a Kaggle Secret named `HF_TOKEN`.
3. Run the training notebook; it pins the upstream commit, applies the T4 DDP/FP16 patch, and backs up checkpoints and logs.
4. Run [`notebooks/kaggle_eval_t4_max_compare.ipynb`](notebooks/kaggle_eval_t4_max_compare.ipynb) to compute the 40-motion metrics, save trajectories, render previews, and upload run artifacts.

The shell entry points are [`scripts/train_blinkmimic_t4.sh`](scripts/train_blinkmimic_t4.sh) and [`scripts/render_all40_t4.sh`](scripts/render_all40_t4.sh). The latter renders all 40 clips from an existing trajectory archive without rerunning policy evaluation.

## Repository layout

```text
configs/                         Kaggle and formal-run configuration
docs/                            environment, data, results, costs, and failure analysis
notebooks/                       current training/evaluation notebooks; archive/ stores superseded runs
scripts/                         launch, patch, and rendering scripts
third_party/                     local upstream checkout (not committed)
data/                            local motion data (not committed)
results/                         local checkpoints and logs (not committed)
```

Key documents:

- [Coverage and failure analysis](docs/coverage-and-failure-analysis.md)
- [Training and evaluation cost](docs/cost-report.md)
- [Environment lock](docs/env-lock.md)
- [Motion data manifest and license](docs/data-manifest.md)
- [Formal run and backup procedure](docs/formal-run.md)

## Data, license, and scope

The motion source is the LAFAN1 dataset and its G1-retargeted derivative. The project manifest records the source and the reported CC BY-NC-ND 4.0 / BSD-3-Clause terms. Motion CSVs are intentionally excluded from this repository and must not be redistributed without confirming the applicable rights. Checkpoint publication does not grant rights to redistribute source motions or derivative data; review the dataset terms and checkpoint contents before mirroring or redistributing model weights.

Project-authored code is provided under the repository license. The BlinkMimic-derived adaptation patch and upstream files retain their applicable upstream license terms. BlinkMimic is MIT-licensed at the pinned revision. See [LICENSE](LICENSE) and [docs/data-manifest.md](docs/data-manifest.md).
