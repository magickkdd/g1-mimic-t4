---
license: other
library_name: pytorch
tags:
- robotics
- humanoid-robot
- motion-tracking
- simulation
- unitree-g1
- blinkmimic
---

# BlinkMimic-T4 G1 Motion Tracking Checkpoints

## Model summary

These checkpoints are produced by adapting the BlinkMimic training stack to a dual-NVIDIA-T4 Kaggle environment. The policy tracks retargeted Unitree G1 motions in simulation. This is a reproduction/adaptation artifact, not a new control algorithm.

## Checkpoints

| Run | Training updates | Per-GPU environments | Policy / WM batch | Policy widths | Peak VRAM per GPU |
|---|---:|---:|---:|---|---:|
| `20261122_083958` | 3,500 | 2,048 | 1,024 / 1,024 | 1024-512-256 | 5,879 MiB |
| `20261122_105913` | 3,500 | 4,096 | 3,072 / 2,048 | 2048-1024-512-256 | 12,589 MiB |

Both runs use PyTorch DDP and FP16 with static loss scale 256 on 2 × NVIDIA Tesla T4 GPUs. The higher-capacity run uses `torch.compile`; `compile_more` is disabled.

## Evaluation

Both checkpoints were evaluated with deterministic rollouts on 40 LAFAN1 G1-retargeted motions. The evaluator flagged zero failed clips in both runs.

| Metric | Baseline | Higher-capacity |
|---|---:|---:|
| Normalized return ↑ | 0.8542 | 0.8762 |
| Failed motions ↓ | 0 / 40 | 0 / 40 |
| Body position error (cm) ↓ | 3.6764 | 3.3260 |
| Body orientation error (deg) ↓ | 7.9559 | 7.1339 |
| Root position error (cm) ↓ | 70.0645 | 55.8304 |
| Joint error (deg) ↓ | 4.6720 | 4.2468 |
| Foot slip (cm/s) ↓ | 6.8466 | 6.4244 |
| Contact agreement ↑ | 75.69% | 77.13% |
| Penetration proxy (mm) ↓ | 0.0612 | 0.0530 |

The `eval/` folders contain JSON/Markdown metrics, recorded rollout archives, logs, and four-motion previews. The earlier fixed-duration aggregate video may truncate clips and is not the complete-action deliverable. The repository includes a corrected renderer that creates one MP4 per motion with duration derived from `clip_len`; rerun it to produce and upload the complete per-motion videos.

## Intended use and limitations

- Simulation evaluation only using MuJoCo/mjlab and the pinned BlinkMimic stack.
- No real-robot testing, safety validation, sim-to-real transfer, or deployment claim is provided.
- The policy is conditioned/trained for the motion representation and robot model in the project; generalization outside the 40 evaluated motions is not established.
- These artifacts are not intended for direct control of physical robots.

## Data and rights

The training/evaluation motions are from LAFAN1 and a Unitree G1-retargeted derivative. The project manifest reports CC BY-NC-ND 4.0 for LAFAN1 and BSD-3-Clause for the robot description/retargeting assets. The motion CSV files are not included in this repository. These licenses may restrict commercial use, modification, and redistribution of derivatives. The repository owner has chosen to make the checkpoint repository publicly accessible; this access setting is not a legal determination that the weights are unrestricted or that third-party terms permit every downstream use. Public access to these weights does not grant rights to the source motions or third-party assets. Users must review the upstream data and asset terms before use or redistribution. The applicable licensing status of model weights trained on these data has not been independently cleared; `license: other` is used rather than asserting an unrestricted weights license.

## Sources

- Code: [magickkdd/g1-mimic-t4](https://github.com/magickkdd/g1-mimic-t4)
- Upstream method implementation: [fuyichen0725/blinkmimic](https://github.com/fuyichen0725/blinkmimic), commit `d264658d6c9ce61fb52f9f0f8985a369f3ccc409`
- Motion data manifest: [docs/data-manifest.md](https://github.com/magickkdd/g1-mimic-t4/blob/main/docs/data-manifest.md)
