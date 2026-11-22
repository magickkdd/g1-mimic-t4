#!/usr/bin/env bash
# 双卡 T4：BlinkMimic + DDP + fp16（需要先打 scripts/patches/blinkmimic-t4-ddp-fp16.patch）
# 用法: bash scripts/train_blinkmimic_t4.sh <log_dir> [额外 train.py 参数]
# 注意: num_envs / pi_batch / wm_batch 都是"每卡"值，全局值 = 每卡值 × WORLD_SIZE。
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="${REPO_DIR:-${HERE}/third_party/blinkmimic}"
LOG_DIR="${1:?usage: bash scripts/train_blinkmimic_t4.sh <log_dir> [train.py flags]}"
shift

if ! grep -q "fp16_loss_scale" "${REPO_DIR}/src/blinkmimic/train.py"; then
  echo "缺少 T4 补丁，先执行: git -C ${REPO_DIR} apply ${HERE}/scripts/patches/blinkmimic-t4-ddp-fp16.patch" >&2
  exit 2
fi

cd "${REPO_DIR}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [[ "${DISABLE_COMPILE:-0}" == "1" ]]; then
  export TORCH_COMPILE_DISABLE=1
fi

T4_FLAGS=(
  --amp fp16 --fp16_loss_scale "${FP16_LOSS_SCALE:-256}"
  --motion_dir data/lafan1_g1
  --num_envs "${PER_RANK_ENVS:-4096}"
  --iterations "${ITERATIONS:-3500}"
  --pi_batch "${PER_RANK_PI_BATCH:-2048}"
  --wm_batch "${PER_RANK_WM_BATCH:-2048}"
  --wm_batches 5
  --horizon 32
  --wm_hidden 1024,1024,1024
  --enc_hidden "${POLICY_HIDDEN:-2048,1024,512,256}"
  --dec_hidden "${POLICY_HIDDEN:-2048,1024,512,256}"
  --lr 4e-4
  --pi_lr_mult 4
  --lr_decay_ret 0.75
  --lr_decay_len 300
  --lr_decay_by 1400
  --pi_ema 0.995
  --no_sync
  --fused_adam
  --fused_rot
  --fail_reset 0.3
  --contact_reward 0.25
  --wm_structured
  --wm_mirror 0.5
  --pi_opt muon
  --wm_opt muon
  --pi_ref_skip
  --wm_only_iters 0
  --bin_fix
  --ref_contact_fix
  --ckpt_minutes "${CKPT_MINUTES:-10}"
  --ckpt_every "${CKPT_EVERY:-500}"
  --log_every "${LOG_EVERY:-50}"
  --max_hours "${MAX_HOURS:-8}"
)
if [[ "${COMPILE_MORE:-1}" == "1" ]]; then
  T4_FLAGS+=(--compile_more)
fi

uv run torchrun --standalone --nproc_per_node "${WORLD_SIZE:-2}" -m blinkmimic.train \
  --log_dir "${LOG_DIR}" "${T4_FLAGS[@]}" "$@"
