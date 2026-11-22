#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/kaggle/working/proj2-humanoid-mimic}"
UPSTREAM_DIR="${UPSTREAM_DIR:-${PROJECT_ROOT}/third_party/SuperMimic-Reproduced}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/CuboiLeo/SuperMimic-Reproduced.git}"
UPSTREAM_COMMIT="${UPSTREAM_COMMIT:-3602aed6a397b3202ac052d688d47300f443a3ea}"
DATA_DIR="${DATA_DIR:-}"
NUM_ENVS="${NUM_ENVS:-1}"
POLICY_BATCH="${POLICY_BATCH:-256}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-100}"
RUN_NAME="${RUN_NAME:-kaggle_smoke}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/results/${RUN_NAME}}"

mkdir -p "${PROJECT_ROOT}/third_party" "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/smoke.log") 2>&1

printf 'project_root=%s\n' "${PROJECT_ROOT}"
printf 'upstream_url=%s\n' "${UPSTREAM_URL}"
printf 'upstream_commit=%s\n' "${UPSTREAM_COMMIT}"
printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
printf 'gpu_count=%s\n' "${GPU_COUNT}"

if [[ "${GPU_COUNT}" -lt 1 ]]; then
  printf 'status=FAIL_NO_GPU\n'
  exit 2
fi

if [[ "${REQUIRE_TWO_GPUS:-1}" == "1" && "${GPU_COUNT}" -ne 2 ]]; then
  printf 'status=FAIL_EXPECTED_TWO_GPUS\n'
  exit 2
fi

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  git clone --filter=blob:none "${UPSTREAM_URL}" "${UPSTREAM_DIR}"
fi
git -C "${UPSTREAM_DIR}" fetch --depth 1 origin "${UPSTREAM_COMMIT}"
git -C "${UPSTREAM_DIR}" checkout --detach "${UPSTREAM_COMMIT}"
printf 'git_head=%s\n' "$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"

cd "${UPSTREAM_DIR}"
python -V
python -m pip install --upgrade pip
python -m pip install uv
uv sync --frozen --no-dev

uv run python - <<'PY'
import importlib.metadata as metadata
import sys
import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang"):
    try:
        print(f"{name}={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}=NOT_FOUND")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY

uv run python -m supermimic.scripts.train --help > "${LOG_DIR}/train_help.txt"
printf 'train_help=PASS\n'

if [[ -z "${DATA_DIR}" ]]; then
  printf 'status=PARTIAL_NO_DATA\n'
  printf 'next=Set DATA_DIR to a licensed directory containing one or two retargeted G1 npz files and rerun.\n'
  exit 0
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  printf 'status=FAIL_DATA_DIR_NOT_FOUND\n'
  exit 3
fi

mapfile -t NPZ_FILES < <(find "${DATA_DIR}" -maxdepth 1 -type f -name '*.npz' | sort | head -n 2)
if [[ "${#NPZ_FILES[@]}" -eq 0 ]]; then
  printf 'status=FAIL_NO_NPZ\n'
  exit 4
fi

SMOKE_DATA_DIR="${PROJECT_ROOT}/data/smoke_npz"
mkdir -p "${SMOKE_DATA_DIR}"
for file in "${NPZ_FILES[@]}"; do
  ln -sf "${file}" "${SMOKE_DATA_DIR}/$(basename "${file}")"
done

export PYTHONPATH="${UPSTREAM_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_LOG="${LOG_DIR}/gpu_samples.csv"
printf 'timestamp,gpu_index,memory_used_mib,utilization_pct\n' > "${GPU_LOG}"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits >> "${GPU_LOG}" || true
    sleep "${GPU_SAMPLE_SECONDS:-2}"
  done
) &
GPU_MONITOR_PID=$!
trap 'kill "${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT

set +e
timeout "${SMOKE_TIMEOUT_SECONDS:-1800}" uv run sm-train \
  --run-name "${RUN_NAME}" \
  --log-root "${LOG_DIR}/logs" \
  --data.npz-dir "${SMOKE_DATA_DIR}" \
  --data.limit "${#NPZ_FILES[@]}" \
  --env.num-envs "${NUM_ENVS}" \
  --policy-trainer.batch-size "${POLICY_BATCH}" \
  --schedule.total-iterations "${TOTAL_ITERATIONS}" \
  --schedule.warmup-iterations 40 \
  --schedule.log-interval 10 \
  --schedule.checkpoint-interval 100 \
  --schedule.max-hours 0.5
TRAIN_STATUS=$?
set -e

if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  printf 'status=FAIL_TRAIN_EXIT_%s\n' "${TRAIN_STATUS}"
fi

awk -F', ' '
  NR > 1 {
    used[$2] = ($3 > used[$2] ? $3 : used[$2])
  }
  END {
    for (gpu in used) {
      printf "gpu_%s_peak_memory_mib=%s\n", gpu, used[gpu]
    }
  }
' "${GPU_LOG}" | sort -V

if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  exit "${TRAIN_STATUS}"
fi
printf 'status=PASS_100_STEP_SMOKE\n'
printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
