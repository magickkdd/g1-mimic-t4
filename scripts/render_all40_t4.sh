#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR=${REPO_DIR:-/kaggle/working/proj2-humanoid-mimic/third_party/blinkmimic}
REPO_URL=${REPO_URL:-https://github.com/fuyichen0725/blinkmimic.git}
COMMIT=${COMMIT:-d264658d6c9ce61fb52f9f0f8985a369f3ccc409}
BASE_RESULTS=${BASE_RESULTS:-/kaggle/working/proj2-humanoid-mimic/results/blinkmimic_t4_ddp}
HF_REPO_NAME=${HF_REPO_NAME:-proj2-blinkmimic-t4}
VIDEO_FPS=${VIDEO_FPS:-2}
VIDEO_WIDTH=${VIDEO_WIDTH:-640}
VIDEO_HEIGHT=${VIDEO_HEIGHT:-360}
CPU_COUNT=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')
CPU_THREADS=$((CPU_COUNT / 2))
if [ "${CPU_THREADS}" -lt 1 ]; then CPU_THREADS=1; fi

export REPO_DIR REPO_URL COMMIT BASE_RESULTS HF_REPO_NAME VIDEO_FPS VIDEO_WIDTH VIDEO_HEIGHT CPU_COUNT CPU_THREADS
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PROJECT_DIR

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$CPU_THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$CPU_THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$CPU_THREADS}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$CPU_THREADS}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export UV_LINK_MODE=${UV_LINK_MODE:-copy}

if [ -z "${HF_TOKEN:-}" ]; then
  echo 'HF_TOKEN is required' >&2
  exit 2
fi

mkdir -p "$(dirname "${REPO_DIR}")"
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone --filter=blob:none "${REPO_URL}" "${REPO_DIR}"
fi
git fetch --depth 1 origin "${COMMIT}"
HEAD=$(git -C "${REPO_DIR}" rev-parse HEAD)
if [ "${HEAD}" != "${COMMIT}" ]; then
  git -C "${REPO_DIR}" reset --hard "${COMMIT}"
fi
uv sync --frozen --directory "${REPO_DIR}"

python3 - <<'PY'
import os
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'], check=True)
    from huggingface_hub import HfApi, hf_hub_download

repo_dir = Path(os.environ['REPO_DIR'])
base = Path(os.environ['BASE_RESULTS'])
repo_name = os.environ['HF_REPO_NAME']
token = os.environ['HF_TOKEN']
api = HfApi(token=token)
hf_repo = f"{api.whoami()['name']}/{repo_name}"
runs = {'old_083958': '20261122_083958', 'new_105913': '20261122_105913'}

for name, run_id in runs.items():
    root = base / f'run_{run_id}'
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'eval_traj.npz'
    if not target.exists():
        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename=f'runs/run_{run_id}/eval/eval_traj.npz',
            local_dir=str(root / 'hf_download'),
            token=token,
        )
        target.write_bytes(Path(downloaded).read_bytes())
    print(f'{name}: {target} {target.stat().st_size / 1e6:.1f} MB')

import numpy as np
for name, run_id in runs.items():
    root = base / f'run_{run_id}'
    video_dir = root / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    for partial in video_dir.glob('*.partial.mp4'):
        partial.unlink()
    log = root / 'eval_render_each_motion_stdout.txt'
    command = [
        str(repo_dir / '.venv' / 'bin' / 'python'), str(Path(os.environ['PROJECT_DIR']) / 'scripts' / 'render_each_motion.py'),
        str(root / 'eval_traj.npz'), str(repo_dir / 'tools' / 'render.py'), str(video_dir),
        '--fps', os.environ['VIDEO_FPS'],
        '--width', os.environ['VIDEO_WIDTH'],
        '--height', os.environ['VIDEO_HEIGHT'],
        '--title', f'G1-Mimic-T4 {run_id}',
        '--clean',
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0' if name == 'old_083958' else '1'
    env['MUJOCO_EGL_DEVICE_ID'] = env['CUDA_VISIBLE_DEVICES']
    print('$', ' '.join(command))
    with log.open('w') as handle:
        result = subprocess.run(command, cwd=repo_dir.parent.parent, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT)
    videos = sorted(video_dir.glob('*.mp4'))
    if result.returncode == 0 and (len(videos) != 40 or not (video_dir / 'manifest.json').exists()):
        raise SystemExit(f'{name}: expected 40 complete per-motion MP4s and manifest, found {len(videos)}')
    api.upload_folder(
        folder_path=str(video_dir),
        path_in_repo=f'runs/run_{run_id}/eval/videos',
        repo_id=hf_repo,
        repo_type='model',
        commit_message=f'Upload completed per-motion videos for {run_id}',
    )
    api.upload_file(
        path_or_fileobj=str(log),
        path_in_repo=f'runs/run_{run_id}/eval/{log.name}',
        repo_id=hf_repo,
        repo_type='model',
    )
    print(f'[hf] uploaded {name}/videos ({len(videos)} MP4s plus manifest)')
    if result.returncode != 0:
        raise SystemExit(f'{name} render failed after uploading completed videos; rerun to resume, see {log}')
    print(f'{name}: per_motion_videos={len(videos)}')

print('40_separate_complete_motion_videos_render_upload=PASS')
PY
