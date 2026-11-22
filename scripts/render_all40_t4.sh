#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR=${REPO_DIR:-/kaggle/working/proj2-humanoid-mimic/third_party/blinkmimic}
REPO_URL=${REPO_URL:-https://github.com/fuyichen0725/blinkmimic.git}
COMMIT=${COMMIT:-d264658d6c9ce61fb52f9f0f8985a369f3ccc409}
BASE_RESULTS=${BASE_RESULTS:-/kaggle/working/proj2-humanoid-mimic/results/blinkmimic_t4_ddp}
HF_REPO_NAME=${HF_REPO_NAME:-proj2-blinkmimic-t4}
CLIP_SECONDS=${CLIP_SECONDS:-3}
VIDEO_FPS=${VIDEO_FPS:-15}
VIDEO_WIDTH=${VIDEO_WIDTH:-640}
VIDEO_HEIGHT=${VIDEO_HEIGHT:-360}
CPU_COUNT=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')
CPU_THREADS=$((CPU_COUNT / 2))
if [ "${CPU_THREADS}" -lt 1 ]; then CPU_THREADS=1; fi

export REPO_DIR REPO_URL COMMIT BASE_RESULTS HF_REPO_NAME CLIP_SECONDS VIDEO_FPS VIDEO_WIDTH VIDEO_HEIGHT CPU_COUNT CPU_THREADS

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
traj = np.load(str(base / 'run_20261122_105913' / 'eval_traj.npz'))
clips = [str(x) for x in traj['names']]
if len(clips) != 40:
    raise RuntimeError(f'expected 40 clips, found {len(clips)}')
clip_file = base / 'full40_clip_manifest.txt'
clip_file.write_text('\n'.join(clips) + '\n')
os.environ['FULL40_CLIPS'] = '\n'.join(clips)
print(f'full_clip_count={len(clips)}')

for name, run_id in runs.items():
    root = base / f'run_{run_id}'
    video = root / 'eval_video_all40.mp4'
    command = [
        'uv', 'run', 'python', 'tools/render.py',
        str(root / 'eval_traj.npz'), str(video),
        '--clips', *clips,
        '--seconds', os.environ['CLIP_SECONDS'],
        '--fps', os.environ['VIDEO_FPS'],
        '--width', os.environ['VIDEO_WIDTH'],
        '--height', os.environ['VIDEO_HEIGHT'],
        '--title', f'BlinkMimic-T4 {run_id} all 40 clips',
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0' if name == 'old_083958' else '1'
    log = root / 'eval_render_all40_stdout.txt'
    print('$', ' '.join(command))
    with log.open('w') as handle:
        result = subprocess.run(command, cwd=repo_dir, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f'{name} render failed; see {log}')
    print(f'{name}: video={video} size={video.stat().st_size / 1e6:.1f} MB')
    api.upload_file(
        path_or_fileobj=str(video),
        path_in_repo=f'runs/run_{run_id}/eval/{video.name}',
        repo_id=hf_repo,
        repo_type='model',
    )
    api.upload_file(
        path_or_fileobj=str(log),
        path_in_repo=f'runs/run_{run_id}/eval/{log.name}',
        repo_id=hf_repo,
        repo_type='model',
    )
    print(f'[hf] uploaded {name}/{video.name}')

print('full40_render_upload=PASS')
PY
