#!/usr/bin/env python3
import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from PIL import Image, ImageDraw

CONTROL_HZ = 50


def safe_name(name):
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "motion"


def video_duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def video_decodes(path):
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traj", type=Path)
    parser.add_argument("render_py", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--title", default="G1 Mimic T4")
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if CONTROL_HZ % args.fps:
        raise ValueError(f"fps must divide the {CONTROL_HZ} Hz trajectory rate exactly")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH")
    sys.path.insert(0, str(args.render_py.resolve().parent.parent))
    from blinkmimic.robot import get_robot

    render_spec = importlib.util.spec_from_file_location("blinkmimic_render", args.render_py.resolve())
    if render_spec is None or render_spec.loader is None:
        raise RuntimeError(f"cannot load renderer helpers from {args.render_py}")
    render_module = importlib.util.module_from_spec(render_spec)
    render_spec.loader.exec_module(render_module)
    mat_to_quat = render_module._mat_to_quat
    two_robot_xml = render_module.two_robot_xml

    with np.load(args.traj, allow_pickle=False) as archive:
        required = {"names", "clip_len", "root_p", "root_quat", "q", "alive", "f", "ref_root_p", "ref_root_R", "ref_q"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"trajectory is missing fields: {sorted(missing)}")
        data_np = {key: archive[key] for key in required}
    names = [str(value) for value in data_np["names"]]
    clip_len = np.asarray(data_np["clip_len"], dtype=np.int64)
    max_steps = int(data_np["root_p"].shape[0])
    clip_count = len(names)

    if clip_count == 0 or clip_len.shape != (clip_count,):
        raise ValueError(f"invalid clip metadata: names={clip_count}, clip_len={clip_len.shape}")
    if len(set(names)) != clip_count:
        raise ValueError("motion names are not unique; refusing to overwrite per-motion videos")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for partial in args.out_dir.glob("*.partial.mp4"):
        partial.unlink()
    manifest = args.out_dir / "manifest.json"
    rows_by_name = {}
    if manifest.exists():
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            rows_by_name = {row["motion"]: row for row in saved.get("videos", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            rows_by_name = {}

    robot = get_robot(args.robot)
    nq = 7 + robot.num_joints
    lookat_z = float(np.median(data_np["ref_root_p"][:, 2]))
    model = mujoco.MjModel.from_xml_string(two_robot_xml(robot, args.clean, 0.25, False))
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    mj_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.6, 135.0, -12.0
    rows = []
    for index, (name, frames) in enumerate(zip(names, clip_len)):
        motion_started = time.monotonic()
        steps = int(frames) - 1
        if steps <= 0 or steps > max_steps:
            raise ValueError(f"invalid clip_len for {name}: frames={frames}, trajectory_steps={max_steps}")

        duration = steps / CONTROL_HZ
        output = args.out_dir / f"{index:02d}_{safe_name(name)}.mp4"
        log_path = args.out_dir / f"{index:02d}_{safe_name(name)}.log"
        print(
            f"[{index + 1:02d}/{clip_count}] start {name} "
            f"expected={duration:.3f}s output={output.name}",
            flush=True,
        )

        if output.is_file() and output.stat().st_size > 0:
            try:
                actual_duration = video_duration(output)
                decodes, _ = video_decodes(output)
            except (subprocess.CalledProcessError, ValueError):
                actual_duration = -1.0
                decodes = False
            if decodes and 0 < actual_duration and abs(actual_duration - duration) <= 1.0 / args.fps + 0.02:
                row = {
                    "index": index,
                    "motion": name,
                    "reference_frames": int(frames),
                    "simulation_steps": steps,
                    "expected_duration_seconds": duration,
                    "video_duration_seconds": actual_duration,
                    "video": output.name,
                    "bytes": output.stat().st_size,
                }
                rows.append(row)
                print(
                    f"[{index + 1:02d}/{clip_count}] reused {name}: "
                    f"{actual_duration:.3f}s",
                    flush=True,
                )
                continue

            output.unlink()

        print(f"[{index + 1:02d}/{clip_count}] rendering {name}", flush=True)
        temp_output = output.with_name(output.stem + ".partial.mp4")
        render_started = time.monotonic()
        ff = None
        try:
            ff = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
                 "-an", "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
                 "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(temp_output)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            c = index
            first_end_idx = np.flatnonzero(~data_np["alive"][:, c])
            first_end = int(first_end_idx[0]) if len(first_end_idx) else None
            frame_indices = range(0, steps, CONTROL_HZ // args.fps)
            total_frames = len(frame_indices)
            for frame_no, t in enumerate(frame_indices, start=1):
                ended = first_end is not None and t >= first_end
                tt = first_end if ended else min(t, max_steps - 1)
                f = int(data_np["f"][tt, c])
                mj_data.qpos[0:3] = data_np["root_p"][tt, c]
                mj_data.qpos[3:7] = data_np["root_quat"][tt, c]
                mj_data.qpos[7:nq] = data_np["q"][tt, c]
                mj_data.qpos[nq:nq + 3] = data_np["ref_root_p"][f]
                mj_data.qpos[nq:nq + 2] = data_np["root_p"][tt, c][:2]
                mj_data.qpos[nq + 3:nq + 7] = mat_to_quat(data_np["ref_root_R"][f])
                mj_data.qpos[nq + 7:2 * nq] = data_np["ref_q"][f]
                mujoco.mj_kinematics(model, mj_data)
                camera.lookat[:] = [mj_data.qpos[0], mj_data.qpos[1], lookat_z]
                renderer.update_scene(mj_data, camera=camera)
                image = Image.fromarray(renderer.render())
                if ended and args.clean:
                    draw = ImageDraw.Draw(image)
                    draw.text((16, 16), "terminated", fill=(210, 70, 50))
                ff.stdin.write(np.asarray(image).tobytes())
                if frame_no % max(1, args.fps * 30) == 0 or frame_no == total_frames:
                    print(
                        f"[{index + 1:02d}/{clip_count}] {name}: "
                        f"frames={frame_no}/{total_frames} elapsed={time.monotonic() - render_started:.0f}s",
                        flush=True,
                    )
            ff.stdin.close()
            stderr = ff.stderr.read().decode(errors="replace") if ff.stderr else ""
            return_code = ff.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg exit={return_code}: {stderr[-2000:]}")
            if not temp_output.is_file() or temp_output.stat().st_size == 0:
                raise RuntimeError("ffmpeg produced no output")
            actual_duration = video_duration(temp_output)
            tolerance = 1.0 / args.fps + 0.02
            if abs(actual_duration - duration) > tolerance:
                raise RuntimeError(f"duration mismatch: expected={duration:.3f}s actual={actual_duration:.3f}s")
            decodes, decode_error = video_decodes(temp_output)
            if not decodes:
                raise RuntimeError(f"ffmpeg decode check failed: {decode_error[-2000:]}")
            temp_output.replace(output)
        except Exception as exc:
            if ff is not None and ff.poll() is None:
                ff.kill()
                ff.wait()
            temp_output.unlink(missing_ok=True)
            log_path.write_text(str(exc), encoding="utf-8")
            raise RuntimeError(f"render failed for {name}: {exc}") from exc
        log_path.write_text(
            f"frames={total_frames}\nexpected_duration={duration:.3f}\nactual_duration={actual_duration:.3f}\n",
            encoding="utf-8",
        )
        row = {
            "index": index,
            "motion": name,
            "reference_frames": int(frames),
            "simulation_steps": steps,
            "expected_duration_seconds": duration,
            "video_duration_seconds": actual_duration,
            "video": output.name,
            "bytes": output.stat().st_size,
        }
        rows = [existing for existing in rows if existing["motion"] != name]
        rows.append(row)
        rows.sort(key=lambda existing: existing["index"])
        manifest.write_text(
            json.dumps(
                {
                    "trajectory": str(args.traj),
                    "control_hz": CONTROL_HZ,
                    "video_fps": args.fps,
                    "clip_count": clip_count,
                    "completed_count": len(rows),
                    "videos": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[{index + 1:02d}/{clip_count}] done {name}: "
            f"{actual_duration:.3f}s, render_time={time.monotonic() - render_started:.1f}s, "
            f"{output.stat().st_size / 1e6:.2f} MB",
            flush=True,
        )

    manifest.write_text(
        json.dumps(
            {
                "trajectory": str(args.traj),
                "control_hz": CONTROL_HZ,
                "video_fps": args.fps,
                "clip_count": clip_count,
                "completed_count": len(rows),
                "videos": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs = [path for path in args.out_dir.glob("*.mp4") if not path.name.endswith(".partial.mp4")]
    if {path.name for path in outputs} != {f"{i:02d}_{safe_name(name)}.mp4" for i, name in enumerate(names)}:
        raise RuntimeError("output directory does not contain exactly one expected MP4 per motion")
    print(f"render_complete={clip_count} manifest={manifest}", flush=True)
    renderer.close()


if __name__ == "__main__":
    main()
