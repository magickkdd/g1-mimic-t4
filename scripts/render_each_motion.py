#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traj", type=Path)
    parser.add_argument("render_py", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--title", default="G1 Mimic T4")
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if CONTROL_HZ % args.fps:
        raise ValueError(f"fps must divide the {CONTROL_HZ} Hz trajectory rate exactly")

    with np.load(args.traj, allow_pickle=False) as archive:
        required = {"names", "clip_len", "root_p", "alive"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"trajectory is missing fields: {sorted(missing)}")
        names = [str(value) for value in archive["names"]]
        clip_len = np.asarray(archive["clip_len"], dtype=np.int64)
        max_steps = int(archive["root_p"].shape[0])
        clip_count = len(names)

    if clip_count == 0 or clip_len.shape != (clip_count,):
        raise ValueError(f"invalid clip metadata: names={clip_count}, clip_len={clip_len.shape}")
    if len(set(names)) != clip_count:
        raise ValueError("motion names are not unique; refusing to overwrite per-motion videos")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.json"
    rows_by_name = {}
    if manifest.exists():
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            rows_by_name = {row["motion"]: row for row in saved.get("videos", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            rows_by_name = {}

    rows = []
    for index, (name, frames) in enumerate(zip(names, clip_len)):
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
            except (subprocess.CalledProcessError, ValueError):
                actual_duration = -1.0
            if 0 < actual_duration and abs(actual_duration - duration) <= 1.0 / args.fps + 0.02:
                row = rows_by_name.get(name, {
                    "index": index,
                    "motion": name,
                    "reference_frames": int(frames),
                    "simulation_steps": steps,
                    "expected_duration_seconds": duration,
                    "video_duration_seconds": actual_duration,
                    "video": output.name,
                    "bytes": output.stat().st_size,
                })
                rows.append(row)
                print(
                    f"[{index + 1:02d}/{clip_count}] reused {name}: "
                    f"{actual_duration:.3f}s",
                    flush=True,
                )
                continue

            output.unlink()

        command = [
            "uv",
            "--directory",
            str(args.render_py.parent.parent),
            "run",
            "python",
            str(args.render_py),
            str(args.traj),
            str(output),
            "--robot",
            args.robot,
            "--clips",
            name,
            "--seconds",
            f"{duration + 1e-6:.8f}",
            "--fps",
            str(args.fps),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--title",
            f"{args.title} | {name}",
        ]
        if args.clean:
            command.append("--clean")

        print(f"[{index + 1:02d}/{clip_count}] rendering {name}", flush=True)
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=args.render_py.parent.parent,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            last_heartbeat = time.monotonic()
            while process.poll() is None:
                time.sleep(5)
                now = time.monotonic()
                if now - last_heartbeat >= 30:
                    print(
                        f"[{index + 1:02d}/{clip_count}] alive {name}: "
                        f"{now - last_heartbeat:.0f}s since last heartbeat",
                        flush=True,
                    )
                    last_heartbeat = now
            return_code = process.returncode
        if return_code != 0 or not output.is_file() or output.stat().st_size == 0:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"render failed for {name} (exit={return_code}):\n{tail}")

        actual_duration = video_duration(output)
        tolerance = 1.0 / args.fps + 0.02
        if abs(actual_duration - duration) > tolerance:
            raise RuntimeError(
                f"duration mismatch for {name}: expected {duration:.3f}s, got {actual_duration:.3f}s"
            )

        rows.append(
            {
                "index": index,
                "motion": name,
                "reference_frames": int(frames),
                "simulation_steps": steps,
                "expected_duration_seconds": duration,
                "video_duration_seconds": actual_duration,
                "video": output.name,
                "bytes": output.stat().st_size,
            }
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
        rows.append(row)
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
            f"{actual_duration:.3f}s, {output.stat().st_size / 1e6:.2f} MB",
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
    if len(list(args.out_dir.glob("*.mp4"))) != clip_count:
        raise RuntimeError("output directory does not contain exactly one MP4 per motion")
    print(f"render_complete={clip_count} manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
