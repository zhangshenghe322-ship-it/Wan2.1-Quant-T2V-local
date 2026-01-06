#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_exp.py

What this script does (summary):
- Runs a given command (typically `python generate.py ...`) and logs stdout/stderr.
- Records experiment config (git info, system, GPU, etc.) into experiments/<timestamp>_<name>/config.json
- Tracks peak VRAM via nvidia-smi polling during the whole subprocess lifetime.
- NEW (added): records a "steady-state" VRAM probe after the subprocess starts, by waiting
  VRAM_PROBE_S seconds (default 8s) and sampling nvidia-smi once. This helps capture
  model-load / resident footprint differences between FP16 and W8A16 even when peak VRAM
  is activation-dominated.
"""

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: str):
    p = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def get_git_info():
    info = {"commit": None, "branch": None, "dirty": None}
    rc, out, _ = run_cmd("git rev-parse HEAD")
    if rc == 0:
        info["commit"] = out
    rc, out, _ = run_cmd("git rev-parse --abbrev-ref HEAD")
    if rc == 0:
        info["branch"] = out
    rc, out, _ = run_cmd("git status --porcelain")
    if rc == 0:
        info["dirty"] = "yes" if out else "no"
    return info


def get_gpu_name(gpu: int):
    rc, out, _ = run_cmd(
        f"nvidia-smi --query-gpu=name --format=csv,noheader -i {gpu}"
    )
    return out if rc == 0 and out else "Unknown GPU"


def get_driver_cuda():
    rc, out, _ = run_cmd("nvidia-smi")
    driver, cuda = None, None
    if rc == 0 and out:
        m1 = re.search(r"Driver Version:\s*([0-9.]+)", out)
        m2 = re.search(r"CUDA Version:\s*([0-9.]+)", out)
        if m1:
            driver = m1.group(1)
        if m2:
            cuda = m2.group(1)
    return driver, cuda


def query_vram_used_mb(gpu: int):
    """
    Returns current GPU memory.used in MB (int) or None if unavailable.
    """
    rc, out, _ = run_cmd(
        f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu}"
    )
    if rc == 0 and out:
        out = out.strip()
        if out.isdigit():
            return int(out)
        # Sometimes output can be like "13561 MiB" if format flags changed.
        m = re.search(r"(\d+)", out)
        if m:
            return int(m.group(1))
    return None


def poll_peak_vram_mb(gpu: int, proc: subprocess.Popen, interval: float):
    """
    Polls nvidia-smi memory.used while proc is running and returns peak MB.
    """
    peak = 0
    while proc.poll() is None:  # robust: stop when process ends
        cur = query_vram_used_mb(gpu)
        if cur is not None:
            peak = max(peak, cur)
        time.sleep(interval)

    # One last sample after exit
    cur = query_vram_used_mb(gpu)
    if cur is not None:
        peak = max(peak, cur)
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--precision", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--resolution", default="")
    ap.add_argument("--frames", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=-1)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("experiments") / f"{ts}_{args.name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = get_gpu_name(args.gpu)
    driver, cuda = get_driver_cuda()

    config = {
        "name": args.name,
        "timestamp": ts,
        "precision": args.precision,
        "prompt": args.prompt,
        "seed": None if args.seed == -1 else args.seed,
        "resolution": args.resolution or None,
        "frames": None if args.frames == -1 else args.frames,
        "steps": None if args.steps == -1 else args.steps,
        "cmd": args.cmd,
        "system": {
            "platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "gpu": {"name": gpu_name, "driver": driver, "cuda": cuda},
        "git": get_git_info(),
    }
    (exp_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    stdout_path = exp_dir / "stdout.log"
    stderr_path = exp_dir / "stderr.log"

    # NEW: steady-state probe window (seconds) after starting subprocess.
    # Use env var VRAM_PROBE_S to override. Default 8 seconds.
    try:
        probe_s = float(os.environ.get("VRAM_PROBE_S", "8"))
    except ValueError:
        probe_s = 8.0

    steady_mb = None
    steady_ts = None

    with open(stdout_path, "w", encoding="utf-8") as f_out, open(
        stderr_path, "w", encoding="utf-8"
    ) as f_err:
        start = time.time()
        proc = subprocess.Popen(
            args.cmd,
            shell=True,
            stdout=f_out,
            stderr=f_err,
            env=os.environ.copy(),
        )

        # ---- NEW: steady VRAM probe (captures resident footprint after startup) ----
        # Rationale: peak VRAM in diffusion is often activation/workspace-dominated and may not
        # change under weight-only quantization. Steady footprint often does.
        if probe_s > 0:
            time.sleep(probe_s)
            steady_mb = query_vram_used_mb(args.gpu)
            steady_ts = round(time.time() - start, 2)

        peak_mb = poll_peak_vram_mb(args.gpu, proc, args.interval)
        ret = proc.wait()
        end = time.time()

    metrics = {
        "return_code": ret,
        "steady_probe_s": probe_s,
        "steady_vram_mb": steady_mb,
        "steady_vram_gb": round(steady_mb / 1024.0, 2) if steady_mb is not None else None,
        "steady_sample_t_s": steady_ts,  # seconds after subprocess start when sampled
        "peak_vram_mb": peak_mb,
        "peak_vram_gb": round(peak_mb / 1024.0, 2),
        "inference_time_s": round(end - start, 2),
        "exp_dir": str(exp_dir),
    }
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_lines = [
        f"GPU: {gpu_name}",
        f"Precision: {args.precision}",
        f"Resolution: {args.resolution if args.resolution else 'N/A'}",
        f"Frames: {args.frames if args.frames != -1 else 'N/A'}",
        f"Steps: {args.steps if args.steps != -1 else 'N/A'}",
        f"Steady VRAM (probe @ {metrics['steady_sample_t_s']}s): "
        f"{metrics['steady_vram_gb']} GB" if metrics["steady_vram_gb"] is not None else
        "Steady VRAM: N/A",
        f"Peak VRAM: {metrics['peak_vram_gb']} GB",
        f"Inference Time: {metrics['inference_time_s']} s",
        f"Return Code: {ret}",
    ]
    (exp_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[OK] Saved to {exp_dir}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
