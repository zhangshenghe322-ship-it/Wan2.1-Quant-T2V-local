#!/usr/bin/env python3
import json, re, sys
from pathlib import Path

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

def try_parse_time(log: str):
    # Try common patterns (you can extend later)
    pats = [
        r"(?:Inference Time|inference time|Total time|total time)\s*[:=]\s*([0-9.]+)\s*(s|sec|seconds)",
        r"(?:Elapsed|elapsed)\s*[:=]\s*([0-9.]+)\s*(s|sec|seconds)",
        r"(?:took)\s*([0-9.]+)\s*(s|sec|seconds)",
    ]
    for pat in pats:
        m = re.search(pat, log, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None

def main():
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if exp_dir is None or not exp_dir.exists():
        print("Usage: python make_summary.py experiments/<your_exp_dir>")
        sys.exit(1)

    cfg = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    stdout = read_text(exp_dir / "stdout.log")
    stderr = read_text(exp_dir / "stderr.log")
    log = stdout + "\n" + stderr

    t = try_parse_time(log)

    gpu = cfg.get("gpu", {}).get("name", "Unknown GPU")
    precision = cfg.get("precision", "N/A")
    res = cfg.get("resolution") or "N/A"
    frames = cfg.get("frames") if cfg.get("frames") is not None else "N/A"
    steps = cfg.get("steps") if cfg.get("steps") is not None else "N/A"

    # Peak VRAM is not available because run_exp didn't finish polling reliably.
    peak_vram = "N/A"

    lines = [
        f"GPU: {gpu}",
        f"Precision: {precision}",
        f"Resolution: {res}",
        f"Frames: {frames}",
        f"Steps: {steps}",
        f"Peak VRAM: {peak_vram}",
        f"Inference Time: {('N/A' if t is None else f'{t:.2f} s')}",
    ]
    (exp_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", exp_dir / "summary.txt")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
