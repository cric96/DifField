#!/usr/bin/env python3
"""Generate README showcase assets by running examples and collecting outputs to pics/."""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PICS_DIR = ROOT / "pics"
TMP_DIR = ROOT / "generated" / "readme-assets"


def run(cmd, description):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)  # noqa: S603
    if result.returncode != 0:
        print(f"WARNING: {description} failed with return code {result.returncode}")
    return result.returncode == 0


def copy_gif(src, dst_name):
    dst = PICS_DIR / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_path = Path(src)
    if src_path.exists():
        shutil.copy2(src_path, dst)
        print(f"Copied {src_path} -> {dst}")
        return True
    else:
        print(f"WARNING: Source file not found: {src}")
        return False


def generate_boids_flocking():
    out_dir = TMP_DIR / "boids"
    if not run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "examples.seams",
            "boids",
            "--profile",
            "paper-cpu",
            "--out",
            str(out_dir),
        ],
        "Boids observed learning and GNN comparison",
    ):
        return
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(out_dir / "videos/observed.mp4"),
            "-t",
            "6",
            "-filter_complex",
            "fps=8,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
            "-loop",
            "0",
            str(PICS_DIR / "boids_flocking.gif"),
        ],
        "Boids README animation",
    )


def generate_spatial_channel():
    prefix = str(TMP_DIR / "spatial_channel")
    run(
        [
            "uv", "run", "python", "examples/channel/spatial.py",
            "--num-nodes", "10000",
            "--rounds", "2000",
            "--seed", "1",
            "--viz-prefix", prefix,
        ],
        "Spatial channel (10000 nodes)",
    )
    copy_gif(f"{prefix}.gif", "spatial_channel.gif")


def generate_gradient_large():
    prefix = str(TMP_DIR / "gradient_large")
    run(
        [
            "uv", "run", "python", "examples/gradients/large.py",
            "--rows", "500",
            "--cols", "500",
            "--seed", "7",
            "--viz-prefix", prefix,
        ],
        "Gradient large (500x500)",
    )
    copy_gif(f"{prefix}_evolution.gif", "gradient_large.gif")


def generate_channel_small():
    prefix = str(TMP_DIR / "channel_small")
    run(
        [
            "uv", "run", "python", "examples/channel/small.py",
            "--rows", "15",
            "--cols", "15",
            "--rounds", "100",
            "--seed", "42",
            "--viz-prefix", prefix,
        ],
        "Channel small (15x15)",
    )
    copy_gif(f"{prefix}_evolution.gif", "channel_small.gif")


def generate_gradient_small():
    prefix = str(TMP_DIR / "gradient_small")
    run(
        [
            "uv", "run", "python", "examples/gradients/fixed.py",
            "--rows", "15",
            "--cols", "15",
            "--seed", "7",
            "--viz-prefix", prefix,
        ],
        "Gradient small (15x15)",
    )
    copy_gif(f"{prefix}_evolution.gif", "gradient_small.gif")


def main():
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating README showcase assets...")
    print(f"Output directory: {PICS_DIR}")

    generate_boids_flocking()
    generate_spatial_channel()
    generate_gradient_large()
    generate_channel_small()
    generate_gradient_small()

    print(f"\n{'='*60}")
    print("Done! Generated files in pics/:")
    for f in sorted(PICS_DIR.glob("*.gif")):
        print(f"  {f.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
