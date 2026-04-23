#!/usr/bin/env python3
"""Generate README showcase assets by running examples and collecting outputs to pics/."""

import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PICS_DIR = ROOT / "pics"
TMP_DIR = ROOT / "generated" / "readme-assets"


def run(cmd, description):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT))
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


def find_latest_boids_gif(out_dir):
    validation_dir = Path(out_dir) / "validation"
    if not validation_dir.exists():
        return None
    epochs = sorted(validation_dir.glob("epoch_*"))
    if not epochs:
        return None
    latest_epoch = epochs[-1]
    gifs = list(latest_epoch.glob("validation_seed*_pred.gif"))
    return gifs[0] if gifs else None


def generate_boids_flocking():
    out_dir = str(TMP_DIR / "boids")
    run(
        [
            "uv", "run", "python", "examples/boids-evaluation/main.py",
            "--epochs", "70",
            "--seeds", "5",
            "--eval-seeds", "101",
            "--out-dir", out_dir,
            "--hide-links",
        ],
        "Boids evaluation (70 epochs)",
    )
    final_gif = find_latest_boids_gif(out_dir)
    if final_gif:
        copy_gif(final_gif, "boids_flocking.gif")
    else:
        print("WARNING: No boids GIF found, trying fallback path...")
        fallback = Path(out_dir) / "validation" / "epoch_0070" / "validation_seed101_pred.gif"
        copy_gif(fallback, "boids_flocking.gif")


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
