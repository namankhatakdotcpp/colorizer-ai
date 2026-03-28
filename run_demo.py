#!/usr/bin/env python3
"""
Colorizer-AI Demo Script
Usage: python run_demo.py test.jpg
       python run_demo.py test.jpg --full   # includes SR if available
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def main():
    p = argparse.ArgumentParser(description="Colorizer-AI demo")
    p.add_argument("image", nargs="?", default="test.jpg")
    p.add_argument("--full", action="store_true",
                   help="Run full pipeline including SR (experimental)")
    p.add_argument("--checkpoints", default="checkpoints/")
    p.add_argument("--output-dir",  default="outputs/")
    args = p.parse_args()

    if not Path(args.image).exists():
        print(f"Error: '{args.image}' not found")
        sys.exit(1)

    print("═" * 50)
    print("  Colorizer-AI — Computational Photography")
    print("═" * 50)
    print(f"  Input : {args.image}")
    print(f"  Output: {args.output_dir}")
    print()

    if args.full:
        stages = "colorizer sr depth bokeh"
        print("  Mode: Full pipeline (colorizer + SR + depth + bokeh)")
    else:
        stages = "colorizer"
        print("  Mode: Colorizer only (recommended — best quality)")

    print()
    cmd = (
        f"python inference_pipeline.py {args.image} "
        f"--stages {stages} "
        f"--checkpoints {args.checkpoints} "
        f"--output-dir {args.output_dir}"
    )
    ret = run(cmd)

    if ret == 0:
        print()
        print("═" * 50)
        print("  Done!")
        print(f"  Result saved to: {args.output_dir}")
        print("═" * 50)
    else:
        print(f"Error: pipeline exited with code {ret}")
        sys.exit(ret)


if __name__ == "__main__":
    main()
