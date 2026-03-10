import argparse
from pathlib import Path

from inference_pipeline import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline smoke test")
    parser.add_argument("--image", type=Path, required=True, help="Input grayscale test image")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--full-pipeline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run(args.image, args.output_dir, args.checkpoints, args.full_pipeline)
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"Pipeline test failed, output missing or empty: {output}")
    print(f"Pipeline test passed: {output}")


if __name__ == "__main__":
    main()
