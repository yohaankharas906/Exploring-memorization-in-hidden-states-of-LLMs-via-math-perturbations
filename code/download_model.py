#!/usr/bin/env python3
"""Download a complete Hugging Face model repository for offline use."""

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "Qwen/Qwen2.5-Math-7B-Instruct"
DEFAULT_OUTPUT = ROOT / "models" / "Qwen2.5-Math-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Qwen2.5-Math-7B-Instruct for local/offline use."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git branch, tag, or commit to download (default: main)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with:\n"
            "  python -m pip install -U huggingface_hub"
        ) from exc

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DEFAULT_REPO} to {output_dir}")
    downloaded_path = snapshot_download(
        repo_id=DEFAULT_REPO,
        revision=args.revision,
        local_dir=output_dir,
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Download complete: {downloaded_path}")


if __name__ == "__main__":
    main()
