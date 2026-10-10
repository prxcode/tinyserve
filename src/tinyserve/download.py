"""Download model files from the Hugging Face Hub.

Usage: python -m tinyserve.download [--repo REPO] [--revision SHA] [--dest DIR]
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Pinned to a commit so tests and benchmarks always run against the same weights.
DEFAULT_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"

NEEDED_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
]


def download(repo: str, revision: str, dest: Path) -> Path:
    path = snapshot_download(
        repo_id=repo, revision=revision, local_dir=dest, allow_patterns=NEEDED_FILES
    )
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--dest", type=Path, help="defaults to models/<repo name>")
    args = parser.parse_args()

    dest = args.dest or Path("models") / args.repo.split("/")[-1]
    print(f"downloaded to {download(args.repo, args.revision, dest)}")


if __name__ == "__main__":
    main()
