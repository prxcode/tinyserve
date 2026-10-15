"""Compare fp16 and int8 perplexity on the WikiText-2 test set.

Both variants are scored on the same tokens with the same windows, one after the other, so
only the weights differ. The dataset (CC BY-SA) is downloaded into data/, which git ignores.

Usage:
    python bench/perplexity.py --out bench/results/perplexity.json
"""

import argparse
import json
import platform
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from tinyserve.evaluation import perplexity
from tinyserve.model.loader import load_model
from tinyserve.model.quant import weight_bytes
from tinyserve.tokenizer import Tokenizer

DATASET = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TEST_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def load_wikitext(data_dir: Path) -> str:
    path = hf_hub_download(
        DATASET,
        TEST_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=data_dir / "wikitext",
    )
    # Joined the same way as the Hugging Face perplexity guide, so numbers are comparable.
    return "\n\n".join(pq.read_table(path).column("text").to_pylist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=Path("models/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-windows", type=int, help="score only the first N windows")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tokens = Tokenizer.from_model_dir(args.model_dir).encode(load_wikitext(args.data_dir))
    print(f"{len(tokens):,} tokens")

    results = {}
    for variant in ("fp16", "int8"):
        model = load_model(args.model_dir, args.device, quantize=variant == "int8")
        started = time.perf_counter()
        value, scored = perplexity(model, tokens, args.max_length, args.stride, args.max_windows)
        results[variant] = {
            "perplexity": round(value, 4),
            "scored_tokens": scored,
            "weight_bytes": weight_bytes(model),
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(f"{variant}: perplexity {value:.4f} over {scored:,} tokens")
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    fp16, int8 = results["fp16"], results["int8"]
    summary = {
        "perplexity_change": round(int8["perplexity"] - fp16["perplexity"], 4),
        "perplexity_change_pct": round(100 * (int8["perplexity"] / fp16["perplexity"] - 1), 2),
        "weight_memory_reduction_pct": round(
            100 * (1 - int8["weight_bytes"] / fp16["weight_bytes"]), 1
        ),
    }
    print(summary)

    if args.out:
        record = {
            "dataset": {"name": DATASET, "revision": DATASET_REVISION, "file": TEST_FILE},
            "model_dir": args.model_dir.name,
            "max_length": args.max_length,
            "stride": args.stride,
            "max_windows": args.max_windows,
            "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
            "torch": torch.__version__,
            "python": platform.python_version(),
            "results": results,
            "summary": summary,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
