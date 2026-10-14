"""Closed-loop load generator for a running tinyserve server.

Each of `concurrency` clients sends streaming requests back to back until `num_requests`
have completed. Every request has a prompt of about `prompt_words` words and generates
exactly `output_tokens` tokens (ignore_eos), so runs are comparable across commits.

Usage:
    python bench/load_test.py --concurrency 1 8 32 --out bench/results/run.json
"""

import argparse
import asyncio
import json
import math
import platform
import random
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

# A plain word list keeps prompts readable; the exact words do not affect throughput.
WORDS = (  # noqa: SIM905
    "the a model server token cache memory batch request stream river mountain city music "
    "history science ocean forest light energy system network window garden letter story "
    "planet engine market winter summer animal language number question answer"
).split()


@dataclass
class RequestResult:
    prompt_tokens: int
    output_tokens: int
    ttft: float  # seconds from sending the request to the first streamed text
    latency: float  # seconds from sending the request to the final event
    token_gaps: list[float] = field(default_factory=list)  # seconds between streamed chunks


GPU_FIELDS = (
    "name,pstate,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,"
    "power.draw,utilization.gpu,memory.used,memory.total"
)


@dataclass
class RunSummary:
    concurrency: int
    num_requests: int
    duration_s: float
    output_tokens_per_s: float
    requests_per_s: float
    mean_prompt_tokens: float
    mean_output_tokens: float
    ttft_ms: dict[str, float]
    inter_token_ms: dict[str, float]
    latency_ms: dict[str, float]
    gpu_during_run: dict[str, Any] | None


def nvidia_smi(*args: str) -> str | None:
    # Windows installs nvidia-smi in System32, which is not always on PATH for every shell.
    exe = shutil.which("nvidia-smi") or shutil.which(r"C:\Windows\System32\nvidia-smi.exe")
    if exe is None:
        return None
    try:
        return subprocess.run([exe, *args], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_state() -> dict[str, Any] | None:
    """Snapshot of the local GPU, so throttled or shared runs show up in the results."""
    gpus = nvidia_smi(f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits")
    apps = nvidia_smi("--query-compute-apps=pid,process_name", "--format=csv,noheader")
    if gpus is None or apps is None:
        return None
    state: dict[str, Any] = dict(
        zip(
            GPU_FIELDS.split(","), (v.strip() for v in gpus.splitlines()[0].split(",")), strict=True
        )
    )
    state["compute_apps"] = [line.strip() for line in apps.splitlines() if line.strip()]
    return state


def gpu_warnings(state: dict[str, Any] | None) -> list[str]:
    """Reasons a run's numbers cannot be trusted, from a snapshot taken under load."""
    if state is None:
        return []
    warnings = []
    if state["pstate"] != "P0":
        warnings.append(f"GPU in power state {state['pstate']} under load, not P0")
    if float(state["clocks.mem"]) < 0.9 * float(state["clocks.max.mem"]):
        warnings.append(
            f"memory clock {state['clocks.mem']} of {state['clocks.max.mem']} MHz "
            "(on battery or a power-saving plan?)"
        )
    others = [app for app in state["compute_apps"] if "python" not in app.lower()]
    if others:
        warnings.append(f"other processes are using the GPU: {others}")
    return warnings


def make_prompt(rng: random.Random, num_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(num_words))


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def at(q: float) -> float:
        # Nearest-rank percentile: the smallest value with at least q of the data at or below it.
        rank = math.ceil(q * len(ordered))
        return round(ordered[max(rank, 1) - 1] * 1000, 2)

    return {
        "mean": round(statistics.fmean(ordered) * 1000, 2),
        "p50": at(0.50),
        "p90": at(0.90),
        "p99": at(0.99),
    }


async def send(client: httpx.AsyncClient, prompt: str, output_tokens: int) -> RequestResult:
    body = {"prompt": prompt, "max_tokens": output_tokens, "ignore_eos": True, "stream": True}
    start = time.perf_counter()
    first = last = None
    gaps = []
    final: dict[str, Any] = {}
    async with client.stream("POST", "/v1/generate", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line.removeprefix("data: "))
            now = time.perf_counter()
            if "text" in event:
                if first is None:
                    first = now
                else:
                    gaps.append(now - last)  # type: ignore[operator]
                last = now
            if "finish_reason" in event:
                final = event
    end = time.perf_counter()
    if final.get("finish_reason") != "length":
        raise RuntimeError(f"request did not finish normally: {final}")
    return RequestResult(
        prompt_tokens=final["usage"]["prompt_tokens"],
        output_tokens=final["usage"]["completion_tokens"],
        ttft=(first or end) - start,
        latency=end - start,
        token_gaps=gaps,
    )


async def run(
    url: str,
    concurrency: int,
    num_requests: int,
    prompt_words: int,
    output_tokens: int,
    seed: int,
) -> RunSummary:
    rng = random.Random(seed)
    prompts = [make_prompt(rng, prompt_words) for _ in range(num_requests)]
    queue: asyncio.Queue[str] = asyncio.Queue()
    for prompt in prompts:
        queue.put_nowait(prompt)
    results: list[RequestResult] = []
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(base_url=url, timeout=600, limits=limits) as client:

        async def worker() -> None:
            while not queue.empty():
                prompt = queue.get_nowait()
                results.append(await send(client, prompt, output_tokens))

        async def sample_gpu() -> dict[str, Any] | None:
            # Sample once the run is under way; an idle GPU always drops its clocks.
            await asyncio.sleep(2)
            return await asyncio.to_thread(gpu_state)

        start = time.perf_counter()
        sampler = asyncio.create_task(sample_gpu())
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        duration = time.perf_counter() - start
        gpu = await sampler

    total_output = sum(r.output_tokens for r in results)
    return RunSummary(
        concurrency=concurrency,
        num_requests=len(results),
        duration_s=round(duration, 3),
        output_tokens_per_s=round(total_output / duration, 1),
        requests_per_s=round(len(results) / duration, 3),
        mean_prompt_tokens=statistics.fmean(r.prompt_tokens for r in results),
        mean_output_tokens=statistics.fmean(r.output_tokens for r in results),
        ttft_ms=percentiles([r.ttft for r in results]),
        inter_token_ms=percentiles([gap for r in results for gap in r.token_gaps]),
        latency_ms=percentiles([r.latency for r in results]),
        gpu_during_run=gpu,
    )


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--num-requests", type=int, help="per run; default 4x concurrency, min 8")
    parser.add_argument("--prompt-words", type=int, default=100)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="", help="free text stored with the results")
    parser.add_argument("--out", type=Path, help="write results as JSON")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=600) as client:
        health = (await client.get("/health")).json()
        # Warm up CUDA kernels and allocator so the first measured run is not penalised.
        for _ in range(2):
            await send(client, make_prompt(random.Random(-1), args.prompt_words), 16)

    runs = []
    for concurrency in args.concurrency:
        num_requests = args.num_requests or max(8, 4 * concurrency)
        summary = await run(
            args.url, concurrency, num_requests, args.prompt_words, args.output_tokens, args.seed
        )
        runs.append(summary)
        ttft, itl = summary.ttft_ms, summary.inter_token_ms
        print(
            f"concurrency {concurrency:3d}: {summary.output_tokens_per_s:8.1f} tok/s  "
            f"TTFT p50 {ttft['p50']:8.1f} ms  p99 {ttft['p99']:8.1f} ms  "
            f"ITL p50 {itl['p50']:6.1f} ms  p99 {itl['p99']:6.1f} ms"
        )
        for warning in gpu_warnings(summary.gpu_during_run):
            print(f"  WARNING: {warning}; these numbers do not reflect the hardware")

    if args.out:
        record = {
            "label": args.label,
            "commit": git_commit(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "client": {"python": platform.python_version(), "platform": platform.platform()},
            "server_at_start": health,
            "workload": {
                "prompt_words": args.prompt_words,
                "output_tokens": args.output_tokens,
                "seed": args.seed,
            },
            "runs": [asdict(r) for r in runs],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
