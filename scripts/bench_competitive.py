#!/usr/bin/env python3
"""Hayabusa vs vllm-mlx competitive benchmark.

Both servers must be running before starting:
  # Hayabusa (port 8080):
  .build/debug/Hayabusa mlx-community/Qwen3.5-9B-MLX-4bit --backend mlx
  # vllm-mlx (port 9090):
  vllm-mlx serve mlx-community/Qwen3.5-9B-MLX-4bit --port 9090 --continuous-batching

Usage:
  python scripts/bench_competitive.py --samples 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

# ── Config ──────────────────────────────────────────────────────────

MODEL_ID = "mlx-community/Qwen3.5-9B-MLX-4bit"
MAX_TOKENS = 128
TEMPERATURE = 0
CONCURRENCIES = [1, 4, 8, 16]

PROMPTS = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "user", "content": "Say hello in Japanese."},
    {"role": "user", "content": "What color is the sky?"},
    {"role": "user", "content": "Explain the difference between a list and a tuple in Python in 2 sentences."},
    {"role": "user", "content": "Write a one-line Python function that reverses a string."},
    {"role": "user", "content": "What are the three pillars of object-oriented programming?"},
    {"role": "user", "content": "Explain how a hash table works. Cover: the hash function, collision resolution, and time complexity for insert/lookup. Keep it under 100 words."},
    {"role": "user", "content": "Compare merge sort and quicksort. Discuss their time complexity, space usage, and stability. Answer in 3-4 sentences."},
    {"role": "user", "content": "What is the CAP theorem in distributed systems? Give a brief example for each of the three trade-offs."},
    {"role": "user", "content": "Describe how garbage collection works in Java. Mention generational GC, mark-and-sweep, and when a full GC is triggered."},
]

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "bench_competitive.json"

# ── Data classes ────────────────────────────────────────────────────

@dataclass
class RequestResult:
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error: str | None = None


@dataclass
class BenchResult:
    target: str
    concurrency: int
    total_requests: int
    successful: int
    failed: int
    latencies_ms: list[float] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    wall_time_sec: float = 0.0

    @property
    def p50(self) -> float:
        return _pct(self.latencies_ms, 50)

    @property
    def p95(self) -> float:
        return _pct(self.latencies_ms, 95)

    @property
    def p99(self) -> float:
        return _pct(self.latencies_ms, 99)

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    @property
    def tok_per_sec(self) -> float:
        return self.total_completion_tokens / self.wall_time_sec if self.wall_time_sec > 0 else 0

    @property
    def req_per_sec(self) -> float:
        return self.successful / self.wall_time_sec if self.wall_time_sec > 0 else 0


def _pct(data: list[float], pct: int) -> float:
    if not data:
        return 0
    s = sorted(data)
    idx = (len(s) - 1) * pct / 100
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# ── API ─────────────────────────────────────────────────────────────

async def call_api(
    session: aiohttp.ClientSession,
    url: str,
    messages: list[dict],
    model: str,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    t0 = time.perf_counter()
    try:
        async with semaphore:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                raw = await resp.read()
                data = json.loads(raw.decode("utf-8", errors="replace"), strict=False)
                elapsed = (time.perf_counter() - t0) * 1000
                usage = data.get("usage", {})
                return RequestResult(
                    latency_ms=elapsed,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    success=True,
                )
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return RequestResult(latency_ms=elapsed, prompt_tokens=0,
                             completion_tokens=0, success=False, error=str(e))


async def warmup(session: aiohttp.ClientSession, url: str, model: str):
    sem = asyncio.Semaphore(1)
    for _ in range(3):
        await call_api(session, url,
                       [{"role": "user", "content": "Hi"}], model, sem)


async def wait_for_server(url: str, model: str, timeout: int = 300) -> bool:
    """Wait until server is ready."""
    health_url = url.rsplit("/v1/", 1)[0] + "/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with aiohttp.ClientSession() as session:
                # Try health endpoint first
                try:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            return True
                except Exception:
                    pass
                # Fallback: try a completion
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "temperature": 0,
                }
                async with session.post(
                    url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        return True
        except Exception:
            pass
        await asyncio.sleep(2)
    return False


# ── Runner ──────────────────────────────────────────────────────────

async def run_bench(
    url: str, model: str, target_name: str,
    concurrency: int, num_samples: int,
) -> BenchResult:
    requests = []
    for i in range(num_samples):
        p = PROMPTS[i % len(PROMPTS)]
        requests.append([{"role": "system", "content": "Answer briefly."}, p])

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        sys.stderr.write(f"  Warming up {target_name}... ")
        sys.stderr.flush()
        await warmup(session, url, model)
        sys.stderr.write("done\n")
        sys.stderr.write(f"  Running {num_samples} reqs (concurrency={concurrency})...\n")

        t0 = time.perf_counter()
        tasks = [call_api(session, url, msgs, model, semaphore) for msgs in requests]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    bench = BenchResult(
        target=target_name, concurrency=concurrency,
        total_requests=num_samples,
        successful=sum(1 for r in results if r.success),
        failed=sum(1 for r in results if not r.success),
        wall_time_sec=wall_time,
    )
    for r in results:
        if r.success:
            bench.latencies_ms.append(r.latency_ms)
            bench.total_prompt_tokens += r.prompt_tokens
            bench.total_completion_tokens += r.completion_tokens

    errors = [r for r in results if not r.success]
    if errors:
        for e in errors[:3]:
            sys.stderr.write(f"    ERROR: {e.error}\n")
        if len(errors) > 3:
            sys.stderr.write(f"    ... and {len(errors)-3} more errors\n")

    return bench


# ── Display ─────────────────────────────────────────────────────────

def print_comparison_table(results: list[BenchResult]):
    by_target: dict[str, dict[int, BenchResult]] = {}
    for r in results:
        by_target.setdefault(r.target, {})[r.concurrency] = r

    hayabusa = by_target.get("Hayabusa", {})
    vllm = by_target.get("vllm-mlx", {})

    print()
    print("┌──────┬──────────────────────┬──────────────────────┬──────────┐")
    print("│ Conc │      Hayabusa        │      vllm-mlx        │   倍率   │")
    print("│      │ tok/s  Avg(ms)  P95  │ tok/s  Avg(ms)  P95  │          │")
    print("├──────┼──────────────────────┼──────────────────────┼──────────┤")

    for c in CONCURRENCIES:
        h = hayabusa.get(c)
        v = vllm.get(c)
        if not h or not v:
            continue
        ratio = h.tok_per_sec / v.tok_per_sec if v.tok_per_sec > 0 else float("inf")
        print(f"│ {c:>4} │ {h.tok_per_sec:>5.1f} {h.avg_latency:>7.0f} {h.p95:>5.0f} │"
              f" {v.tok_per_sec:>5.1f} {v.avg_latency:>7.0f} {v.p95:>5.0f} │"
              f" {ratio:>6.2f}x  │")

    print("└──────┴──────────────────────┴──────────────────────┴──────────┘")
    print()


def save_results(results: list[BenchResult]):
    data = []
    for r in results:
        data.append({
            "target": r.target,
            "concurrency": r.concurrency,
            "total_requests": r.total_requests,
            "successful": r.successful,
            "failed": r.failed,
            "wall_time_sec": round(r.wall_time_sec, 3),
            "avg_latency_ms": round(r.avg_latency, 1),
            "p50_ms": round(r.p50, 1),
            "p95_ms": round(r.p95, 1),
            "p99_ms": round(r.p99, 1),
            "tok_per_sec": round(r.tok_per_sec, 2),
            "req_per_sec": round(r.req_per_sec, 3),
            "total_prompt_tokens": r.total_prompt_tokens,
            "total_completion_tokens": r.total_completion_tokens,
        })

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "model": MODEL_ID,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "prompts_count": len(PROMPTS),
            "concurrencies": CONCURRENCIES,
        },
        "results": data,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_PATH}")


# ── Main ────────────────────────────────────────────────────────────

async def main_async(args):
    num_samples = args.samples

    hayabusa_url = f"http://localhost:{args.hayabusa_port}/v1/chat/completions"
    vllm_url = f"http://localhost:{args.vllm_port}/v1/chat/completions"

    targets = [
        ("Hayabusa", hayabusa_url, "local"),
        ("vllm-mlx", vllm_url, MODEL_ID),
    ]

    # Check servers
    for name, url, model in targets:
        sys.stderr.write(f"Waiting for {name} at {url}... ")
        sys.stderr.flush()
        ok = await wait_for_server(url, model, timeout=10)
        if ok:
            sys.stderr.write("OK\n")
        else:
            sys.stderr.write("UNAVAILABLE\n")
            print(f"\nERROR: {name} is not running.")
            print(f"\nStart servers first:")
            print(f"  # Hayabusa:")
            print(f"  .build/debug/Hayabusa {MODEL_ID} --backend mlx")
            print(f"  # vllm-mlx:")
            print(f"  vllm-mlx serve {MODEL_ID} --port {args.vllm_port} --continuous-batching")
            sys.exit(1)

    print()
    print("=" * 70)
    print("  Hayabusa vs vllm-mlx Competitive Benchmark")
    print(f"  Model: {MODEL_ID}")
    print(f"  max_tokens={MAX_TOKENS}  samples={num_samples}")
    print("=" * 70)
    print()

    all_results: list[BenchResult] = []

    for conc in CONCURRENCIES:
        for name, url, model in targets:
            print(f"--- {name} (concurrency={conc}) ---")
            result = await run_bench(url, model, name, conc, num_samples)
            all_results.append(result)
            print(f"  => avg={result.avg_latency:.0f}ms  p95={result.p95:.0f}ms  "
                  f"tok/s={result.tok_per_sec:.1f}  ({result.successful}/{result.total_requests} ok)")
            print()

    print_comparison_table(all_results)
    save_results(all_results)


def main():
    parser = argparse.ArgumentParser(
        description="Hayabusa vs vllm-mlx competitive benchmark")
    parser.add_argument("--samples", type=int, default=20,
                        help="Requests per concurrency level (default: 20)")
    parser.add_argument("--hayabusa-port", type=int, default=8080,
                        help="Hayabusa port (default: 8080)")
    parser.add_argument("--vllm-port", type=int, default=9090,
                        help="vllm-mlx port (default: 9090)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
