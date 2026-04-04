#!/usr/bin/env python3
"""
Benchmark script for Hayabusa with vllm-mlx backend.

Sends concurrent requests to the Hayabusa server and measures:
  - Tokens per second (tok/s)
  - Latency (p50, p95, p99)
  - Throughput (requests/s)

Usage:
    # Start Hayabusa with vllm-mlx backend first:
    #   hayabusa --backend vllm-mlx --vllm-endpoint http://localhost:8000
    #
    # Then run the benchmark:
    python3 scripts/bench_vllm_mlx.py --url http://127.0.0.1:8080 --concurrency 8 --requests 50

    # Compare with native MLX backend:
    python3 scripts/bench_vllm_mlx.py --url http://127.0.0.1:8080 --concurrency 8 --requests 50 --tag mlx-native
"""

import argparse
import asyncio
import json
import time
import statistics
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required. Install with: pip install aiohttp")
    sys.exit(1)


@dataclass
class RequestResult:
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    tag: str
    url: str
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_s: float
    total_prompt_tokens: int
    total_completion_tokens: int
    throughput_rps: float
    tok_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float
    target_292_tps: str  # "PASS" or "FAIL"


PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to compute the Fibonacci sequence using memoization.",
    "What are the main differences between REST and GraphQL?",
    "Describe the architecture of a modern web browser.",
    "How does garbage collection work in Java?",
    "Explain the CAP theorem with practical examples.",
    "Write a haiku about machine learning.",
    "What is the difference between a process and a thread?",
    "Explain how TLS/SSL handshake works step by step.",
    "Describe the benefits and drawbacks of microservices architecture.",
]


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    start = time.monotonic()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            body = await resp.json()
            elapsed = time.monotonic() - start
            if resp.status != 200:
                return RequestResult(
                    latency_s=elapsed,
                    prompt_tokens=0,
                    completion_tokens=0,
                    success=False,
                    error=f"HTTP {resp.status}: {json.dumps(body)[:200]}",
                )
            usage = body.get("usage", {})
            return RequestResult(
                latency_s=elapsed,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                success=True,
            )
    except Exception as e:
        elapsed = time.monotonic() - start
        return RequestResult(
            latency_s=elapsed,
            prompt_tokens=0,
            completion_tokens=0,
            success=False,
            error=str(e),
        )


async def run_benchmark(
    url: str,
    concurrency: int,
    total_requests: int,
    max_tokens: int,
    tag: str,
) -> BenchmarkResult:
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def bounded_request(session, prompt):
        async with sem:
            return await send_request(session, url, prompt, max_tokens)

    print(f"[bench] Target: {url}")
    print(f"[bench] Concurrency: {concurrency}, Requests: {total_requests}, Max tokens: {max_tokens}")
    print(f"[bench] Tag: {tag}")
    print(f"[bench] Running...\n")

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        wall_start = time.monotonic()
        tasks = []
        for i in range(total_requests):
            prompt = PROMPTS[i % len(PROMPTS)]
            tasks.append(bounded_request(session, prompt))
        results = await asyncio.gather(*tasks)
        wall_elapsed = time.monotonic() - wall_start

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        print("[bench] ERROR: all requests failed!")
        for f in failed[:5]:
            print(f"  -> {f.error}")
        sys.exit(1)

    latencies_ms = [r.latency_s * 1000 for r in successful]
    latencies_ms.sort()

    total_prompt_tokens = sum(r.prompt_tokens for r in successful)
    total_completion_tokens = sum(r.completion_tokens for r in successful)

    tok_per_sec = total_completion_tokens / wall_elapsed if wall_elapsed > 0 else 0
    throughput = len(successful) / wall_elapsed if wall_elapsed > 0 else 0

    def percentile(data, p):
        k = (len(data) - 1) * (p / 100)
        f = int(k)
        c = f + 1
        if c >= len(data):
            return data[f]
        return data[f] + (k - f) * (data[c] - data[f])

    result = BenchmarkResult(
        tag=tag,
        url=url,
        concurrency=concurrency,
        total_requests=total_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_time_s=round(wall_elapsed, 3),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        throughput_rps=round(throughput, 2),
        tok_per_sec=round(tok_per_sec, 1),
        latency_p50_ms=round(percentile(latencies_ms, 50), 1),
        latency_p95_ms=round(percentile(latencies_ms, 95), 1),
        latency_p99_ms=round(percentile(latencies_ms, 99), 1),
        latency_mean_ms=round(statistics.mean(latencies_ms), 1),
        latency_min_ms=round(latencies_ms[0], 1),
        latency_max_ms=round(latencies_ms[-1], 1),
        target_292_tps="PASS" if tok_per_sec >= 292 else "FAIL",
    )

    # Print summary
    print(f"{'=' * 60}")
    print(f"  Benchmark Results ({tag})")
    print(f"{'=' * 60}")
    print(f"  Requests:    {result.successful_requests}/{result.total_requests} succeeded")
    print(f"  Wall time:   {result.total_time_s:.3f}s")
    print(f"  Throughput:  {result.throughput_rps:.2f} req/s")
    print(f"  Tok/s:       {result.tok_per_sec:.1f} (target: 292)")
    print(f"  Target:      {result.target_292_tps}")
    print(f"  Latency p50: {result.latency_p50_ms:.1f}ms")
    print(f"  Latency p95: {result.latency_p95_ms:.1f}ms")
    print(f"  Latency p99: {result.latency_p99_ms:.1f}ms")
    print(f"  Latency avg: {result.latency_mean_ms:.1f}ms")
    print(f"  Tokens:      {result.total_prompt_tokens} prompt + {result.total_completion_tokens} completion")
    if failed:
        print(f"  Failures:    {len(failed)}")
        for f in failed[:3]:
            print(f"    -> {f.error}")
    print(f"{'=' * 60}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hayabusa with vllm-mlx backend")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Hayabusa server URL")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent requests")
    parser.add_argument("--requests", type=int, default=50, help="Total requests to send")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per response")
    parser.add_argument("--tag", default="vllm-mlx", help="Tag for results (e.g., vllm-mlx, mlx-native)")
    parser.add_argument("--output", default="scripts/bench_vllm_mlx.json", help="Output JSON path")
    args = parser.parse_args()

    result = asyncio.run(
        run_benchmark(
            url=args.url,
            concurrency=args.concurrency,
            total_requests=args.requests,
            max_tokens=args.max_tokens,
            tag=args.tag,
        )
    )

    # Write JSON results
    with open(args.output, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\n[bench] Results written to {args.output}")


if __name__ == "__main__":
    main()
