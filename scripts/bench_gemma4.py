#!/usr/bin/env python3
"""Hayabusa vs Ollama — Gemma 4 スピード対決ベンチマーク.

Google Gemma 4 ファミリーで Hayabusa と Ollama の推論速度を比較する。

Models:
  - Gemma 4 E2B (5B)   : 超軽量、モバイル級
  - Gemma 4 E4B (8B)   : Qwen3.5-9Bと同クラス
  - Gemma 4 26B-A4B    : MoE、アクティブ4Bで26B級の性能
  - Gemma 4 31B        : フルサイズ

Setup:
    # --- Hayabusa (port 8080) ---
    # E4B (8B) — Qwen3.5-9Bと同クラス比較
    .build/release/Hayabusa models/gemma-4-E4B-it-Q4_K_M.gguf --kv-quantize int8

    # 26B-A4B MoE — アクティブ4Bなので16GB Macでも動く
    .build/release/Hayabusa models/gemma-4-26B-A4B-it-Q4_K_M.gguf --kv-quantize tq3

    # --- Ollama (port 11434) ---
    ollama pull gemma4:e4b
    ollama pull gemma4:26b-a4b
    ollama serve

Usage:
    python scripts/bench_gemma4.py
    python scripts/bench_gemma4.py --variant e4b
    python scripts/bench_gemma4.py --variant 26b-a4b --samples 20
    python scripts/bench_gemma4.py --variant all  # 全モデル連続ベンチ
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

# ── Config ──────────────────────────────────────────────────────────

HAYABUSA_BASE = "http://localhost:{port}"
OLLAMA_BASE = "http://localhost:11434"

# Gemma 4 バリアント定義
VARIANTS = {
    "e2b": {
        "name": "Gemma 4 E2B (5B)",
        "ollama_model": "gemma4:e2b",
        "hayabusa_model": "gemma-4-E2B-it-Q4_K_M.gguf",
        "params": "5B",
    },
    "e4b": {
        "name": "Gemma 4 E4B (8B)",
        "ollama_model": "gemma4:e4b",
        "hayabusa_model": "gemma-4-E4B-it-Q4_K_M.gguf",
        "params": "8B",
    },
    "26b-a4b": {
        "name": "Gemma 4 26B-A4B (MoE)",
        "ollama_model": "gemma4:26b-a4b",
        "hayabusa_model": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
        "params": "26B (active 4B)",
    },
    "31b": {
        "name": "Gemma 4 31B",
        "ollama_model": "gemma4:31b",
        "hayabusa_model": "gemma-4-31B-it-Q4_K_M.gguf",
        "params": "31B",
    },
}

# 多様なプロンプト
PROMPTS = [
    "What is 2+2?",
    "Say hello in Japanese.",
    "Name three primary colors.",
    "Explain the difference between a stack and a queue in 2 sentences.",
    "Write a Python function that checks if a number is prime.",
    "What are the SOLID principles? List them briefly.",
    "Explain how a hash table works, covering hash function, collision resolution, and time complexity. Keep it under 100 words.",
    "Compare merge sort and quicksort in 3 sentences.",
    "What is the CAP theorem? Explain each property briefly.",
    "Write a Python implementation of binary search with type hints.",
    "Describe how TCP three-way handshake works in 3 sentences.",
    "Write a Swift function to reverse a linked list.",
]

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "bench_gemma4.json"


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class RequestResult:
    latency_ms: float
    ttft_ms: float
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error: str | None = None


@dataclass
class BenchResult:
    target: str
    variant: str
    concurrency: int
    total_requests: int
    successful: int
    failed: int
    latencies_ms: list[float] = field(default_factory=list)
    ttfts_ms: list[float] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    wall_time_sec: float = 0.0

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

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
    def avg_ttft(self) -> float:
        return statistics.mean(self.ttfts_ms) if self.ttfts_ms else 0

    @property
    def p50_ttft(self) -> float:
        return _pct(self.ttfts_ms, 50)

    @property
    def p95_ttft(self) -> float:
        return _pct(self.ttfts_ms, 95)

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


# ── Streaming API call ──────────────────────────────────────────────

async def call_streaming(
    session: aiohttp.ClientSession,
    url: str,
    messages: list[dict],
    model: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    t0 = time.perf_counter()
    ttft = 0.0
    completion_tokens = 0
    prompt_tokens = 0
    first_token_seen = False

    try:
        async with semaphore:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                async for line in resp.content:
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text or not text.startswith("data: "):
                        continue
                    data_str = text[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if not first_token_seen:
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            ttft = (time.perf_counter() - t0) * 1000
                            first_token_seen = True

                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)

                elapsed = (time.perf_counter() - t0) * 1000

                if completion_tokens == 0:
                    completion_tokens = max_tokens

                return RequestResult(
                    latency_ms=elapsed,
                    ttft_ms=ttft if ttft > 0 else elapsed,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    success=True,
                )
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return RequestResult(
            latency_ms=elapsed, ttft_ms=0,
            prompt_tokens=0, completion_tokens=0,
            success=False, error=str(e),
        )


# ── Server check ───────────────────────────────────────────────────

async def check_server(base_url: str, name: str) -> bool:
    health_urls = [f"{base_url}/health", f"{base_url}/api/tags"]
    async with aiohttp.ClientSession() as session:
        for url in health_urls:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                continue
    return False


async def get_ollama_version() -> str | None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{OLLAMA_BASE}/api/version",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
                return data.get("version", "unknown")
    except Exception:
        return None


# ── Warmup ─────────────────────────────────────────────────────────

async def warmup(session: aiohttp.ClientSession, url: str, model: str, max_tokens: int):
    sem = asyncio.Semaphore(1)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": min(max_tokens, 16),
        "temperature": 0,
        "stream": False,
    }
    for _ in range(3):
        try:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                await resp.read()
        except Exception:
            pass


# ── Bench runner ───────────────────────────────────────────────────

async def run_bench(
    url: str, model: str, target_name: str, variant: str,
    concurrency: int, num_samples: int, max_tokens: int,
) -> BenchResult:
    requests = []
    for i in range(num_samples):
        prompt = PROMPTS[i % len(PROMPTS)]
        requests.append([
            {"role": "system", "content": "Answer briefly and precisely."},
            {"role": "user", "content": prompt},
        ])

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        sys.stderr.write(f"  Warming up {target_name}... ")
        sys.stderr.flush()
        await warmup(session, url, model, max_tokens)
        sys.stderr.write("done\n")
        sys.stderr.write(f"  Running {num_samples} reqs (conc={concurrency})...\n")

        t0 = time.perf_counter()
        tasks = [
            call_streaming(session, url, msgs, model, max_tokens, semaphore)
            for msgs in requests
        ]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    bench = BenchResult(
        target=target_name, variant=variant,
        concurrency=concurrency,
        total_requests=num_samples,
        successful=sum(1 for r in results if r.success),
        failed=sum(1 for r in results if not r.success),
        wall_time_sec=wall_time,
    )
    for r in results:
        if r.success:
            bench.latencies_ms.append(r.latency_ms)
            bench.ttfts_ms.append(r.ttft_ms)
            bench.total_prompt_tokens += r.prompt_tokens
            bench.total_completion_tokens += r.completion_tokens

    errors = [r for r in results if not r.success]
    if errors:
        for e in errors[:3]:
            sys.stderr.write(f"    ERROR: {e.error}\n")

    return bench


# ── Display ────────────────────────────────────────────────────────

def print_header(variant_info: dict, ollama_version: str | None, max_tokens: int, samples: int):
    print()
    print("=" * 90)
    print(f"  HAYABUSA vs OLLAMA — Gemma 4 スピード対決")
    print(f"  Model: {variant_info['name']} ({variant_info['params']})")
    print(f"  Ollama: v{ollama_version or 'unknown'}  |  max_tokens={max_tokens}  |  samples={samples}")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    print()


def print_result_table(results: list[BenchResult]):
    print()
    header = (
        f"{'Target':<12} {'Variant':<14} {'Conc':>4} {'OK':>4} "
        f"{'TTFT':>7} {'Avg(ms)':>8} {'P95(ms)':>8} "
        f"{'tok/s':>7} {'req/s':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.target:<12} {r.variant:<14} {r.concurrency:>4} {r.successful:>4} "
            f"{r.avg_ttft:>7.0f} {r.avg_latency:>8.0f} {r.p95:>8.0f} "
            f"{r.tok_per_sec:>7.1f} {r.req_per_sec:>6.2f}"
        )
    print()


def print_versus(results: list[BenchResult]):
    by_target: dict[str, dict[int, BenchResult]] = {}
    for r in results:
        by_target.setdefault(r.target, {})[r.concurrency] = r

    hayabusa = by_target.get("Hayabusa", {})
    ollama = by_target.get("Ollama", {})

    if not hayabusa or not ollama:
        return

    print("┌──────────────────────────────────────────────────────────────────────┐")
    print("│              HAYABUSA vs OLLAMA  Gemma 4 直接対決                    │")
    print("├──────┬──────────────────┬──────────────────┬────────────────────────┤")
    print("│ Conc │    Hayabusa      │     Ollama       │       勝敗             │")
    print("│      │ tok/s  Avg(ms)   │ tok/s  Avg(ms)   │ Speed   Latency       │")
    print("├──────┼──────────────────┼──────────────────┼────────────────────────┤")

    concurrencies = sorted(set(list(hayabusa.keys()) + list(ollama.keys())))
    h_wins = o_wins = 0

    for c in concurrencies:
        h = hayabusa.get(c)
        o = ollama.get(c)
        if not h or not o:
            continue

        tok_ratio = h.tok_per_sec / o.tok_per_sec if o.tok_per_sec > 0 else float("inf")
        lat_ratio = o.avg_latency / h.avg_latency if h.avg_latency > 0 else 0

        tok_w = "H" if tok_ratio > 1.05 else ("O" if tok_ratio < 0.95 else "=")
        lat_w = "H" if lat_ratio > 1.05 else ("O" if lat_ratio < 0.95 else "=")

        for w in [tok_w, lat_w]:
            if w == "H":
                h_wins += 1
            elif w == "O":
                o_wins += 1

        def fmt(label, ratio):
            if label == "H":
                return f"H {ratio:.2f}x"
            elif label == "O":
                return f"O {1/ratio:.2f}x"
            return " DRAW  "

        print(
            f"│ {c:>4} │ {h.tok_per_sec:>5.1f} {h.avg_latency:>7.0f}   │"
            f" {o.tok_per_sec:>5.1f} {o.avg_latency:>7.0f}   │"
            f" {fmt(tok_w, tok_ratio):>7s}  {fmt(lat_w, lat_ratio):>7s}     │"
        )

    print("├──────┴──────────────────┴──────────────────┴────────────────────────┤")
    total = h_wins + o_wins
    if h_wins > o_wins:
        verdict = f"HAYABUSA WIN  ({h_wins}/{total})"
    elif o_wins > h_wins:
        verdict = f"OLLAMA WIN  ({o_wins}/{total})"
    else:
        verdict = "DRAW"
    print(f"│  VERDICT: {verdict:<60s} │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    print()


def save_results(results: list[BenchResult], variant_info: dict, ollama_version: str | None, max_tokens: int):
    data = []
    for r in results:
        data.append({
            "target": r.target,
            "variant": r.variant,
            "concurrency": r.concurrency,
            "total_requests": r.total_requests,
            "successful": r.successful,
            "failed": r.failed,
            "wall_time_sec": round(r.wall_time_sec, 3),
            "avg_latency_ms": round(r.avg_latency, 1),
            "p50_ms": round(r.p50, 1),
            "p95_ms": round(r.p95, 1),
            "p99_ms": round(r.p99, 1),
            "avg_ttft_ms": round(r.avg_ttft, 1),
            "p50_ttft_ms": round(r.p50_ttft, 1),
            "p95_ttft_ms": round(r.p95_ttft, 1),
            "tok_per_sec": round(r.tok_per_sec, 2),
            "req_per_sec": round(r.req_per_sec, 3),
            "total_prompt_tokens": r.total_prompt_tokens,
            "total_completion_tokens": r.total_completion_tokens,
        })

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "variant": variant_info["name"],
            "params": variant_info["params"],
            "ollama_version": ollama_version,
            "ollama_model": variant_info["ollama_model"],
            "hayabusa_model": variant_info["hayabusa_model"],
            "max_tokens": max_tokens,
            "prompts_count": len(PROMPTS),
        },
        "results": data,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_PATH}")


# ── Main ────────────────────────────────────────────────────────────

async def run_variant(
    variant_key: str, args, ollama_version: str | None,
) -> list[BenchResult]:
    variant_info = VARIANTS[variant_key]
    hayabusa_base = HAYABUSA_BASE.format(port=args.hayabusa_port)
    hayabusa_url = f"{hayabusa_base}/v1/chat/completions"
    ollama_url = f"{OLLAMA_BASE}/v1/chat/completions"

    targets = []
    if "hayabusa" in args.target:
        ok = await check_server(hayabusa_base, "Hayabusa")
        if ok:
            targets.append(("Hayabusa", hayabusa_url, variant_info["hayabusa_model"]))
        else:
            sys.stderr.write(f"  Hayabusa not available at {hayabusa_base}\n")

    if "ollama" in args.target:
        ok = await check_server(OLLAMA_BASE, "Ollama")
        if ok:
            targets.append(("Ollama", ollama_url, variant_info["ollama_model"]))
        else:
            sys.stderr.write(f"  Ollama not available at {OLLAMA_BASE}\n")

    if not targets:
        print(f"\nNo servers available for {variant_info['name']}.")
        print(f"\nSetup:")
        print(f"  # Hayabusa:")
        print(f"  .build/release/Hayabusa models/{variant_info['hayabusa_model']} --kv-quantize int8")
        print(f"  # Ollama:")
        print(f"  ollama pull {variant_info['ollama_model']} && ollama serve")
        return []

    print_header(variant_info, ollama_version, args.max_tokens, args.samples)

    all_results: list[BenchResult] = []

    for conc in args.concurrency:
        for name, url, model in targets:
            print(f"--- {name} / {variant_info['name']} (concurrency={conc}) ---")
            result = await run_bench(
                url, model, name, variant_key,
                conc, args.samples, args.max_tokens,
            )
            all_results.append(result)
            print(
                f"  => TTFT={result.avg_ttft:.0f}ms  avg={result.avg_latency:.0f}ms  "
                f"tok/s={result.tok_per_sec:.1f}  "
                f"({result.successful}/{result.total_requests} ok)"
            )
            print()

    print_result_table(all_results)
    print_versus(all_results)
    save_results(all_results, variant_info, ollama_version, args.max_tokens)

    return all_results


async def main_async(args):
    ollama_version = await get_ollama_version()

    if args.variant == "all":
        variants = list(VARIANTS.keys())
    else:
        variants = [args.variant]

    for v in variants:
        if v not in VARIANTS:
            print(f"Unknown variant: {v}")
            print(f"Available: {', '.join(VARIANTS.keys())}")
            sys.exit(1)

    for v in variants:
        await run_variant(v, args, ollama_version)


def main():
    parser = argparse.ArgumentParser(
        description="Hayabusa vs Ollama — Gemma 4 スピード対決",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--variant", default="e4b",
        choices=list(VARIANTS.keys()) + ["all"],
        help="Gemma 4 variant (default: e4b)",
    )
    parser.add_argument(
        "--target", nargs="+", default=["hayabusa", "ollama"],
        choices=["hayabusa", "ollama"],
    )
    parser.add_argument(
        "--concurrency", nargs="+", type=int, default=[1, 4, 8, 16],
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--hayabusa-port", type=int, default=8080)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
