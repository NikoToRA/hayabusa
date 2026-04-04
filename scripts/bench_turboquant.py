#!/usr/bin/env python3
"""
Benchmark TurboQuant KV cache quantization modes.

Tests KV cache memory usage and inference speed across:
  - f16 (baseline, no quantization)
  - int8 (Q8_0)
  - tq3 (TQ3_0, 3-bit TurboQuant)
  - tq4 (TQ4_0, 4-bit TurboQuant)

Usage:
  python3 scripts/bench_turboquant.py --model models/Qwen3.5-9B-Q4_K_M.gguf
  python3 scripts/bench_turboquant.py --model models/Qwen3.5-9B-Q4_K_M.gguf --output bench_tq.json
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HAYABUSA_BIN = ".build/release/Hayabusa"
DEFAULT_PORT = 18199
PROMPT = "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves."


def find_hayabusa():
    """Find the Hayabusa binary."""
    candidates = [
        HAYABUSA_BIN,
        ".build/debug/Hayabusa",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    print("ERROR: Hayabusa binary not found. Run 'swift build -c release' first.")
    sys.exit(1)


def start_server(binary, model, kv_mode, port, slots=4, ctx_per_slot=4096):
    """Start a Hayabusa server with the given KV quantization mode."""
    cmd = [binary, model, "--slots", str(slots), "--ctx-per-slot", str(ctx_per_slot)]
    if kv_mode != "off":
        cmd += ["--kv-quantize", kv_mode]

    env = os.environ.copy()
    env["HAYABUSA_PORT"] = str(port)

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_server(port, timeout=60):
    """Wait for the server to be ready."""
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(0.5)
    return False


def bench_completion(port, prompt, max_tokens=256):
    """Run a completion benchmark and return timing stats."""
    import urllib.request

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        elapsed = time.time() - start
    except Exception as e:
        return {"error": str(e), "tok_per_sec": 0, "tokens": 0, "elapsed": 0}

    # Extract usage stats
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

    return {
        "tok_per_sec": round(tok_per_sec, 2),
        "tokens": completion_tokens,
        "elapsed": round(elapsed, 3),
    }


def stop_server(proc):
    """Stop the server process."""
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def estimate_kv_memory(slots, ctx_per_slot, n_layers, n_heads, head_dim, mode):
    """Estimate KV cache memory in bytes."""
    total_elements = 2 * slots * ctx_per_slot * n_layers * n_heads * head_dim

    if mode == "off":
        return total_elements * 2  # fp16
    elif mode == "int8":
        scale_overhead = (total_elements // 32) * 2
        return total_elements * 1 + scale_overhead
    elif mode == "tq3":
        return (total_elements // 32) * 14  # 14 bytes per 32 elements
    elif mode == "tq4":
        return (total_elements // 32) * 18  # 18 bytes per 32 elements
    return 0


def main():
    parser = argparse.ArgumentParser(description="Benchmark TurboQuant KV cache quantization")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--output", default="bench_turboquant.json", help="Output JSON path")
    parser.add_argument("--slots", type=int, default=4, help="Number of KV cache slots")
    parser.add_argument("--ctx-per-slot", type=int, default=4096, help="Context per slot")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs before timing")
    parser.add_argument("--bench-runs", type=int, default=3, help="Timed benchmark runs")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["off", "int8", "tq3", "tq4"],
        help="KV quantization modes to test",
    )
    args = parser.parse_args()

    binary = find_hayabusa()
    results = []

    # Qwen3.5-9B params
    n_layers = 32
    n_heads = 16
    head_dim = 256

    print(f"Model: {args.model}")
    print(f"Slots: {args.slots}, Context/slot: {args.ctx_per_slot}")
    print(f"Modes: {args.modes}")
    print()

    for mode in args.modes:
        print(f"=== Testing KV mode: {mode} ===")

        mem_bytes = estimate_kv_memory(
            args.slots, args.ctx_per_slot, n_layers, n_heads, head_dim, mode
        )
        mem_mb = mem_bytes / (1024 * 1024)
        print(f"  Estimated KV memory: {mem_mb:.1f} MB")

        proc = start_server(binary, args.model, mode, args.port, args.slots, args.ctx_per_slot)

        try:
            print("  Waiting for server...", end="", flush=True)
            if not wait_for_server(args.port):
                print(" TIMEOUT")
                stop_server(proc)
                results.append({
                    "mode": mode,
                    "error": "server_timeout",
                    "kv_memory_mb": round(mem_mb, 1),
                })
                continue
            print(" ready")

            # Warmup
            for _ in range(args.warmup_runs):
                bench_completion(args.port, PROMPT, args.max_tokens)

            # Timed runs
            runs = []
            for r in range(args.bench_runs):
                stats = bench_completion(args.port, PROMPT, args.max_tokens)
                runs.append(stats)
                print(f"  Run {r+1}: {stats['tok_per_sec']} tok/s ({stats['tokens']} tokens in {stats['elapsed']}s)")

            avg_tps = sum(r["tok_per_sec"] for r in runs) / len(runs) if runs else 0
            avg_tokens = sum(r["tokens"] for r in runs) / len(runs) if runs else 0

            result = {
                "mode": mode,
                "kv_memory_mb": round(mem_mb, 1),
                "avg_tok_per_sec": round(avg_tps, 2),
                "avg_tokens": round(avg_tokens, 1),
                "runs": runs,
            }

            if mode == "off":
                result["memory_savings_pct"] = 0.0
            else:
                baseline_mem = estimate_kv_memory(
                    args.slots, args.ctx_per_slot, n_layers, n_heads, head_dim, "off"
                )
                savings = (1 - mem_bytes / baseline_mem) * 100
                result["memory_savings_pct"] = round(savings, 1)

            results.append(result)
            print(f"  Average: {avg_tps:.2f} tok/s, memory savings: {result.get('memory_savings_pct', 0):.1f}%")

        finally:
            stop_server(proc)
            time.sleep(1)  # Brief pause between modes

    # Summary
    print()
    print("=" * 60)
    print(f"{'Mode':<8} {'KV Memory':>10} {'Savings':>10} {'tok/s':>10}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['mode']:<8} {r.get('kv_memory_mb', '?'):>9} MB {'ERROR':>10} {'N/A':>10}")
        else:
            print(f"{r['mode']:<8} {r['kv_memory_mb']:>9.1f} MB {r['memory_savings_pct']:>9.1f}% {r['avg_tok_per_sec']:>9.2f}")
    print("=" * 60)

    # Max parallel slots estimation (16GB Mac Mini)
    print()
    print("Max parallel slots on 16GB Mac Mini (estimated):")
    available_gb = 10.0  # ~10GB available after model loading for 9B Q4_K_M
    for r in results:
        if "error" not in r and r["kv_memory_mb"] > 0:
            slots_per_gb = (available_gb * 1024) / (r["kv_memory_mb"] / args.slots)
            print(f"  {r['mode']:<8}: ~{int(slots_per_gb)} slots")

    # Save JSON
    output = {
        "model": args.model,
        "slots": args.slots,
        "ctx_per_slot": args.ctx_per_slot,
        "max_tokens": args.max_tokens,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
