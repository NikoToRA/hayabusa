#!/usr/bin/env python3
"""Gemma 4 E4B vs Qwen3.5-9B — Ollama上での速度比較 + Hayabusa参照値.

Ollama v0.20+ 上で Gemma 4 E4B と Qwen3.5-9B を同条件で比較し、
既存の Hayabusa (Qwen3.5-9B) ベンチ結果と並べて表示する。
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

OLLAMA_BASE = "http://localhost:11434"

MODELS = {
    "gemma4:e4b": "Gemma4-E4B (8B)",
    "qwen3.5:9b": "Qwen3.5-9B",
}

PROMPTS = [
    "What is 2+2?",
    "Say hello in Japanese.",
    "Name three primary colors.",
    "Explain the difference between a stack and a queue in 2 sentences.",
    "Write a Python function that checks if a number is prime.",
    "What are the SOLID principles? List them briefly.",
    "Explain how a hash table works. Cover hash function, collision resolution, and time complexity. Keep it under 100 words.",
    "Compare merge sort and quicksort in 3 sentences.",
    "What is the CAP theorem? Explain each property briefly.",
    "Write a Python implementation of binary search with type hints.",
    "Describe how TCP three-way handshake works in 3 sentences.",
    "Write a Swift function to reverse a linked list.",
]

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "bench_gemma4_vs_qwen.json"


@dataclass
class Result:
    model: str
    model_label: str
    concurrency: int
    total: int
    ok: int
    failed: int
    wall_sec: float
    latencies: list[float] = field(default_factory=list)
    ttfts: list[float] = field(default_factory=list)
    comp_tokens: int = 0

    @property
    def tok_s(self): return self.comp_tokens / self.wall_sec if self.wall_sec else 0
    @property
    def avg_lat(self): return statistics.mean(self.latencies) if self.latencies else 0
    @property
    def avg_ttft(self): return statistics.mean(self.ttfts) if self.ttfts else 0
    @property
    def p95(self): return _pct(self.latencies, 95)
    @property
    def req_s(self): return self.ok / self.wall_sec if self.wall_sec else 0


def _pct(d, p):
    if not d: return 0
    s = sorted(d)
    i = (len(s)-1)*p/100
    lo = int(i); hi = min(lo+1, len(s)-1)
    return s[lo]*(1-(i-lo)) + s[hi]*(i-lo)


async def call_stream(session, url, msgs, model, max_tok, sem):
    payload = {"model": model, "messages": msgs, "max_tokens": max_tok, "temperature": 0, "stream": True}
    t0 = time.perf_counter()
    ttft = comp = 0
    first = False
    try:
        async with sem:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as r:
                async for line in r.content:
                    txt = line.decode("utf-8", errors="replace").strip()
                    if not txt or not txt.startswith("data: "): continue
                    ds = txt[6:]
                    if ds == "[DONE]": break
                    try: chunk = json.loads(ds)
                    except: continue
                    if not first:
                        d = chunk.get("choices",[{}])[0].get("delta",{})
                        if d.get("content"):
                            ttft = (time.perf_counter()-t0)*1000; first = True
                    u = chunk.get("usage")
                    if u: comp = u.get("completion_tokens", comp)
                elapsed = (time.perf_counter()-t0)*1000
                if comp == 0: comp = max_tok
                return elapsed, ttft or elapsed, comp, True
    except Exception as e:
        return (time.perf_counter()-t0)*1000, 0, 0, False


async def warmup(session, url, model, max_tok):
    sem = asyncio.Semaphore(1)
    for _ in range(3):
        await call_stream(session, url, [{"role":"user","content":"Hi"}], model, 8, sem)


async def bench(model, label, conc, samples, max_tok):
    url = f"{OLLAMA_BASE}/v1/chat/completions"
    reqs = []
    for i in range(samples):
        reqs.append([{"role":"system","content":"Answer briefly."},{"role":"user","content":PROMPTS[i%len(PROMPTS)]}])

    sem = asyncio.Semaphore(conc)
    async with aiohttp.ClientSession() as s:
        sys.stderr.write(f"  Warmup {label}..."); sys.stderr.flush()
        await warmup(s, url, model, max_tok)
        sys.stderr.write(f" Run {samples}r conc={conc}\n")
        t0 = time.perf_counter()
        tasks = [call_stream(s, url, m, model, max_tok, sem) for m in reqs]
        raw = await asyncio.gather(*tasks)
        wall = time.perf_counter()-t0

    r = Result(model=model, model_label=label, concurrency=conc, total=samples,
               ok=sum(1 for x in raw if x[3]), failed=sum(1 for x in raw if not x[3]), wall_sec=wall)
    for lat, ttft, comp, ok in raw:
        if ok:
            r.latencies.append(lat); r.ttfts.append(ttft); r.comp_tokens += comp
    return r


async def main():
    concs = [1, 4, 8, 16]
    samples = 20
    max_tok = 128

    # Check
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{OLLAMA_BASE}/api/version", timeout=aiohttp.ClientTimeout(total=5)) as r:
                ver = (await r.json()).get("version","?")
    except:
        print("Ollama not running"); return

    print()
    print("="*80)
    print(f"  Gemma 4 E4B vs Qwen3.5-9B — Ollama v{ver} スピード対決")
    print(f"  + Hayabusa参照値 (bench_vs_ollama.json)")
    print(f"  samples={samples}  max_tokens={max_tok}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()

    all_results = []
    for conc in concs:
        for model, label in MODELS.items():
            r = await bench(model, label, conc, samples, max_tok)
            all_results.append(r)
            print(f"  {label:20s} conc={conc:>2}  tok/s={r.tok_s:>6.1f}  avg={r.avg_lat:>7.0f}ms  ttft={r.avg_ttft:>7.0f}ms  ({r.ok}/{r.total})")
        print()

    # ── Summary Table ──
    print()
    hdr = f"{'Model':<22} {'Conc':>4} {'tok/s':>7} {'Avg(ms)':>8} {'TTFT':>7} {'P95':>8} {'req/s':>6}"
    print(hdr); print("-"*len(hdr))
    for r in all_results:
        print(f"{r.model_label:<22} {r.concurrency:>4} {r.tok_s:>7.1f} {r.avg_lat:>8.0f} {r.avg_ttft:>7.0f} {r.p95:>8.0f} {r.req_s:>6.2f}")

    # ── Load Hayabusa reference ──
    hayabusa_ref = SCRIPT_DIR / "bench_vs_ollama.json"
    h_data = {}
    if hayabusa_ref.exists():
        with open(hayabusa_ref) as f:
            jd = json.load(f)
        for r in jd["results"]:
            if r["target"] == "Hayabusa":
                h_data[r["concurrency"]] = r

    # ── Cross comparison ──
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────────┐")
    print("│             三つ巴対決: Hayabusa(Qwen) vs Ollama(Gemma4) vs Ollama(Qwen)        │")
    print("├──────┬──────────────────┬──────────────────┬──────────────────┬─────────────────┤")
    print("│ Conc │ Hayabusa(Qwen)   │ Ollama(Gemma4)   │ Ollama(Qwen)     │ 最速            │")
    print("│      │ tok/s            │ tok/s            │ tok/s            │                 │")
    print("├──────┼──────────────────┼──────────────────┼──────────────────┼─────────────────┤")

    by_model_conc = {}
    for r in all_results:
        by_model_conc[(r.model, r.concurrency)] = r

    for conc in concs:
        h_tok = h_data.get(conc, {}).get("tok_per_sec", 0)
        g = by_model_conc.get(("gemma4:e4b", conc))
        q = by_model_conc.get(("qwen3.5:9b", conc))
        g_tok = g.tok_s if g else 0
        q_tok = q.tok_s if q else 0

        speeds = {"Hayabusa": h_tok, "Ollama(G4)": g_tok, "Ollama(Qw)": q_tok}
        winner = max(speeds, key=speeds.get) if any(speeds.values()) else "?"
        best = max(speeds.values())
        ratio = f"{best/min(v for v in speeds.values() if v > 0):.1f}x" if min(v for v in speeds.values() if v > 0) > 0 else ""

        print(f"│ {conc:>4} │ {h_tok:>7.1f} tok/s    │ {g_tok:>7.1f} tok/s    │ {q_tok:>7.1f} tok/s    │ {winner} {ratio:>5s}  │")

    print("└──────┴──────────────────┴──────────────────┴──────────────────┴─────────────────┘")
    print()

    # Save
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ollama_version": ver,
        "max_tokens": max_tok,
        "samples": samples,
        "results": [{
            "model": r.model, "label": r.model_label, "concurrency": r.concurrency,
            "tok_per_sec": round(r.tok_s, 2), "avg_latency_ms": round(r.avg_lat, 1),
            "avg_ttft_ms": round(r.avg_ttft, 1), "p95_ms": round(r.p95, 1),
            "req_per_sec": round(r.req_s, 3), "ok": r.ok, "failed": r.failed,
            "wall_sec": round(r.wall_sec, 3), "comp_tokens": r.comp_tokens,
        } for r in all_results],
        "hayabusa_reference": {str(k): v for k, v in h_data.items()},
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
