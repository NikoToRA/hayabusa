#!/usr/bin/env python3
"""品質ベンチマーク: KV量子化劣化テスト + Hayabusa vs Ollama 出力品質比較.

A: KV量子化 off/int8/tq3/tq4 で品質劣化検証（Hayabusa単体）
B: Hayabusa vs Ollama 同一プロンプト出力diff（再現性検証）

Usage:
    # Hayabusa + Ollama 両方テスト
    python scripts/bench_quality.py

    # Hayabusaだけ（KV量子化比較用、サーバーは手動で切替）
    python scripts/bench_quality.py --target hayabusa --label kv-off
    python scripts/bench_quality.py --target hayabusa --label kv-int8
    python scripts/bench_quality.py --target hayabusa --label kv-tq3
    python scripts/bench_quality.py --target hayabusa --label kv-tq4

    # 全結果をマージして分析用JSON生成
    python scripts/bench_quality.py --merge
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp

# ── Config ──────────────────────────────────────────────────────────

HAYABUSA_URL = "http://localhost:{port}/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "quality_results"

# ── Test Suites ─────────────────────────────────────────────────────

# BFCL-style: Function Calling（ツール定義+呼出し精度）
BFCL_TESTS = [
    {
        "id": "bfcl_simple_01",
        "category": "simple",
        "messages": [
            {"role": "system", "content": "You have access to the following function:\n\n```json\n{\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"]}}, \"required\": [\"city\"]}}\n```\n\nRespond with a JSON function call."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ],
        "expected_function": "get_weather",
        "expected_args": {"city": "Tokyo"},
    },
    {
        "id": "bfcl_simple_02",
        "category": "simple",
        "messages": [
            {"role": "system", "content": "You have access to the following function:\n\n```json\n{\"name\": \"calculate\", \"parameters\": {\"type\": \"object\", \"properties\": {\"expression\": {\"type\": \"string\"}, \"precision\": {\"type\": \"integer\"}}, \"required\": [\"expression\"]}}\n```\n\nRespond with a JSON function call."},
            {"role": "user", "content": "Calculate 355 divided by 113 to 6 decimal places."},
        ],
        "expected_function": "calculate",
        "expected_args": {"expression": "355/113", "precision": 6},
    },
    {
        "id": "bfcl_multiple_01",
        "category": "multiple",
        "messages": [
            {"role": "system", "content": "You have access to these functions:\n\n```json\n[\n  {\"name\": \"search_web\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\"}, \"max_results\": {\"type\": \"integer\"}}, \"required\": [\"query\"]}},\n  {\"name\": \"send_email\", \"parameters\": {\"type\": \"object\", \"properties\": {\"to\": {\"type\": \"string\"}, \"subject\": {\"type\": \"string\"}, \"body\": {\"type\": \"string\"}}, \"required\": [\"to\", \"subject\", \"body\"]}}\n]\n```\n\nChoose the right function and respond with a JSON function call."},
            {"role": "user", "content": "Find the top 5 results about Rust programming language."},
        ],
        "expected_function": "search_web",
        "expected_args": {"query": "Rust programming language", "max_results": 5},
    },
    {
        "id": "bfcl_parallel_01",
        "category": "parallel",
        "messages": [
            {"role": "system", "content": "You have access to:\n\n```json\n{\"name\": \"get_stock_price\", \"parameters\": {\"type\": \"object\", \"properties\": {\"symbol\": {\"type\": \"string\"}}, \"required\": [\"symbol\"]}}\n```\n\nWhen multiple items are requested, respond with an array of function calls."},
            {"role": "user", "content": "Get the stock prices for AAPL, GOOGL, and MSFT."},
        ],
        "expected_function": "get_stock_price",
        "expected_count": 3,
    },
    {
        "id": "bfcl_irrelevance_01",
        "category": "irrelevance",
        "messages": [
            {"role": "system", "content": "You have access to:\n\n```json\n{\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}\n```\n\nOnly call the function if the user's request is relevant. Otherwise, respond normally without a function call."},
            {"role": "user", "content": "What is the meaning of life?"},
        ],
        "expected_function": None,  # Should NOT call function
    },
    {
        "id": "bfcl_irrelevance_02",
        "category": "irrelevance",
        "messages": [
            {"role": "system", "content": "You have access to:\n\n```json\n{\"name\": \"book_flight\", \"parameters\": {\"type\": \"object\", \"properties\": {\"origin\": {\"type\": \"string\"}, \"destination\": {\"type\": \"string\"}, \"date\": {\"type\": \"string\"}}, \"required\": [\"origin\", \"destination\", \"date\"]}}\n```\n\nOnly call the function if the user's request is relevant. Otherwise, respond normally."},
            {"role": "user", "content": "Tell me a joke about airplanes."},
        ],
        "expected_function": None,
    },
]

# HumanEval-style: Code Generation（正答率）
HUMANEVAL_TESTS = [
    {
        "id": "he_001",
        "category": "basic",
        "messages": [
            {"role": "system", "content": "Write only the Python function. No explanation, no markdown, no tests."},
            {"role": "user", "content": "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if a string is a palindrome, ignoring case and non-alphanumeric characters.\n    >>> is_palindrome('A man, a plan, a canal: Panama')\n    True\n    >>> is_palindrome('race a car')\n    False\n    >>> is_palindrome('')\n    True\n    \"\"\""},
        ],
        "test_code": """
assert is_palindrome('A man, a plan, a canal: Panama') == True
assert is_palindrome('race a car') == False
assert is_palindrome('') == True
assert is_palindrome('a') == True
assert is_palindrome('ab') == False
assert is_palindrome('Was it a car or a cat I saw?') == True
""",
    },
    {
        "id": "he_002",
        "category": "basic",
        "messages": [
            {"role": "system", "content": "Write only the Python function. No explanation, no markdown, no tests."},
            {"role": "user", "content": "def two_sum(nums: list[int], target: int) -> list[int]:\n    \"\"\"Return indices of two numbers that add up to target.\n    >>> two_sum([2, 7, 11, 15], 9)\n    [0, 1]\n    >>> two_sum([3, 2, 4], 6)\n    [1, 2]\n    \"\"\""},
        ],
        "test_code": """
assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]
assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]
assert sorted(two_sum([3, 3], 6)) == [0, 1]
""",
    },
    {
        "id": "he_003",
        "category": "algorithm",
        "messages": [
            {"role": "system", "content": "Write only the Python function. No explanation, no markdown, no tests."},
            {"role": "user", "content": "def longest_common_subsequence(s1: str, s2: str) -> int:\n    \"\"\"Return the length of the longest common subsequence.\n    >>> longest_common_subsequence('abcde', 'ace')\n    3\n    >>> longest_common_subsequence('abc', 'def')\n    0\n    >>> longest_common_subsequence('abc', 'abc')\n    3\n    \"\"\""},
        ],
        "test_code": """
assert longest_common_subsequence('abcde', 'ace') == 3
assert longest_common_subsequence('abc', 'def') == 0
assert longest_common_subsequence('abc', 'abc') == 3
assert longest_common_subsequence('', 'abc') == 0
assert longest_common_subsequence('abcba', 'abcbcba') == 5
""",
    },
    {
        "id": "he_004",
        "category": "algorithm",
        "messages": [
            {"role": "system", "content": "Write only the Python function. No explanation, no markdown, no tests."},
            {"role": "user", "content": "def max_subarray_sum(nums: list[int]) -> int:\n    \"\"\"Find the contiguous subarray with the largest sum (Kadane's algorithm).\n    >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])\n    6\n    >>> max_subarray_sum([1])\n    1\n    >>> max_subarray_sum([-1, -2, -3])\n    -1\n    \"\"\""},
        ],
        "test_code": """
assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
assert max_subarray_sum([1]) == 1
assert max_subarray_sum([-1, -2, -3]) == -1
assert max_subarray_sum([5, -3, 5]) == 7
assert max_subarray_sum([-1]) == -1
""",
    },
    {
        "id": "he_005",
        "category": "data_structure",
        "messages": [
            {"role": "system", "content": "Write only the Python function. No explanation, no markdown, no tests."},
            {"role": "user", "content": "def is_valid_parentheses(s: str) -> bool:\n    \"\"\"Check if the input string has valid parentheses.\n    >>> is_valid_parentheses('()')\n    True\n    >>> is_valid_parentheses('()[]{}')\n    True\n    >>> is_valid_parentheses('(]')\n    False\n    >>> is_valid_parentheses('([)]')\n    False\n    >>> is_valid_parentheses('{[]}')\n    True\n    \"\"\""},
        ],
        "test_code": """
assert is_valid_parentheses('()') == True
assert is_valid_parentheses('()[]{}') == True
assert is_valid_parentheses('(]') == False
assert is_valid_parentheses('([)]') == False
assert is_valid_parentheses('{[]}') == True
assert is_valid_parentheses('') == True
assert is_valid_parentheses('((((') == False
""",
    },
    {
        "id": "he_006",
        "category": "data_structure",
        "messages": [
            {"role": "system", "content": "Write only the Python function. No explanation, no markdown, no tests."},
            {"role": "user", "content": "def group_anagrams(strs: list[str]) -> list[list[str]]:\n    \"\"\"Group anagrams together. Return groups in any order.\n    >>> sorted([sorted(g) for g in group_anagrams(['eat','tea','tan','ate','nat','bat'])])\n    [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]\n    >>> group_anagrams([''])\n    [['']]\n    >>> group_anagrams(['a'])\n    [['a']]\n    \"\"\""},
        ],
        "test_code": """
result = group_anagrams(['eat','tea','tan','ate','nat','bat'])
normalized = sorted([sorted(g) for g in result])
assert normalized == [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]
assert group_anagrams(['']) == [['']]
assert group_anagrams(['a']) == [['a']]
""",
    },
    {
        "id": "he_007",
        "category": "string",
        "messages": [
            {"role": "system", "content": "Write only the Python function. No explanation, no markdown, no tests."},
            {"role": "user", "content": "def roman_to_int(s: str) -> int:\n    \"\"\"Convert a Roman numeral string to an integer.\n    >>> roman_to_int('III')\n    3\n    >>> roman_to_int('LVIII')\n    58\n    >>> roman_to_int('MCMXCIV')\n    1994\n    \"\"\""},
        ],
        "test_code": """
assert roman_to_int('III') == 3
assert roman_to_int('LVIII') == 58
assert roman_to_int('MCMXCIV') == 1994
assert roman_to_int('IX') == 9
assert roman_to_int('XLII') == 42
""",
    },
    {
        "id": "he_008",
        "category": "math",
        "messages": [
            {"role": "system", "content": "Write only the Python function. No explanation, no markdown, no tests."},
            {"role": "user", "content": "def count_primes(n: int) -> int:\n    \"\"\"Count the number of prime numbers less than n (Sieve of Eratosthenes).\n    >>> count_primes(10)\n    4\n    >>> count_primes(0)\n    0\n    >>> count_primes(1)\n    0\n    >>> count_primes(2)\n    0\n    \"\"\""},
        ],
        "test_code": """
assert count_primes(10) == 4
assert count_primes(0) == 0
assert count_primes(1) == 0
assert count_primes(2) == 0
assert count_primes(100) == 25
assert count_primes(3) == 1
""",
    },
]


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_id: str
    category: str
    suite: str  # "bfcl" or "humaneval"
    target: str
    label: str
    response: str
    latency_ms: float
    passed: bool
    score: float  # 0.0-1.0
    details: str


# ── API call ────────────────────────────────────────────────────────

async def call_api(
    session: aiohttp.ClientSession,
    url: str,
    messages: list[dict],
    model: str,
    max_tokens: int = 512,
    is_ollama: bool = False,
) -> tuple[str, float]:
    t0 = time.perf_counter()
    try:
        if is_ollama:
            # Use Ollama native API to properly handle thinking/content split
            ollama_base = url.rsplit("/v1", 1)[0]
            ollama_url = f"{ollama_base}/api/chat"
            # Qwen3.5 thinking consumes token budget before content
            # 8K is enough for thinking + content in most cases
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 8192},
            }
            async with session.post(
                ollama_url, json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                raw = await resp.read()
                data = json.loads(raw.decode("utf-8", errors="replace"), strict=False)
                elapsed = (time.perf_counter() - t0) * 1000
                msg = data.get("message", {})
                content = msg.get("content", "")
                # Fallback: if content empty, extract code from thinking
                if not content and msg.get("thinking"):
                    thinking = msg["thinking"]
                    # Try to extract code block from thinking
                    if "```python" in thinking:
                        parts = thinking.split("```python")
                        # Take the last code block (likely the final answer)
                        last_block = parts[-1]
                        if "```" in last_block:
                            content = last_block.split("```")[0].strip()
                    elif "def " in thinking:
                        # Extract function definition
                        lines = thinking.split("\n")
                        code_lines = []
                        in_func = False
                        for line in lines:
                            if line.strip().startswith("def "):
                                in_func = True
                            if in_func:
                                if line.strip() and not line.startswith(" ") and not line.startswith("\t") and not line.strip().startswith("def "):
                                    break
                                code_lines.append(line)
                        content = "\n".join(code_lines).strip()
                return content, elapsed
        else:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
            }
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                raw = await resp.read()
                data = json.loads(raw.decode("utf-8", errors="replace"), strict=False)
                elapsed = (time.perf_counter() - t0) * 1000
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content, elapsed
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return f"ERROR: {e}", elapsed


# ── BFCL Scoring ────────────────────────────────────────────────────

def score_bfcl(test: dict, response: str) -> tuple[bool, float, str]:
    expected_fn = test.get("expected_function")

    # Irrelevance detection: should NOT call function
    if expected_fn is None:
        has_call = any(kw in response.lower() for kw in [
            '"name"', "'name'", "function_call", "get_weather", "book_flight",
            '{"name', "{'name",
        ])
        if not has_call:
            return True, 1.0, "Correctly avoided function call"
        else:
            return False, 0.0, "Incorrectly made function call"

    # Check if function name appears
    if expected_fn not in response:
        return False, 0.0, f"Function '{expected_fn}' not found in response"

    # Check expected args
    expected_args = test.get("expected_args", {})
    score = 0.5  # Function name found
    details = [f"Function '{expected_fn}' found"]

    for key, val in expected_args.items():
        val_str = str(val)
        if val_str in response or json.dumps(val) in response:
            score += 0.5 / len(expected_args)
            details.append(f"Arg '{key}={val}' found")
        else:
            details.append(f"Arg '{key}={val}' NOT found")

    # Parallel check
    expected_count = test.get("expected_count")
    if expected_count:
        count = response.count(expected_fn)
        if count >= expected_count:
            score = 1.0
            details.append(f"Parallel calls: {count}/{expected_count}")
        else:
            score *= count / expected_count
            details.append(f"Parallel calls: {count}/{expected_count} (incomplete)")

    passed = score >= 0.8
    return passed, min(score, 1.0), "; ".join(details)


# ── HumanEval Scoring ──────────────────────────────────────────────

def score_humaneval(test: dict, response: str) -> tuple[bool, float, str]:
    # Extract Python code from response
    code = response.strip()

    # Remove markdown fences
    if "```python" in code:
        code = code.split("```python", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]
    elif "```" in code:
        code = code.split("```", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]

    code = code.strip()

    # Get function name from test
    first_msg = test["messages"][-1]["content"]
    func_name = first_msg.split("(")[0].replace("def ", "").strip()

    # Try to execute
    test_code = test["test_code"]
    full_code = code + "\n" + test_code

    try:
        exec(full_code, {})
        return True, 1.0, "All tests passed"
    except AssertionError as e:
        return False, 0.0, f"Test failed: {e}"
    except SyntaxError as e:
        return False, 0.0, f"Syntax error: {e}"
    except Exception as e:
        return False, 0.0, f"Runtime error: {type(e).__name__}: {e}"


# ── Runner ──────────────────────────────────────────────────────────

async def run_quality_bench(
    url: str, model: str, target: str, label: str, is_ollama: bool = False,
) -> list[TestResult]:
    results = []

    async with aiohttp.ClientSession() as session:
        # Warmup
        sys.stderr.write(f"  Warming up {target} ({label})... ")
        await call_api(session, url, [{"role": "user", "content": "Hi"}], model, 16, is_ollama=is_ollama)
        sys.stderr.write("done\n")

        # BFCL tests
        sys.stderr.write(f"  Running {len(BFCL_TESTS)} BFCL tests...\n")
        for test in BFCL_TESTS:
            response, latency = await call_api(session, url, test["messages"], model, is_ollama=is_ollama)
            passed, score, details = score_bfcl(test, response)
            results.append(TestResult(
                test_id=test["id"], category=test["category"], suite="bfcl",
                target=target, label=label, response=response,
                latency_ms=latency, passed=passed, score=score, details=details,
            ))
            status = "PASS" if passed else "FAIL"
            sys.stderr.write(f"    [{status}] {test['id']} ({test['category']}): {details[:60]}\n")

        # HumanEval tests
        sys.stderr.write(f"  Running {len(HUMANEVAL_TESTS)} HumanEval tests...\n")
        for test in HUMANEVAL_TESTS:
            response, latency = await call_api(session, url, test["messages"], model, is_ollama=is_ollama)
            passed, score, details = score_humaneval(test, response)
            results.append(TestResult(
                test_id=test["id"], category=test["category"], suite="humaneval",
                target=target, label=label, response=response,
                latency_ms=latency, passed=passed, score=score, details=details,
            ))
            status = "PASS" if passed else "FAIL"
            sys.stderr.write(f"    [{status}] {test['id']} ({test['category']}): {details[:60]}\n")

    return results


# ── Display ─────────────────────────────────────────────────────────

def print_summary(results: list[TestResult]):
    # Group by suite and category
    by_suite: dict[str, list[TestResult]] = {}
    for r in results:
        by_suite.setdefault(r.suite, []).append(r)

    print()
    print("=" * 80)
    print(f"  品質ベンチマーク結果: {results[0].target} ({results[0].label})")
    print("=" * 80)

    for suite, items in sorted(by_suite.items()):
        total = len(items)
        passed = sum(1 for r in items if r.passed)
        avg_score = sum(r.score for r in items) / total if total else 0

        print(f"\n  [{suite.upper()}] {passed}/{total} passed (avg score: {avg_score:.2f})")

        by_cat: dict[str, list[TestResult]] = {}
        for r in items:
            by_cat.setdefault(r.category, []).append(r)

        for cat, cat_items in sorted(by_cat.items()):
            cat_passed = sum(1 for r in cat_items if r.passed)
            cat_score = sum(r.score for r in cat_items) / len(cat_items)
            print(f"    {cat:<20s} {cat_passed}/{len(cat_items)}  score={cat_score:.2f}")

    print()


def print_comparison(all_results: dict[str, list[TestResult]]):
    if len(all_results) < 2:
        return

    labels = list(all_results.keys())
    print()
    print("=" * 80)
    print("  品質比較: " + " vs ".join(labels))
    print("=" * 80)

    # Compare by test_id
    for suite in ["bfcl", "humaneval"]:
        print(f"\n  [{suite.upper()}]")
        print(f"  {'Test ID':<20s}", end="")
        for label in labels:
            print(f" {label:>12s}", end="")
        print(f" {'Diff':>8s}")
        print("  " + "-" * (22 + 13 * len(labels) + 8))

        test_ids = []
        for results in all_results.values():
            for r in results:
                if r.suite == suite and r.test_id not in test_ids:
                    test_ids.append(r.test_id)

        for tid in test_ids:
            scores = {}
            for label, results in all_results.items():
                for r in results:
                    if r.test_id == tid and r.suite == suite:
                        scores[label] = r
                        break

            if not scores:
                continue

            print(f"  {tid:<20s}", end="")
            score_vals = []
            for label in labels:
                r = scores.get(label)
                if r:
                    status = "PASS" if r.passed else "FAIL"
                    print(f" {status:>6s} {r.score:.2f}", end="")
                    score_vals.append(r.score)
                else:
                    print(f" {'N/A':>12s}", end="")
                    score_vals.append(0)

            if len(score_vals) >= 2:
                diff = score_vals[0] - score_vals[-1]
                marker = "=" if abs(diff) < 0.01 else ("+" if diff > 0 else "-")
                print(f" {marker}{abs(diff):.2f}")
            else:
                print()

    # Response diff (first vs last)
    if len(labels) >= 2:
        first_label = labels[0]
        last_label = labels[-1]
        print(f"\n  出力diff ({first_label} vs {last_label}):")
        for suite in ["bfcl", "humaneval"]:
            diffs_found = 0
            for r1 in all_results[first_label]:
                if r1.suite != suite:
                    continue
                for r2 in all_results[last_label]:
                    if r2.test_id == r1.test_id:
                        sim = difflib.SequenceMatcher(None, r1.response, r2.response).ratio()
                        if sim < 0.95:
                            diffs_found += 1
                            print(f"    {r1.test_id}: similarity={sim:.2f}")
                        break
            if diffs_found == 0:
                print(f"    [{suite}] All outputs >=95% similar")

    print()


def save_results(all_results: dict[str, list[TestResult]], filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename

    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": {},
    }

    for label, results in all_results.items():
        suite_data = {"bfcl": [], "humaneval": []}
        for r in results:
            entry = {
                "test_id": r.test_id,
                "category": r.category,
                "target": r.target,
                "label": label,
                "passed": r.passed,
                "score": r.score,
                "latency_ms": round(r.latency_ms, 1),
                "details": r.details,
                "response": r.response,
            }
            suite_data[r.suite].append(entry)
        data["results"][label] = suite_data

    # Summary stats
    data["summary"] = {}
    for label, results in all_results.items():
        bfcl = [r for r in results if r.suite == "bfcl"]
        he = [r for r in results if r.suite == "humaneval"]
        data["summary"][label] = {
            "bfcl_pass_rate": sum(1 for r in bfcl if r.passed) / len(bfcl) if bfcl else 0,
            "bfcl_avg_score": sum(r.score for r in bfcl) / len(bfcl) if bfcl else 0,
            "humaneval_pass_rate": sum(1 for r in he if r.passed) / len(he) if he else 0,
            "humaneval_avg_score": sum(r.score for r in he) / len(he) if he else 0,
            "total_pass_rate": sum(1 for r in results if r.passed) / len(results) if results else 0,
        }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")
    return path


def merge_results():
    """Merge all result files in quality_results/ into one analysis-ready JSON."""
    if not OUTPUT_DIR.exists():
        print(f"No results directory found: {OUTPUT_DIR}")
        return

    files = sorted(OUTPUT_DIR.glob("quality_*.json"))
    if not files:
        print("No result files found.")
        return

    merged = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": []}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            merged["runs"].append({"file": f.name, **data})

    out = OUTPUT_DIR / "merged_quality_analysis.json"
    with open(out, "w") as fh:
        json.dump(merged, fh, indent=2, ensure_ascii=False)
    print(f"Merged {len(files)} files -> {out}")

    # Print Claude analysis prompt
    print()
    print("=" * 70)
    print("  以下をClaudeに貼り付けて分析:")
    print("=" * 70)
    print(f"""
以下はHayabusa（ローカルLLM推論エンジン）とOllamaの品質ベンチマーク結果です。

<quality_results>
{json.dumps(merged, indent=2, ensure_ascii=False)[:50000]}
</quality_results>

以下の観点で分析してください：

1. BFCL（Function Calling）
   - カテゴリ別精度: simple / multiple / parallel / irrelevance
   - KV量子化モード間の品質劣化度
   - Hayabusa vs Ollamaの精度差

2. HumanEval（Code Generation）
   - カテゴリ別pass@1: basic / algorithm / data_structure / string / math
   - KV量子化による品質劣化の有無
   - Hayabusa vs Ollamaの出力diff

3. 総合評価
   - 推論速度2.5xの優位性と品質トレードオフの結論
   - KV-int8の品質劣化が許容範囲かの判定
   - 最もインパクトのあるアピール軸の提案
""")


# ── Server check ────────────────────────────────────────────────────

async def check_server(url: str) -> bool:
    base = url.rsplit("/v1", 1)[0]
    try:
        async with aiohttp.ClientSession() as session:
            for endpoint in [f"{base}/health", f"{base}/api/tags"]:
                try:
                    async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            return True
                except Exception:
                    continue
    except Exception:
        pass
    return False


# ── Main ────────────────────────────────────────────────────────────

async def main_async(args):
    if args.merge:
        merge_results()
        return

    all_results: dict[str, list[TestResult]] = {}

    targets = []
    if "hayabusa" in args.target:
        url = HAYABUSA_URL.format(port=args.hayabusa_port)
        h_label = args.label if args.label and "ollama" not in args.target else (args.label or "hayabusa")
        targets.append(("Hayabusa", url, "local", h_label, False))
    if "ollama" in args.target:
        o_label = "ollama"
        targets.append(("Ollama", OLLAMA_URL, args.model, o_label, True))

    for name, url, model, label, is_ollama in targets:
        sys.stderr.write(f"\nChecking {name}... ")
        ok = await check_server(url)
        if not ok:
            sys.stderr.write(f"UNAVAILABLE (skipping)\n")
            continue
        sys.stderr.write("OK\n")

        print(f"\n{'='*60}")
        print(f"  {name} ({label})")
        print(f"{'='*60}")

        results = await run_quality_bench(url, model, name, label, is_ollama=is_ollama)
        all_results[label] = results
        print_summary(results)

    if len(all_results) >= 2:
        print_comparison(all_results)

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    labels_str = "_vs_".join(all_results.keys())
    filename = f"quality_{labels_str}_{ts}.json"
    save_results(all_results, filename)


def main():
    parser = argparse.ArgumentParser(description="品質ベンチマーク: BFCL + HumanEval")
    parser.add_argument("--target", nargs="+", default=["hayabusa", "ollama"],
                        choices=["hayabusa", "ollama"])
    parser.add_argument("--model", default="qwen3.5:9b", help="Ollama model")
    parser.add_argument("--label", default=None,
                        help="Label for this run (e.g. kv-off, kv-int8)")
    parser.add_argument("--hayabusa-port", type=int, default=8080)
    parser.add_argument("--merge", action="store_true",
                        help="Merge all results and generate analysis prompt")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
