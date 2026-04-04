#!/usr/bin/env python3
"""SOAP品質ベンチマーク: Gemma 4 vs Qwen3.5 — 医療SOAP記録の精度比較.

臨床シナリオからSOAP形式のカルテ記録を生成し、
構造化・正確性・日本語品質を比較評価する。

Usage:
    # Hayabusa (Gemma4 Q8) on port 8080, Ollama (Qwen3.5:9b) on 11434
    python scripts/bench_soap_quality.py

    # Hayabusa vs Ollama both Gemma4
    python scripts/bench_soap_quality.py --ollama-model gemma4:e4b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp

HAYABUSA_URL = "http://localhost:{port}/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "bench_soap_quality.json"

# ── SOAP System Prompt ──────────────────────────────────────────────

SOAP_SYSTEM = """あなたは経験豊富な内科医です。以下の患者情報からSOAPノートを作成してください。

## フォーマット
必ず以下の4セクションに分けて記載してください：

### S（Subjective / 主観的情報）
患者の訴え、症状の経過、既往歴、生活習慣など

### O（Objective / 客観的情報）
バイタルサイン、身体所見、検査結果など

### A（Assessment / 評価）
鑑別診断、最も考えられる診断名、根拠

### P（Plan / 計画）
検査オーダー、処方、生活指導、フォローアップ計画

## ルール
- 各セクションは必ず「### S」「### O」「### A」「### P」で始めること
- 日本語で記載すること
- 簡潔かつ正確に記載すること
- 根拠のない推測は避けること
"""

# ── Clinical Scenarios ──────────────────────────────────────────────

SCENARIOS = [
    {
        "id": "soap_01_hypertension",
        "name": "高血圧初診",
        "prompt": "52歳男性。会社の健康診断で血圧160/95mmHgを指摘され来院。自覚症状なし。喫煙20本/日×30年。飲酒ビール500ml/日。父が脳卒中で死亡。BMI 27.3。来院時血圧 158/92mmHg、脈拍78/分整。心音正常、頸部血管雑音なし。採血：LDL 158mg/dL、HbA1c 5.8%、Cr 0.9mg/dL、K 4.2mEq/L。尿蛋白(-)。心電図：左室肥大なし。",
        "required_sections": ["S", "O", "A", "P"],
        "key_findings": {
            "S": ["自覚症状なし", "喫煙", "飲酒", "家族歴"],
            "O": ["158/92", "LDL 158", "HbA1c 5.8"],
            "A": ["高血圧", "脂質異常"],
            "P": ["生活指導", "禁煙", "減塩"],
        },
    },
    {
        "id": "soap_02_diabetes",
        "name": "糖尿病フォロー",
        "prompt": "68歳女性。2型糖尿病で通院中。メトホルミン500mg×2/日服用中。最近口渇と頻尿が増悪。体重が1ヶ月で2kg減少。空腹時血糖 198mg/dL、HbA1c 8.9%（前回7.2%）。BMI 24.1。血圧 138/82mmHg。眼底検査：単純網膜症あり。足背動脈触知良好。アキレス腱反射やや低下。尿アルブミン/Cr比 45mg/gCr。eGFR 62mL/min。",
        "required_sections": ["S", "O", "A", "P"],
        "key_findings": {
            "S": ["口渇", "頻尿", "体重減少"],
            "O": ["HbA1c 8.9", "網膜症", "尿アルブミン", "eGFR 62"],
            "A": ["コントロール不良", "合併症"],
            "P": ["薬剤追加", "眼科"],
        },
    },
    {
        "id": "soap_03_chest_pain",
        "name": "胸痛精査",
        "prompt": "45歳男性。3日前から労作時の前胸部圧迫感。安静で数分で改善。階段を上ると再現する。冷汗なし、放散痛なし。高血圧・脂質異常症の既往あり（未治療）。喫煙15本/日。来院時：血圧 148/90mmHg、脈拍 82/分整、SpO2 98%。胸部聴診正常。安静時心電図：ST変化なし。トロポニンI陰性。胸部X線：心拡大なし、肺野清。",
        "required_sections": ["S", "O", "A", "P"],
        "key_findings": {
            "S": ["労作時", "圧迫感", "安静で改善"],
            "O": ["ST変化なし", "トロポニン陰性"],
            "A": ["労作性狭心症", "安定狭心症"],
            "P": ["負荷試験", "冠動脈", "ニトロ"],
        },
    },
    {
        "id": "soap_04_pneumonia",
        "name": "市中肺炎",
        "prompt": "73歳女性。3日前から38.5度の発熱、咳嗽、膿性痰。食欲低下あり。ADLは自立。COPD既往なし。来院時：体温38.2度、血圧 128/76mmHg、脈拍96/分、SpO2 93%(room air)。右下肺野にcoarse crackles聴取。WBC 14,200、CRP 12.8mg/dL、PCT 0.8ng/mL。胸部X線：右下肺野にair bronchogramを伴う浸潤影。A-DROPスコア：年齢1点、SpO2 1点、計2点。",
        "required_sections": ["S", "O", "A", "P"],
        "key_findings": {
            "S": ["発熱", "咳嗽", "膿性痰"],
            "O": ["SpO2 93", "crackles", "CRP 12.8", "浸潤影"],
            "A": ["肺炎", "市中肺炎"],
            "P": ["抗菌薬", "入院", "酸素"],
        },
    },
    {
        "id": "soap_05_depression",
        "name": "うつ病スクリーニング",
        "prompt": "38歳女性。2ヶ月前から不眠、倦怠感、食欲低下。仕事への集中力低下を自覚。趣味だった読書も楽しめない。希死念慮は否定。残業月80時間。PHQ-9スコア15点。来院時：表情やや乏しい、涙ぐむ場面あり。バイタル正常。甲状腺機能正常（TSH 2.1、FT4 1.2）。貧血なし（Hb 13.2）。",
        "required_sections": ["S", "O", "A", "P"],
        "key_findings": {
            "S": ["不眠", "倦怠感", "集中力低下", "希死念慮否定"],
            "O": ["PHQ-9 15", "甲状腺正常"],
            "A": ["うつ病", "中等度"],
            "P": ["SSRI", "休職", "精神科"],
        },
    },
]


# ── SOAP Evaluator ──────────────────────────────────────────────────

@dataclass
class SOAPScore:
    scenario_id: str
    target: str
    model: str
    structure_score: float     # S/O/A/P全セクション存在 (0-1)
    completeness_score: float  # key findingsのカバー率 (0-1)
    section_scores: dict       # セクション別スコア
    total_score: float         # 総合 (0-1)
    latency_ms: float
    response: str
    details: str


def evaluate_soap(response: str, scenario: dict, target: str, model: str, latency_ms: float) -> SOAPScore:
    text = response.strip()
    sections_found = {}
    section_labels = {"S": "S", "O": "O", "A": "A", "P": "P"}

    # セクション抽出（### S, ### O, ### A, ### P）
    for label in section_labels:
        patterns = [
            rf"###\s*{label}[（(].*?[)）]?\s*\n(.*?)(?=###\s*[SOAP]|$)",
            rf"###\s*{label}\s*\n(.*?)(?=###\s*[SOAP]|$)",
            rf"\*\*{label}[（(].*?[)）]?\*\*\s*\n(.*?)(?=\*\*[SOAP]|$)",
            rf"^{label}[.。:：]\s*(.*?)(?=^[SOAP][.。:：]|$)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL | re.MULTILINE)
            if m:
                sections_found[label] = m.group(1).strip()
                break

    # 構造スコア
    required = scenario["required_sections"]
    structure_score = sum(1 for s in required if s in sections_found) / len(required)

    # key findings カバー率
    section_scores = {}
    total_hits = 0
    total_expected = 0

    for section, keywords in scenario["key_findings"].items():
        section_text = sections_found.get(section, "")
        # セクション内にない場合は全文から探す（セクション分割が曖昧な場合の救済）
        search_text = section_text if section_text else text
        hits = sum(1 for kw in keywords if kw.lower() in search_text.lower())
        section_scores[section] = hits / len(keywords) if keywords else 1.0
        total_hits += hits
        total_expected += len(keywords)

    completeness_score = total_hits / total_expected if total_expected > 0 else 0

    # 総合スコア（構造40% + カバー率60%）
    total_score = structure_score * 0.4 + completeness_score * 0.6

    details_parts = []
    details_parts.append(f"Structure: {sum(1 for s in required if s in sections_found)}/{len(required)}")
    details_parts.append(f"Coverage: {total_hits}/{total_expected}")
    for s, score in section_scores.items():
        details_parts.append(f"  {s}: {score:.0%}")

    return SOAPScore(
        scenario_id=scenario["id"],
        target=target,
        model=model,
        structure_score=structure_score,
        completeness_score=completeness_score,
        section_scores=section_scores,
        total_score=total_score,
        latency_ms=latency_ms,
        response=text,
        details="; ".join(details_parts),
    )


# ── API Call ────────────────────────────────────────────────────────

async def call_api(session, url, model, messages, max_tokens=1024):
    # Use non-streaming for reliable content extraction (Gemma4 reasoning field, etc.)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
            raw = await resp.read()
            elapsed = (time.perf_counter() - t0) * 1000
            try:
                data = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                # Hayabusa may return text/plain
                text = raw.decode("utf-8", errors="replace")
                return text.strip(), elapsed

            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            # Gemma 4 on Ollama puts output in "reasoning" field when thinking
            reasoning = msg.get("reasoning", "")

            # Use content if available, else reasoning
            full = content if content.strip() else reasoning
            # Strip think tags (Qwen3.5)
            full = re.sub(r"<think>.*?</think>\s*", "", full, flags=re.DOTALL)
            return full.strip(), elapsed
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return f"ERROR: {e}", elapsed


# ── Main ────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="SOAP品質ベンチマーク: Gemma 4 vs Qwen3.5")
    parser.add_argument("--hayabusa-port", type=int, default=8080)
    parser.add_argument("--hayabusa-model", default="local")
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    hayabusa_url = HAYABUSA_URL.format(port=args.hayabusa_port)
    ollama_url = OLLAMA_URL

    targets = []

    # Check servers
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"http://localhost:{args.hayabusa_port}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    targets.append(("Hayabusa(Gemma4)", hayabusa_url, args.hayabusa_model))
        except:
            pass
        try:
            async with session.get(f"http://localhost:11434/api/version", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    targets.append((f"Ollama({args.ollama_model})", ollama_url, args.ollama_model))
        except:
            pass

    if not targets:
        print("No servers available"); return

    print()
    print("=" * 80)
    print("  SOAP品質ベンチマーク: Gemma 4 vs Qwen3.5")
    print(f"  Targets: {', '.join(t[0] for t in targets)}")
    print(f"  Scenarios: {len(SCENARIOS)}  max_tokens={args.max_tokens}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    all_scores: list[SOAPScore] = []

    async with aiohttp.ClientSession() as session:
        for scenario in SCENARIOS:
            print(f"--- {scenario['name']} ({scenario['id']}) ---")
            messages = [
                {"role": "system", "content": SOAP_SYSTEM},
                {"role": "user", "content": scenario["prompt"]},
            ]

            for target_name, url, model in targets:
                response, latency = await call_api(session, url, model, messages, args.max_tokens)
                score = evaluate_soap(response, scenario, target_name, model, latency)
                all_scores.append(score)
                print(f"  {target_name:25s}  score={score.total_score:.2f}  struct={score.structure_score:.0%}  coverage={score.completeness_score:.0%}  {latency:.0f}ms")
            print()

    # ── Summary ──
    print()
    print("=" * 80)
    print("  SOAP品質スコア集計")
    print("=" * 80)

    by_target: dict[str, list[SOAPScore]] = {}
    for s in all_scores:
        by_target.setdefault(s.target, []).append(s)

    hdr = f"{'Target':<28} {'Avg Score':>9} {'Structure':>9} {'Coverage':>9} {'Avg ms':>8}"
    print(hdr)
    print("-" * len(hdr))

    target_avgs = {}
    for target, scores in by_target.items():
        avg_total = sum(s.total_score for s in scores) / len(scores)
        avg_struct = sum(s.structure_score for s in scores) / len(scores)
        avg_cover = sum(s.completeness_score for s in scores) / len(scores)
        avg_lat = sum(s.latency_ms for s in scores) / len(scores)
        target_avgs[target] = avg_total
        print(f"{target:<28} {avg_total:>9.2f} {avg_struct:>8.0%} {avg_cover:>9.0%} {avg_lat:>8.0f}")

    print()

    # Per-scenario comparison
    print("--- シナリオ別比較 ---")
    for scenario in SCENARIOS:
        print(f"\n  {scenario['name']}:")
        for target, scores in by_target.items():
            s = next((x for x in scores if x.scenario_id == scenario["id"]), None)
            if s:
                sec = " ".join(f"{k}:{v:.0%}" for k, v in s.section_scores.items())
                print(f"    {target:25s} total={s.total_score:.2f}  {sec}")

    # Winner
    print()
    if len(target_avgs) >= 2:
        winner = max(target_avgs, key=target_avgs.get)
        print(f"  VERDICT: {winner} が SOAP 品質で優勝 (avg={target_avgs[winner]:.2f})")
    print()

    # Save
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scenarios": len(SCENARIOS),
        "targets": {t[0]: t[2] for t in targets},
        "results": [asdict(s) for s in all_scores],
        "summary": {t: {"avg_score": sum(s.total_score for s in ss)/len(ss),
                        "avg_structure": sum(s.structure_score for s in ss)/len(ss),
                        "avg_coverage": sum(s.completeness_score for s in ss)/len(ss)}
                    for t, ss in by_target.items()},
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
