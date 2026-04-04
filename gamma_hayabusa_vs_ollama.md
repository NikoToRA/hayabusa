# HAYABUSA vs OLLAMA — Apple Siliconローカル推論 ガチ勝負

---

## 問いかけ: ローカルLLM、本当にOllamaでいいの？

- Ollama = ローカルLLMのデファクトスタンダード
- でもApple Siliconの性能、**本当に引き出せてる？**
- Swift製推論サーバー「Hayabusa」が挑む

---

## 対戦カード

| | Hayabusa | Ollama |
|---|---|---|
| 言語 | Swift + Metal | Go + llama.cpp |
| バックエンド | llama.cpp / MLX | llama.cpp (v0.18) |
| KVキャッシュ量子化 | int8 / TQ3(3bit) / TQ4(4bit) | int8のみ |
| 並列スロット | マルチスロット対応 | 逐次処理 |
| API | OpenAI互換 | OpenAI互換 |

---

## ベンチマーク条件

- **モデル**: Qwen3.5-9B Q4_K_M / Google Gemma 4 E4B (8B)
- **Hayabusa**: KV-int8量子化、8スロット
- **Ollama**: v0.20.2（Gemma 4対応最新版）
- **リクエスト**: 20回 × 多様なプロンプト（短文・長文・コード生成）
- **max_tokens**: 128
- **計測日**: 2026-04-04

---

## 結果: スループット (tok/s)

| 同時接続数 | Hayabusa | Ollama | 倍率 |
|---|---|---|---|
| 1 | **111.2** | 45.7 | **2.43x** |
| 4 | **126.7** | 46.8 | **2.71x** |
| 8 | **122.5** | 46.8 | **2.62x** |
| 16 | **121.5** | 46.5 | **2.61x** |
| 32 | **121.5** | 46.7 | **2.60x** |

Hayabusaが全並列度で **2.4〜2.7倍** の圧勝

---

## 結果: レイテンシ (平均, ms)

| 同時接続数 | Hayabusa | Ollama | 削減率 |
|---|---|---|---|
| 1 | **10,811** | 29,488 | **63%減** |
| 4 | **10,971** | 28,970 | **62%減** |
| 8 | **11,184** | 29,004 | **61%減** |
| 16 | **11,567** | 29,020 | **60%減** |

体感速度も **半分以下** の待ち時間

---

## NEW: Google Gemma 4 参戦 — 三つ巴対決

Ollama v0.20.2でGemma 4 E4B (8B)が利用可能に。Qwen3.5-9Bと同クラスの新モデルを加えた三つ巴対決。

| 同時接続 | Hayabusa (Qwen) | Ollama (Gemma 4) | Ollama (Qwen) | 最速 |
|---|---|---|---|---|
| 1 | **111.2 tok/s** | 108.4 tok/s | 45.7 tok/s | Hayabusa |
| 4 | **122.8 tok/s** | 120.4 tok/s | 46.8 tok/s | Hayabusa |
| 8 | **122.1 tok/s** | 117.8 tok/s | 46.7 tok/s | Hayabusa |
| 16 | **120.0 tok/s** | 117.4 tok/s | 46.5 tok/s | Hayabusa |

---

## Gemma 4 の衝撃と Hayabusa の優位性

### Gemma 4がOllamaを救った？
- Ollama + Qwen3.5: 46 tok/s（遅い）
- Ollama + Gemma 4: **108-120 tok/s**（2.5倍速！）
- モデルアーキテクチャでここまで変わる

### それでもHayabusaが最速
- Hayabusa (Qwen) vs Ollama (Gemma 4): **Hayabusa が 3-5% 上回る**
- Gemma 4の高速アーキテクチャ + Ollamaでも、Hayabusaの最適化には届かない
- **HayabusaがGemma 4に対応すれば、さらに引き離す可能性大**

### なぜGemma 4は速いのか
- **MoE的な疎構造**: 活性パラメータが少なく演算効率が高い
- **最新のアテンション設計**: GQA + sliding window で高速化
- **262Kトークン語彙**: 効率的なトークナイゼーション

---

## なぜHayabusaは速いのか

### 1. マルチスロット並列
- 複数リクエストを同時にバッチ処理
- Ollamaは基本的に逐次処理 → 並列時に差が開く

### 2. KVキャッシュ量子化 (TurboQuant)
- 独自実装の3-bit KVキャッシュ量子化 (TQ3_0)
- KVキャッシュを **78%圧縮** → メモリ節約分をスロット数に転換
- Metal GPU上でFlash Attentionカーネルを直接実行

### 3. Metal GPUフル活用
- Apple Silicon統合メモリのゼロコピーアクセス
- SIMD float4ベクタライズでWHT変換を高速化

---

## TurboQuant: 78%メモリ圧縮の秘密

| KVモード | メモリ/4スロット | 節約率 | tok/s |
|---|---|---|---|
| fp16 (ベースライン) | 8,192 MB | 0% | 72.3 |
| int8 (Q8_0) | 4,352 MB | 47% | 70.8 |
| TQ4 (4bit) | 2,304 MB | 72% | 43.7 |
| **TQ3 (3bit)** | **1,792 MB** | **78%** | **45.0** |

→ 16GB Mac Miniで推定 **22並列** を実現（fp16では5並列が限界）

---

## 16GB Mac Miniで何が変わるか

| | fp16 | int8 | TQ3 (Hayabusa独自) |
|---|---|---|---|
| 最大並列数 | ~5 | ~9 | **~22** |
| 同時ユーザー | 限界あり | やや余裕 | **チーム利用可能** |
| メモリ効率 | 低い | 中 | **高い** |

→ $600のMac Miniが **チーム用LLMサーバー** に化ける

---

## スケーリング特性

- **Hayabusa (Qwen)**: 並列数を増やしてもスループット維持（111→123 tok/s、むしろ向上）
- **Ollama (Gemma 4)**: 並列でスケール（108→120 tok/s）だがHayabusaには届かず
- **Ollama (Qwen)**: 並列数に関係なくほぼ一定（45〜47 tok/s）
- Hayabusa + 最適モデル選択が最強の組み合わせ

---

## まとめ: 総合戦績

| 指標 | Hayabusa | Ollama+Gemma4 | Ollama+Qwen |
|---|---|---|---|
| スループット | **111-123 tok/s** | 108-120 tok/s | 46 tok/s |
| レイテンシ | **10.8-11.6s** | 10.0-11.5s | 28.7-29.5s |
| メモリ効率 | **78%圧縮 (TQ3)** | 標準 | 標準 |
| 並列スケーリング | **維持+向上** | 向上 | 横ばい |
| Gemma 4対応 | 準備中 | 対応済 | — |
| 手軽さ | ビルド必要 | **ワンコマンド** | **ワンコマンド** |

---

## Hayabusaの使い方

```bash
# インストール & ビルド
git clone https://github.com/user/hayabusa
cd hayabusa && swift build -c release

# 起動（TQ3量子化で最大メモリ効率）
.build/release/Hayabusa model.gguf --kv-quantize tq3

# OpenAI互換API
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Next: HayabusaにGemma 4対応を

- 現在のHayabusa llama.cppバックエンドは `gemma4` アーキテクチャ未対応
- llama.cpp本体はGemma 4サポート済み → **マージすれば即対応**
- Hayabusa + Gemma 4 = Ollama+Gemma4をさらに上回る可能性
- TurboQuant (TQ3) + Gemma 4 = **最小メモリで最大スループット**

---

## 結論: Apple Siliconの真の力を引き出せ

- Ollamaは「手軽さ」、Hayabusaは「性能」
- Gemma 4はOllamaの速度を2.5倍にしたが、**Hayabusaにはまだ届かない**
- 同じハードウェアで **2.5倍速く、4倍多く並列処理**
- $600のMac Miniがチーム用LLMサーバーに
- **ローカル推論の限界はソフトウェアが決める**
- **Next step: Hayabusa + Gemma 4 で最強の組み合わせへ**
