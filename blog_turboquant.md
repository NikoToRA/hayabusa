# TurboQuant: 3-bit KVキャッシュ量子化をllama.cppに実装してMetal GPUで動かした話

## TL;DR

LLMの推論サーバー「Hayabusa」に、論文ベースの3-bit KVキャッシュ量子化（TurboQuant TQ3_0）を実装した。Walsh-Hadamard変換 + Lloyd-Maxコードブックで**KVキャッシュを78%圧縮**し、Apple Silicon Metal GPUのFlash Attentionカーネルまで書いて**45 tok/s**を達成。16GB Mac Miniで推定22並列を実現した。

## 背景

### Hayabusaとは

Hayabusa は Swift で書かれた LLM 推論サーバー。llama.cpp と MLX をバックエンドに持ち、OpenAI互換APIを提供する。マルチスロットのKVキャッシュで並列リクエストを処理できる。

### 問題: KVキャッシュがメモリを食い尽くす

LLMの推論で最もメモリを消費するのがKVキャッシュ。Qwen3.5-9Bの場合：

- 32層 × 16ヘッド × 256次元 × 4096コンテキスト × 2(K+V) × 2バイト(fp16) = **約2GB/スロット**
- 16GB Mac Miniでは4-5スロットが限界

並列数を増やすにはKVキャッシュの圧縮が必要。

### 既存手法: Q8_0 (int8量子化)

llama.cppには `--ctk q8_0 --ctv q8_0` でKVキャッシュをint8量子化する機能がある。32要素ごとにスケールファクター（fp16）を持ち、各要素を8bitで保存。メモリ47%削減で速度劣化はほぼなし。

ただし47%では足りない。**3-4bit**まで落とせれば並列数が劇的に増える。

## TurboQuant TQ3_0 の仕組み

参考にしたのは [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)（Zandieh et al.）と、[Aaryan-Kapoor氏のllama.cpp実装](https://github.com/Aaryan-Kapoor/llama.cpp/tree/turboquant-tq3_0)。

### アルゴリズム概要

1. **RMS正規化**: 32要素のブロックのRMSをスケールファクターとして保存（fp16, 2バイト）、単位分散に正規化
2. **Randomized Hadamard Transform (RHT)**: 決定性の符号反転 + Walsh-Hadamard変換で情報を拡散
3. **Lloyd-Max 8レベル量子化**: 正規分布に最適な8個のセントロイドで各要素を3bitに量子化
4. **3bitパッキング**: 8個のインデックスを3バイトにパック

### なぜWHTが必要なのか

ナイーブに各要素を個別に量子化すると、外れ値がある場合に大きな誤差が生じる。WHTは直交変換で、情報を全座標に均一に分散させる。変換後のデータは正規分布に近くなり、Lloyd-Max量子化器の理論的最適性が活きる。

### ストレージ効率

```
block_tq3_0 = {
    fp16 scale;         // 2バイト
    uint8_t qs[12];     // 32 × 3bit = 96bit = 12バイト
}
// 合計: 14バイト / 32要素 = 3.5 bits per value
```

fp16の2バイト/要素と比べて**78%圧縮**。

## 実装

### Step 1: llama.cpp に TQ3_0/TQ4_0 を追加

ggml の型システムに新しい量子化タイプを登録：

```c
// ggml.h
GGML_TYPE_TQ3_0 = 41,  // 3-bit TurboQuant
GGML_TYPE_TQ4_0 = 42,  // 4-bit TurboQuant

// ggml-common.h
#define QK_TQ3_0 32
typedef struct {
    ggml_half d;
    uint8_t qs[QK_TQ3_0 * 3 / 8];  // 12バイト
} block_tq3_0;
```

量子化・逆量子化の実装（`ggml-quants.c`、約200行）、CPU vec_dotカーネル、型トレイツの登録。

### Step 2: Metal Flash Attention カーネル — 最大の挑戦

ここからが本番。KVキャッシュをGPUに置くには、Metal Shading Languageで以下が必要：

1. **SET_ROWS**: f32 → TQ3_0（KV書き込み時の量子化）
2. **GET_ROWS**: TQ3_0 → f32（KV読み出し時の逆量子化）
3. **Flash Attention**: TQ3_0のK/Vで直接Attention計算

#### 最初の壁: `offload_kqv = false` 地獄

Metal カーネルなしでは KV をCPUに置くしかない。しかし `offload_kqv = false` にすると **Q×K^T、softmax、×V の全Attention計算がCPUフォールバック**する。結果:

```
f16:  72.3 tok/s (GPU)
tq3:  22.6 tok/s (CPU fallback)  ← 3.2倍遅い
```

#### Flash Attention テンプレートシステムの理解

llama.cppのMetal FA カーネルは巨大なC++テンプレートで構成されている：

```metal
template<
    typename q_t, typename q4_t, typename q8x8_t,  // Query types
    typename k_t, typename k4x4_t, typename k8x8_t, // Key types
    typename v_t, typename v4x4_t, typename v8x8_t, // Value types
    // ... 20+ template parameters
    typename kd4x4_t, short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK, short DV>
kernel void kernel_flash_attn_ext(...) { ... }
```

テンプレートにはdequantize関数ポインタが渡される。TQ3_0用のdequantize関数を書いてインスタンス化すれば動く。

#### dequantize関数の実装

Metal Shading Languageで32要素のWHT逆変換を実装：

```metal
template <typename type4x4>
void dequantize_tq3_0(device const block_tq3_0 * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    float tmp[32];

    // 1. 3bitインデックスをアンパックしてセントロイドをルックアップ
    for (int g = 0; g < 4; ++g) {
        device const uint8_t * qp = xb->qs + g * 3;
        // 3バイトから8個の3bitインデックスを展開
        // ... bit manipulation ...
        for (int j = 0; j < 8; ++j)
            tmp[g*8 + j] = tq3_centroids_metal[idx[j]];
    }

    // 2. WHT逆変換
    wht_32_metal(tmp);

    // 3. 符号反転を戻してスケーリング
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            reg[i][j] = tmp[il*16 + i*4 + j] * tq_sign(il*16+i*4+j) * d;
}
```

14種のhead dimensionに対してテンプレートをインスタンス化（Qwen3.5-9Bはdk=256, dv=256）：

```metal
template [[host_name("kernel_flash_attn_ext_tq3_0_dk256_dv256")]]
kernel flash_attn_ext_t kernel_flash_attn_ext<
    FA_TYPES, block_tq3_0, 2, dequantize_tq3_0,
    block_tq3_0, 2, dequantize_tq3_0, 256, 256>;
```

#### Metal コンパイルとの格闘

Metal Shading Languageには独自の制約がある：

- **アドレス空間修飾子必須**: `const uint8_t * idx` → `thread const uint8_t * idx`
- **`type4::value_type` 使用不可**: Metal vectorの型メンバアクセスは非対応
- **embedded metallib**: ソース変更後はcmake clean buildが必須（プリコンパイル済みライブラリ）

### Step 3: SIMD最適化 — float4ベクタライズ

FA カーネルが動いた時点で **30 tok/s**。まだf16の42%。

ボトルネックはWHT。32要素の5段バタフライ変換を毎ブロック実行している。

#### 最適化1: 事前計算テーブル

符号反転パターンを毎回計算するのをやめ、定数テーブルに：

```metal
constant float4 tq_signs_v4[8] = {
    float4(+1,-1,+1,-1), float4(+1,+1,-1,+1),
    float4(-1,-1,+1,-1), float4(+1,+1,-1,+1),
    float4(-1,-1,+1,-1), float4(+1,-1,-1,+1),
    float4(-1,+1,+1,-1), float4(+1,-1,-1,+1),
};
```

#### 最適化2: float4ベクタライズWHT

ステージ3-5（stride ≥ 4）はfloat4で4要素同時に処理：

```metal
thread float4 * v = (thread float4 *)x;

// Stage 3: stride=4
for (int i = 0; i < 4; i++) {
    float4 a = v[2*i], b = v[2*i+1];
    v[2*i] = a + b; v[2*i+1] = a - b;
}

// Stage 4: stride=8
float4 a0=v[0],a1=v[1],b0=v[2],b1=v[3];
v[0]=a0+b0; v[1]=a1+b1; v[2]=a0-b0; v[3]=a1-b1;
// ...

// Stage 5: stride=16
// 上半分と下半分のfloat4を加減算
```

#### 最適化3: 符号反転+スケールの融合

逆量子化の最後のステップで、符号反転とスケーリングをfloat4の1回の乗算に融合：

```metal
float4 sd = float4(d);
for (int i = 0; i < 4; i++)
    v_out[i] = v[base+i] * tq_signs_v4[base+i] * sd;
```

## ベンチマーク結果

Qwen3.5-9B Q4_K_M、Mac Studio M3 Ultra、4スロット × 4096コンテキスト：

| モード | KVメモリ | 節約率 | tok/s | f16比 |
|--------|---------|--------|-------|-------|
| **f16** (ベースライン) | 8,192 MB | 0% | **72.3** | 100% |
| **int8** (Q8_0) | 4,352 MB | 47% | **70.8** | 98% |
| **tq4** (TQ4_0, 4bit) | 2,304 MB | 72% | **43.7** | 60% |
| **tq3** (TQ3_0, 3bit) | 1,792 MB | **78%** | **45.0** | **62%** |

### 最適化の軌跡

```
CPUフォールバック:    22.6 tok/s  (offload_kqv=false)
Metal FA カーネル:    30.0 tok/s  (+33%)
SIMD float4最適化:    45.0 tok/s  (+50%, 合計2倍)
```

### 16GB Mac Mini での推定並列数

| モード | KV/スロット | 推定並列数 |
|--------|-----------|-----------|
| f16    | ~2 GB     | ~5        |
| int8   | ~1 GB     | ~9        |
| tq4    | ~576 MB   | ~17       |
| **tq3** | ~448 MB  | **~22**   |

## TQ4_0 も同時実装

同じアーキテクチャで4-bit版（TQ4_0）も実装。Lloyd-Max 16レベルコードブック、4bitニブルパッキング。メモリ効率はtq3に劣るが、実装はシンプル。

## 学んだこと

### Metal Shading Language の罠
- ポインタには必ず `device`/`thread`/`threadgroup` 修飾子が必要
- `float4::value_type` のような型メンバアクセスは使えない
- embedded metallibはソースから自動生成されるが、cmake clean buildが必要

### llama.cppへの型追加は大仕事
- `ggml.h`, `ggml-common.h`, `ggml-quants.c`, `ggml.c`, `ggml-cpu/quants.c`, `ggml-cpu/ggml-cpu.c`, `ggml-cpu/ops.cpp` の7ファイル
- さらに各CPUアーキテクチャ（ARM, x86, RISC-V）のディスパッチコード
- Metal: `ggml-metal.metal`, `ggml-metal-device.m` のsupports_op追加

### Flash Attention のKV型制約
- llama.cppではK/Vが同じ型でないとFAが使えない
- V cacheの量子化にはFlash Attentionが必須（`V cache quantization requires flash_attn` エラー）
- つまりFAカーネルなしではV cache量子化不可 → FAカーネル実装が先決

### パフォーマンスのボトルネック
- WHT変換は本質的にO(n log n)の演算（5段 × 16ペア = 80回の加減算）
- fp16/q8_0のdequantizeは1回の乗算で済む → 構造的に不利
- float4ベクタライズで実効演算数を1/4に削減して改善

## コード

実装は [hayabusa](https://github.com/tanimurahifukka/hayabusa) で公開。

```bash
# TurboQuant 3-bit KVキャッシュで起動
.build/release/Hayabusa model.gguf --kv-quantize tq3

# 4-bit版
.build/release/Hayabusa model.gguf --kv-quantize tq4
```

## 今後

- **Metal simdgroup_shuffle**: 32スレッドで協調的にWHT実行（現在はスレッド内float4）
- **Upstream PR**: llama.cpp本体への貢献
- **ARM NEON/SVE最適化**: CPU側のvec_dot高速化
