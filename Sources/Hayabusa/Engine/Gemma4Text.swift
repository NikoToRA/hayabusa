// Gemma4Text.swift — Gemma 4 text model for MLX backend
// Based on Gemma3Text.swift from mlx-swift-lm, adapted for Gemma 4 architecture.
//
// Key differences from Gemma 3:
// - layer_types array (instead of sliding_window_pattern int)
// - Dual head dimensions: global_head_dim (512) for full attention, head_dim (256) for SWA
// - Shared KV layers (num_kv_shared_layers)
// - Proportional RoPE for full attention (partial_rotary_factor)
// - Final logit softcapping
// - Optional MoE (enable_moe_block) — supported for 26B-A4B
// - Per-layer embeddings (hidden_size_per_layer_input) — simplified

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let headDim: Int
    let globalHeadDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let kvHeads: Int
    let globalKVHeads: Int?
    let slidingWindow: Int
    let layerTypes: [String]
    let maxPositionEmbeddings: Int
    let finalLogitSoftcapping: Float?
    let numKVSharedLayers: Int
    let tieWordEmbeddings: Bool
    let ropeParameters: RopeParameters?

    // MoE fields (for 26B-A4B, not used by E4B)
    let enableMoeBlock: Bool
    let numExperts: Int?
    let topKExperts: Int?
    let moeIntermediateSize: Int?

    // k_eq_v: full attention layers share k_proj output as values (26B/31B)
    let attentionKEqV: Bool

    struct RopeParameters: Codable {
        let fullAttention: AttentionRopeConfig?
        let slidingAttention: AttentionRopeConfig?

        enum CodingKeys: String, CodingKey {
            case fullAttention = "full_attention"
            case slidingAttention = "sliding_attention"
        }
    }

    struct AttentionRopeConfig: Codable {
        let ropeTheta: Float?
        let ropeType: String?
        let partialRotaryFactor: Float?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
            case partialRotaryFactor = "partial_rotary_factor"
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case maxPositionEmbeddings = "max_position_embeddings"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case numKVSharedLayers = "num_kv_shared_layers"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
        case enableMoeBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case attentionKEqV = "attention_k_eq_v"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)
        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2560
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 42
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 10240
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        globalKVHeads = try container.decodeIfPresent(Int.self, forKey: .globalKVHeads)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "sliding_attention", count: hiddenLayers)
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        finalLogitSoftcapping = try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        numKVSharedLayers = try container.decodeIfPresent(Int.self, forKey: .numKVSharedLayers) ?? 0
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        ropeParameters = try container.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        enableMoeBlock = try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        attentionKEqV = try container.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
    }

    func isFullAttention(layer: Int) -> Bool {
        guard layer < layerTypes.count else { return false }
        return layerTypes[layer] == "full_attention"
    }

    /// Index of the KV source layer for shared KV cache
    func kvSourceLayer(for layer: Int) -> Int? {
        guard numKVSharedLayers > 0 else { return nil }
        let kvFromStart = hiddenLayers - numKVSharedLayers
        if layer >= kvFromStart {
            // This layer shares KV with an earlier layer
            return layer - kvFromStart
        }
        return nil
    }
}

// MARK: - RMSNorm (reuse Gemma pattern)

class Gemma4RMSNorm: Module, UnaryLayer {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1.0 + weight, eps: eps)
    }
}

// MARK: - Attention

class Gemma4Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let isFullAttention: Bool
    let slidingWindow: Int
    let scale: Float
    let useKEqV: Bool  // k_eq_v: layers without v_proj use k_proj output as values

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear?
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: Gemma4RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma4RMSNorm
    @ModuleInfo(key: "v_norm") var valueNorm: Gemma4RMSNormNoScale?
    @ModuleInfo var rope: RoPE

    init(_ config: Gemma4TextConfiguration, layerIdx: Int, useKEqV: Bool = false) {
        self.isFullAttention = config.isFullAttention(layer: layerIdx)
        self.slidingWindow = config.slidingWindow
        self.useKEqV = useKEqV

        let effectiveHeadDim = isFullAttention ? config.globalHeadDim : config.headDim
        var effectiveKVHeads = isFullAttention
            ? (config.globalKVHeads ?? config.kvHeads)
            : config.kvHeads

        // k_eq_v layers use global KV heads even for sliding attention
        if useKEqV, let globalKV = config.globalKVHeads {
            effectiveKVHeads = globalKV
        }

        self.nHeads = config.attentionHeads
        self.nKVHeads = effectiveKVHeads
        self.headDim = effectiveHeadDim
        self.scale = 1.0

        let dim = config.hiddenSize
        self._queryProj.wrappedValue = Linear(dim, nHeads * effectiveHeadDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, effectiveKVHeads * effectiveHeadDim, bias: false)
        if !useKEqV {
            self._valueProj.wrappedValue = Linear(dim, effectiveKVHeads * effectiveHeadDim, bias: false)
        }
        self._outputProj.wrappedValue = Linear(nHeads * effectiveHeadDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma4RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma4RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        // v_norm: used in 26B for normalizing values (no learnable weight)
        if config.enableMoeBlock {
            self._valueNorm.wrappedValue = Gemma4RMSNormNoScale(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        }

        // RoPE configuration
        if isFullAttention {
            let partialFactor = config.ropeParameters?.fullAttention?.partialRotaryFactor ?? 0.25
            let ropeTheta = config.ropeParameters?.fullAttention?.ropeTheta ?? 1_000_000.0
            let ropeDims = Int(Float(effectiveHeadDim) * partialFactor)
            self.rope = RoPE(dimensions: ropeDims, traditional: false, base: ropeTheta)
        } else {
            let ropeTheta = config.ropeParameters?.slidingAttention?.ropeTheta ?? 10_000.0
            self.rope = RoPE(dimensions: effectiveHeadDim, traditional: false, base: ropeTheta)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x)
        var keys = keyProj(x)

        // k_eq_v: use raw k_proj output (before k_norm) as values
        var values: MLXArray
        if useKEqV {
            values = keys  // raw k before norm
        } else if let vProj = valueProj {
            values = vProj(x)
        } else {
            values = keys
        }

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = queryNorm(queries)
        keys = keyNorm(keys)
        if let vNorm = valueNorm {
            values = vNorm(values)
        }

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outputProj(output)
    }
}

// MARK: - MLP (GeGLU)

class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE Router (26B-A4B)

class Gemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * rsqrt(mean(x * x, axis: -1, keepDims: true) + eps)
    }
}

class Gemma4Router: Module {
    let numExperts: Int
    let topK: Int
    let rootSize: Float

    @ModuleInfo var norm: Gemma4RMSNormNoScale
    @ModuleInfo var proj: Linear
    @ModuleInfo var scale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(_ config: Gemma4TextConfiguration) {
        let ne = config.numExperts ?? 128
        let tk = config.topKExperts ?? 8
        self.numExperts = ne
        self.topK = tk
        self.rootSize = pow(Float(config.hiddenSize), -0.5)

        self._norm.wrappedValue = Gemma4RMSNormNoScale(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._proj.wrappedValue = Linear(config.hiddenSize, ne, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([ne])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var h = norm(x)
        h = h * rootSize
        h = h * scale

        let expertScores = proj(h)
        let routerProbs = softmax(expertScores, axis: -1)

        let topKIndices = argPartition(-expertScores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var topKWeights = takeAlong(routerProbs, topKIndices, axis: -1)
        topKWeights = topKWeights / sum(topKWeights, axis: -1, keepDims: true)
        topKWeights = topKWeights * perExpertScale[topKIndices]

        return (topKIndices, topKWeights)
    }
}

// MARK: - MoE Experts (26B-A4B)

class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(_ config: Gemma4TextConfiguration) {
        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize ?? config.intermediateSize,
            numExperts: config.numExperts ?? 128,
            activation: { geluApproximate($0) },
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2))
        let K = indices.dim(-1)

        let xFlat = x.reshaped(B * S, H)
        let indicesFlat = indices.reshaped(B * S, K)

        let expertOut = switchGLU(xFlat, indicesFlat)

        let w = weights.reshaped(B * S, K)[.ellipsis, .newAxis]
        return (expertOut * w).sum(axis: -2).reshaped(B, S, H)
    }
}

// MARK: - Transformer Block

class Gemma4TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma4RMSNorm

    // MoE components (26B-A4B: enable_moe_block = true)
    let enableMoE: Bool
    @ModuleInfo var router: Gemma4Router?
    @ModuleInfo var experts: Gemma4Experts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: Gemma4RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: Gemma4RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: Gemma4RMSNorm?

    // Layer scalar (present in all models)
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    // Per-layer embedding components (E4B only: hidden_size_per_layer_input > 0)
    let hasPerLayerEmbed: Bool
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4RMSNorm?

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.enableMoE = config.enableMoeBlock
        // Per-layer embeddings only for E4B-style models (hidden_size_per_layer_input > 0 in config)
        // 26B MoE does not use per-layer embeddings
        self.hasPerLayerEmbed = !config.enableMoeBlock

        // k_eq_v: full attention layers with attention_k_eq_v=true share k as v
        let isKEqV = config.attentionKEqV && config.isFullAttention(layer: layerIdx)
        self._selfAttention.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx, useKEqV: isKEqV)
        self.mlp = Gemma4MLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        self._inputLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // MoE: Router + Experts + additional norms
        if config.enableMoeBlock {
            self._router.wrappedValue = Gemma4Router(config)
            self._experts.wrappedValue = Gemma4Experts(config)
            self._postFeedforwardLayerNorm1.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        // Layer scalar (all models)
        self._layerScalar.wrappedValue = MLXArray.ones([1])

        // Per-layer embedding (E4B only)
        if hasPerLayerEmbed {
            let perLayerDim = 256
            self._perLayerInputGate.wrappedValue = Linear(config.hiddenSize, perLayerDim, bias: false)
            self._perLayerProjection.wrappedValue = Linear(perLayerDim, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        // Attention block
        let inputNorm = inputLayerNorm(x)
        let r = selfAttention(inputNorm, mask: mask, cache: cache)
        let attnNorm = postAttentionLayerNorm(r)
        let h = x + attnNorm

        // Feed-forward block
        if enableMoE, let router, let experts,
           let norm1 = postFeedforwardLayerNorm1,
           let norm2 = postFeedforwardLayerNorm2,
           let preNorm2 = preFeedforwardLayerNorm2 {
            // MoE path: shared MLP + routed experts in parallel (temporarily disabled for debug)
            let h1 = norm1(mlp(preFeedforwardLayerNorm(h)))
            let (indices, weights) = router(h)
            let h2 = norm2(experts(preNorm2(h), indices: indices, weights: weights))
            let ffOut = postFeedforwardLayerNorm(h1 + h2)
            return h + ffOut
        } else {
            // Dense path (E4B)
            let preMLPNorm = preFeedforwardLayerNorm(h)
            let r2 = mlp(preMLPNorm)
            let postMLPNorm = postFeedforwardLayerNorm(r2)
            return h + postMLPNorm
        }
    }
}

// MARK: - Model

public class Gemma4Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: Gemma4RMSNorm?
    @ModuleInfo var layers: [Gemma4TransformerBlock]
    @ModuleInfo var norm: Gemma4RMSNorm

    let config: Gemma4TextConfiguration

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        // Per-layer embeddings (E4B only, not used by 26B MoE)
        if !config.enableMoeBlock {
            let perLayerDim = 256
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: perLayerDim * config.hiddenLayers
            )
            self._perLayerModelProjection.wrappedValue = Linear(config.hiddenSize, perLayerDim * config.hiddenLayers, bias: false)
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNorm(dimensions: perLayerDim, eps: config.rmsNormEps)
        }
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            Gemma4TransformerBlock(config, layerIdx: layerIdx)
        }
        self.norm = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)

        // Build masks: global (full context) and sliding window
        let firstFullIdx = config.layerTypes.firstIndex(of: "full_attention") ?? 0
        let globalMask = createAttentionMask(h: h, cache: cache?[firstFullIdx])
        let slidingWindowMask = createAttentionMask(
            h: h, cache: cache?[0], windowSize: config.slidingWindow)

        for (i, layer) in layers.enumerated() {
            let isFullAttn = config.isFullAttention(layer: i)
            let layerMask = isFullAttn ? globalMask : slidingWindowMask

            // Shared KV cache: later layers reuse cache from earlier layers
            let effectiveCache: KVCache?
            if let sourceLayer = config.kvSourceLayer(for: i), let cache {
                effectiveCache = cache[sourceLayer]
            } else {
                effectiveCache = cache?[i]
            }

            h = layer(h, mask: layerMask, cache: effectiveCache)
        }
        return norm(h)
    }
}

// MARK: - LLMModel

public class Gemma4TextModel: Module, LanguageModel {
    @ModuleInfo public var model: Gemma4Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Gemma4TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.model = Gemma4Model(config)
        // Always create lm_head — even with tied embeddings, quantized models
        // store separate lm_head weights (scales/biases/weight)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, mask: nil, cache: cache)

        out = lmHead(out)

        // Final logit softcapping
        if let cap = config.finalLogitSoftcapping, cap > 0 {
            out = tanh(out / cap) * cap
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        sanitize(weights: weights)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // Handle VLM nested weights
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // MoE weights: key names already match between HuggingFace and Swift module:
        //   model.layers.N.experts.switch_glu.{gate,up,down}_proj.{weight,scales,biases}
        //   model.layers.N.router.{proj,scale,per_expert_scale}
        // No remapping needed — language_model. prefix is stripped above.

        // Truncate embeddings to vocab size
        let keysToCheck = [
            "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ]
        for key in keysToCheck {
            if let tensor = processedWeights[key], tensor.dim(0) > config.vocabularySize {
                processedWeights[key] = tensor[0 ..< config.vocabularySize]
            }
        }

        // For tied embeddings, copy embed_tokens to lm_head
        if config.tieWordEmbeddings && processedWeights["lm_head.weight"] == nil {
            for suffix in ["weight", "scales", "biases"] {
                if let w = processedWeights["model.embed_tokens.\(suffix)"] {
                    processedWeights["lm_head.\(suffix)"] = w
                }
            }
        }

        return processedWeights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        var caches = [KVCache]()
        for i in 0 ..< config.hiddenLayers {
            let isGlobal = config.isFullAttention(layer: i)
            if isGlobal {
                let cache = StandardKVCache()
                cache.step = 1024
                caches.append(cache)
            } else {
                caches.append(RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }
        return caches
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int? = nil
    ) throws -> PrepareResult {
        let promptTokens = input.text.tokens
        guard promptTokens.shape[0] > 0 else {
            let emptyToken = MLXArray(Int32(0))[0 ..< 0]
            return .tokens(.init(tokens: emptyToken))
        }
        return .tokens(input.text)
    }
}
