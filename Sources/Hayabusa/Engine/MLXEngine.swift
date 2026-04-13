import Foundation
import MLX
import MLXLLM
import MLXLMCommon
@preconcurrency import Tokenizers
import Hub

final class MLXEngine: InferenceEngine, @unchecked Sendable {
    private let modelContainer: ModelContainer
    private let scheduler: MLXBatchScheduler
    private let memoryMonitor: MemoryMonitor
    let modelDescription: String
    private let initialSlotCount: Int
    let layerSkipConfig: LayerSkipConfig?

    var slotCount: Int { scheduler.currentSlotCount }

    init(modelId: String, slotCount: Int = 4, maxMemoryGB: Double? = nil, maxContext: Int? = nil,
         layerSkipConfig: LayerSkipConfig? = nil) async throws {
        self.initialSlotCount = slotCount

        // Register Gemma 4 model types (not yet in upstream mlx-swift-lm)
        Self.registerGemma4ModelTypes()

        // Create a temp directory with fixed config.json (stripped "language_model."
        // prefix from quantization keys) and symlinks to all other model files.
        // This preserves the original config.json for Python mlx_lm compatibility.
        let configuration: ModelConfiguration
        if let sourceDir = Self.cachedSnapshotDirectory(for: modelId),
           let tempDir = Self.createFixedConfigDirectory(from: sourceDir) {
            configuration = ModelConfiguration(directory: tempDir)
        } else {
            configuration = Self.resolveModelConfiguration(for: modelId)
        }

        print("[MLX] Downloading/loading model: \(modelId)")
        let downloader = HubDownloaderBridge()
        let tokenizerLoader = TransformersTokenizerLoader()
        self.modelContainer = try await LLMModelFactory.shared.loadContainer(
            from: downloader,
            using: tokenizerLoader,
            configuration: configuration,
            progressHandler: { progress in
                if progress.fractionCompleted < 1.0 {
                    print("[MLX] Progress: \(Int(progress.fractionCompleted * 100))%")
                }
            }
        )

        // Apply memory limits after model load
        if let gb = maxMemoryGB {
            let bytes = Int(gb * 1024 * 1024 * 1024)
            Memory.memoryLimit = bytes
            Memory.cacheLimit = min(256 * 1024 * 1024, bytes / 10)
            Memory.clearCache()
            print("[MLX] Memory limit: \(gb)GB, cache limit: \(min(256, Int(gb * 1024) / 10))MB")
        }
        if let ctx = maxContext {
            print("[MLX] Max KV context: \(ctx)")
        }

        // Apply layer skipping before creating scheduler
        self.layerSkipConfig = layerSkipConfig
        if let config = layerSkipConfig {
            await config.apply(to: modelContainer)
        }

        self.modelDescription = "MLX \(modelId)"
        self.scheduler = MLXBatchScheduler(modelContainer: modelContainer, slotCount: slotCount, maxContext: maxContext)

        // Set up memory monitor with dynamic slot adjustment
        let sched = self.scheduler
        self.memoryMonitor = MemoryMonitor(activeSlots: { [weak sched] in
            sched?.activeSlotCount ?? 0
        })

        let initSlots = slotCount
        self.memoryMonitor.onPressureChange = { [weak sched] pressure, info in
            guard let sched else { return }
            let current = sched.currentSlotCount

            switch pressure {
            case .normal:
                // Free > 4GB: can grow back toward initial count (or +1)
                if current < initSlots {
                    sched.adjustSlots(to: current + 1)
                }
            case .low:
                // 2-4GB free: hold steady, no changes
                break
            case .critical:
                // 1-2GB free: reduce by 1 slot
                if current > MLXBatchScheduler.minimumSlots {
                    sched.adjustSlots(to: current - 1)
                }
                Memory.clearCache()
            case .emergency:
                // < 1GB free: emergency — drop to minimum + clear cache
                sched.adjustSlots(to: MLXBatchScheduler.minimumSlots)
                Memory.clearCache()
                print("[MLX] EMERGENCY: memory critically low, forced to \(MLXBatchScheduler.minimumSlots) slot(s)")
            }
        }

        self.memoryMonitor.start()
        print("[MLX] Model loaded successfully (batch scheduler + memory monitor active)")
    }

    private static func resolveModelConfiguration(for modelId: String) -> ModelConfiguration {
        let fileManager = FileManager.default
        let expandedPath = NSString(string: modelId).expandingTildeInPath
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: expandedPath, isDirectory: &isDirectory), isDirectory.boolValue {
            return ModelConfiguration(directory: URL(fileURLWithPath: expandedPath))
        }

        if let cachedDirectory = cachedSnapshotDirectory(for: modelId) {
            print("[MLX] Using cached snapshot: \(cachedDirectory.path)")
            return ModelConfiguration(directory: cachedDirectory)
        }

        return ModelConfiguration(id: modelId)
    }

    private static func cachedSnapshotDirectory(for modelId: String) -> URL? {
        let fileManager = FileManager.default
        let escapedId = modelId.replacingOccurrences(of: "/", with: "--")
        let homeDirectory = fileManager.homeDirectoryForCurrentUser
        let cacheRoots = [
            homeDirectory.appending(path: ".cache/huggingface/hub"),
            homeDirectory.appending(path: "Library/Caches/huggingface/hub"),
        ]

        for root in cacheRoots {
            let repoDirectory = root.appending(path: "models--\(escapedId)")
            let refsMain = repoDirectory.appending(path: "refs/main")
            let snapshotsDirectory = repoDirectory.appending(path: "snapshots")

            if let revision = try? String(contentsOf: refsMain, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines),
               !revision.isEmpty {
                let resolvedSnapshot = snapshotsDirectory.appending(path: revision)
                if fileManager.fileExists(atPath: resolvedSnapshot.path) {
                    return resolvedSnapshot
                }
            }

            if let snapshots = try? fileManager.contentsOfDirectory(
                at: snapshotsDirectory,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            ),
            let newestSnapshot = snapshots.sorted(by: { lhs, rhs in
                let lhsDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let rhsDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return lhsDate > rhsDate
            }).first {
                return newestSnapshot
            }
        }

        return nil
    }

    func generate(
        messages: [ChatMessage],
        maxTokens: Int,
        temperature: Float,
        priority: SlotPriority
    ) async throws -> GenerationResult {
        let mlxMessages: [[String: String]] = messages.map {
            ["role": $0.role, "content": $0.content]
        }

        return try await withCheckedThrowingContinuation { continuation in
            let job = MLXGenerationJob(
                messages: mlxMessages,
                maxTokens: maxTokens,
                temperature: temperature,
                priority: priority,
                continuation: continuation
            )
            scheduler.submit(job)
        }
    }

    func slotSummary() -> [(index: Int, state: String, priority: String, pos: Int32)] {
        scheduler.slotSummary()
    }

    func collectGenome(config: GenomeConfig) async throws {
        try await GenomeCollector.collect(
            from: modelContainer,
            modelName: modelDescription,
            config: config
        )
    }

    func memoryInfo() -> EngineMemoryInfo? {
        let info = memoryMonitor.latestInfo
        let pressure = memoryMonitor.currentPressure
        return EngineMemoryInfo(
            totalPhysical: info.totalPhysical,
            rssBytes: info.rssBytes,
            freeEstimate: info.freeEstimate,
            activeSlots: info.activeSlots,
            pressure: pressure.rawValue
        )
    }

    // MARK: - Gemma 4 Model Registration

    private static var gemma4Registered = false

    private static func registerGemma4ModelTypes() {
        guard !gemma4Registered else { return }
        gemma4Registered = true

        let creator: @Sendable (Data) throws -> any LanguageModel = { data in
            let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
            return Gemma4TextModel(config)
        }

        Task {
            await LLMTypeRegistry.shared.registerModelType("gemma4", creator: creator)
            await LLMTypeRegistry.shared.registerModelType("gemma4_text", creator: creator)
        }
        Thread.sleep(forTimeInterval: 0.1)
        print("[MLX] Registered Gemma 4 model types (gemma4, gemma4_text)")
    }

    /// Create a temp directory with a fixed config.json (quantization keys stripped of
    /// "language_model." prefix) and symlinks to all other model files.
    /// Returns nil if no fix is needed or on error.
    private static func createFixedConfigDirectory(from sourceDir: URL) -> URL? {
        let configURL = sourceDir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }

        func stripPrefix(_ dict: [String: Any]) -> ([String: Any], Bool) {
            var result = [String: Any]()
            var changed = false
            for (key, value) in dict {
                if key.hasPrefix("language_model.") {
                    result[String(key.dropFirst("language_model.".count))] = value
                    changed = true
                } else {
                    result[key] = value
                }
            }
            return (result, changed)
        }

        var modified = false
        if let quant = json["quantization"] as? [String: Any] {
            let (stripped, changed) = stripPrefix(quant)
            if changed { json["quantization"] = stripped; modified = true }
        }
        if let qc = json["quantization_config"] as? [String: Any] {
            let (stripped, changed) = stripPrefix(qc)
            if changed { json["quantization_config"] = stripped; modified = true }
        }

        guard modified else { return nil }

        // Create temp directory with fixed config + symlinks to all other files
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("hayabusa-mlx-\(UUID().uuidString)")
        do {
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
            let newData = try JSONSerialization.data(withJSONObject: json, options: .prettyPrinted)
            try newData.write(to: tempDir.appendingPathComponent("config.json"))

            // Symlink all other files from source
            let items = try FileManager.default.contentsOfDirectory(at: sourceDir, includingPropertiesForKeys: nil)
            for item in items where item.lastPathComponent != "config.json" {
                try FileManager.default.createSymbolicLink(
                    at: tempDir.appendingPathComponent(item.lastPathComponent),
                    withDestinationURL: item
                )
            }
            print("[MLX] Created temp config with fixed quantization keys at \(tempDir.path)")
            return tempDir
        } catch {
            print("[MLX] Failed to create temp config directory: \(error)")
            try? FileManager.default.removeItem(at: tempDir)
            return nil
        }
    }
}

// MARK: - HuggingFace Bridge types

private struct TokenizerBridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let upstream: any Tokenizers.Tokenizer
    init(_ upstream: any Tokenizers.Tokenizer) { self.upstream = upstream }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { upstream.encode(text: text, addSpecialTokens: addSpecialTokens) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens) }
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        do { return try upstream.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext) }
        catch Tokenizers.TokenizerError.missingChatTemplate {
            // Gemma 4 fallback: build chat prompt manually
            // Format: <bos><|turn>user\n{content}<turn|>\n<|turn>model\n
            let isGemma4 = upstream.convertTokenToId("<|turn>") != nil
            if isGemma4 {
                var prompt = "<bos>"
                for msg in messages {
                    let role = msg["role"] as? String ?? "user"
                    let content = msg["content"] as? String ?? ""
                    prompt += "<|turn>\(role)\n\(content)<turn|>\n"
                }
                prompt += "<|turn>model\n"
                print("[MLX] Applied Gemma 4 chat template fallback")
                return upstream.encode(text: prompt, addSpecialTokens: false)
            }
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

private struct HubDownloaderBridge: MLXLMCommon.Downloader {
    private let hub: HubApi
    init(_ hub: HubApi = HubApi()) { self.hub = hub }
    func download(id: String, revision: String?, matching patterns: [String], useLatest: Bool, progressHandler: @Sendable @escaping (Progress) -> Void) async throws -> URL {
        try await hub.snapshot(from: id, revision: revision ?? "main", matching: patterns, progressHandler: progressHandler)
    }
}

private struct TransformersTokenizerLoader: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let tokenizerConfigURL = directory.appending(path: "tokenizer_config.json")
        let tokenizerDataURL = directory.appending(path: "tokenizer.json")

        let tokenizerConfigData = try Data(contentsOf: tokenizerConfigURL)
        let tokenizerData = try Data(contentsOf: tokenizerDataURL)

        let rawConfig = try JSONSerialization.jsonObject(with: tokenizerConfigData)
        guard let configDictionary = rawConfig as? [String: Any] else {
            throw Tokenizers.TokenizerError.missingConfig
        }

        var patchedConfigDictionary = configDictionary
        if let tokenizerClass = patchedConfigDictionary["tokenizer_class"] as? String,
           tokenizerClass == "TokenizersBackend" {
            patchedConfigDictionary["tokenizer_class"] = "Qwen2Tokenizer"
        }

        let tokenizerConfig = Config((patchedConfigDictionary as NSDictionary) as? [NSString: Any] ?? [:])
        let tokenizerDataConfig = try JSONDecoder().decode(Config.self, from: tokenizerData)
        return TokenizerBridge(try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerDataConfig))
    }
}
