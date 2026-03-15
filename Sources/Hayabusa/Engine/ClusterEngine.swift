import Foundation

/// Wraps a local `InferenceEngine` and distributes requests across cluster nodes
/// using Uzu bandwidth-first routing. Falls back to the local engine if a remote node fails.
final class ClusterEngine: InferenceEngine, @unchecked Sendable {
    private let localEngine: any InferenceEngine
    private let clusterManager: ClusterManager

    var modelDescription: String { localEngine.modelDescription }
    var slotCount: Int { localEngine.slotCount }

    init(localEngine: any InferenceEngine, clusterManager: ClusterManager) {
        self.localEngine = localEngine
        self.clusterManager = clusterManager
    }

    func generate(
        messages: [ChatMessage],
        maxTokens: Int,
        temperature: Float,
        priority: SlotPriority
    ) async throws -> GenerationResult {
        guard let node = clusterManager.nextNode() else {
            return try await localEngine.generate(
                messages: messages, maxTokens: maxTokens,
                temperature: temperature, priority: priority
            )
        }

        clusterManager.router.recordStart(nodeId: node.id)
        let t0 = CFAbsoluteTimeGetCurrent()

        if node.isLocal {
            do {
                let result = try await localEngine.generate(
                    messages: messages, maxTokens: maxTokens,
                    temperature: temperature, priority: priority
                )
                let elapsed = CFAbsoluteTimeGetCurrent() - t0
                clusterManager.router.recordCompletion(
                    nodeId: node.id, tokens: result.completionTokens, durationSec: elapsed
                )
                return result
            } catch {
                clusterManager.router.recordFailure(nodeId: node.id)
                throw error
            }
        }

        // Forward to remote node
        do {
            let result = try await forwardToRemote(
                node: node, messages: messages,
                maxTokens: maxTokens, temperature: temperature
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            clusterManager.router.recordCompletion(
                nodeId: node.id, tokens: result.completionTokens, durationSec: elapsed
            )
            clusterManager.markHealthy(nodeId: node.id)
            return result
        } catch {
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            print("[Uzu] Remote node \(node.id) failed after \(String(format: "%.1f", elapsed))s: \(error)")
            clusterManager.markFailed(nodeId: node.id)

            // Retry on a different node (excluding the failed one), fallback to local
            if let fallback = clusterManager.nextNode(excluding: [node.id]) {
                clusterManager.router.recordStart(nodeId: fallback.id)
                let t1 = CFAbsoluteTimeGetCurrent()
                do {
                    let result: GenerationResult
                    if fallback.isLocal {
                        result = try await localEngine.generate(
                            messages: messages, maxTokens: maxTokens,
                            temperature: temperature, priority: priority
                        )
                    } else {
                        result = try await forwardToRemote(
                            node: fallback, messages: messages,
                            maxTokens: maxTokens, temperature: temperature
                        )
                        clusterManager.markHealthy(nodeId: fallback.id)
                    }
                    let elapsed1 = CFAbsoluteTimeGetCurrent() - t1
                    clusterManager.router.recordCompletion(
                        nodeId: fallback.id, tokens: result.completionTokens, durationSec: elapsed1
                    )
                    return result
                } catch {
                    clusterManager.router.recordFailure(nodeId: fallback.id)
                    throw error
                }
            }

            // Last resort: local engine directly
            return try await localEngine.generate(
                messages: messages, maxTokens: maxTokens,
                temperature: temperature, priority: priority
            )
        }
    }

    func slotSummary() -> [(index: Int, state: String, priority: String, pos: Int32)] {
        localEngine.slotSummary()
    }

    func memoryInfo() -> EngineMemoryInfo? {
        localEngine.memoryInfo()
    }

    // MARK: - Remote Forwarding

    private func forwardToRemote(
        node: ClusterNode,
        messages: [ChatMessage],
        maxTokens: Int,
        temperature: Float
    ) async throws -> GenerationResult {
        let url = URL(string: "\(node.baseURL)/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 120

        let body = ChatRequest(
            messages: messages,
            model: nil,
            max_tokens: maxTokens,
            temperature: temperature,
            priority: nil
        )
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw HayabusaError.remoteNodeFailed
        }

        let chatResponse = try JSONDecoder().decode(RemoteResponse.self, from: data)
        let text = chatResponse.choices.first?.message.content ?? ""
        return GenerationResult(
            text: text,
            promptTokens: chatResponse.usage?.prompt_tokens ?? 0,
            completionTokens: chatResponse.usage?.completion_tokens ?? 0
        )
    }
}

// MARK: - Remote Response Decoding

private struct RemoteResponse: Decodable {
    struct Choice: Decodable {
        struct Message: Decodable {
            let content: String
        }
        let message: Message
    }
    struct Usage: Decodable {
        let prompt_tokens: Int
        let completion_tokens: Int
    }
    let choices: [Choice]
    let usage: Usage?
}
