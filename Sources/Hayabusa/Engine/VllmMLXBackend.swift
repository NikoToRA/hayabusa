import Foundation

/// Backend that proxies inference requests to an external vllm-mlx server.
/// vllm-mlx exposes OpenAI-compatible endpoints, so we forward
/// /v1/chat/completions and parse the response (including SSE streaming).
final class VllmMLXBackend: InferenceEngine, @unchecked Sendable {
    let modelDescription: String
    let slotCount: Int = 1  // proxy — single logical slot

    private let endpoint: URL
    private let session: URLSession

    init(endpoint urlString: String) async throws {
        guard let url = URL(string: urlString) else {
            throw HayabusaError.modelLoadFailed("Invalid vllm-mlx endpoint URL: \(urlString)")
        }
        self.endpoint = url

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 300
        config.timeoutIntervalForResource = 600
        self.session = URLSession(configuration: config)

        // Fetch model name from vllm-mlx /v1/models
        self.modelDescription = try await VllmMLXBackend.fetchModelDescription(
            endpoint: url, session: self.session
        )
        print("[vllm-mlx] Connected to \(urlString), model: \(self.modelDescription)")
    }

    // MARK: - InferenceEngine

    func generate(
        messages: [ChatMessage],
        maxTokens: Int,
        temperature: Float,
        priority: SlotPriority
    ) async throws -> GenerationResult {
        let completionsURL = endpoint.appendingPathComponent("v1/chat/completions")
        var request = URLRequest(url: completionsURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(priority == .high ? "high" : "low",
                        forHTTPHeaderField: "X-Hayabusa-Priority")

        // Build the OpenAI-compatible request body.
        // Use streaming so we can collect partial chunks, but we return
        // the full result once complete (matching InferenceEngine contract).
        let body: [String: Any] = [
            "model": modelDescription,
            "messages": messages.map { ["role": $0.role, "content": $0.content] },
            "max_tokens": maxTokens,
            "temperature": temperature,
            "stream": true
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        // Use URLSession bytes for streaming SSE
        let (bytes, response) = try await session.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw HayabusaError.decodeFailed
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            // Try to read error body
            var errorBody = ""
            for try await line in bytes.lines {
                errorBody += line
                if errorBody.count > 1024 { break }
            }
            throw HayabusaError.modelLoadFailed(
                "vllm-mlx returned HTTP \(httpResponse.statusCode): \(errorBody)"
            )
        }

        // Parse SSE stream: collect content deltas
        var fullContent = ""
        var completionTokens = 0
        var promptTokens = 0

        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst(6))
            if payload == "[DONE]" { break }

            guard let data = payload.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                continue
            }

            // Extract content delta
            if let choices = json["choices"] as? [[String: Any]],
               let first = choices.first,
               let delta = first["delta"] as? [String: Any],
               let content = delta["content"] as? String {
                fullContent += content
                completionTokens += 1  // approximate: 1 chunk ~ 1 token for SSE
            }

            // Extract usage if present (some vllm implementations include it)
            if let usage = json["usage"] as? [String: Any] {
                if let pt = usage["prompt_tokens"] as? Int { promptTokens = pt }
                if let ct = usage["completion_tokens"] as? Int { completionTokens = ct }
            }
        }

        return GenerationResult(
            text: fullContent,
            promptTokens: promptTokens,
            completionTokens: completionTokens
        )
    }

    func slotSummary() -> [(index: Int, state: String, priority: String, pos: Int32)] {
        // Single proxy slot — always report as idle (the real scheduling is
        // handled by the vllm-mlx server).
        return [(index: 0, state: "proxy", priority: "n/a", pos: 0)]
    }

    func memoryInfo() -> EngineMemoryInfo? {
        // We don't have direct access to the vllm-mlx process memory.
        // Return nil so the server reports "unknown" pressure.
        return nil
    }

    // MARK: - Helpers

    /// Fetches the first available model name from vllm-mlx /v1/models.
    private static func fetchModelDescription(
        endpoint: URL, session: URLSession
    ) async throws -> String {
        let modelsURL = endpoint.appendingPathComponent("v1/models")
        var request = URLRequest(url: modelsURL)
        request.httpMethod = "GET"
        request.timeoutInterval = 10

        do {
            let (data, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                  (200..<300).contains(httpResponse.statusCode) else {
                return "vllm-mlx"
            }

            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let dataArray = json["data"] as? [[String: Any]],
               let first = dataArray.first,
               let id = first["id"] as? String {
                return "vllm-mlx/\(id)"
            }
        } catch {
            print("[vllm-mlx] Warning: could not reach /v1/models (\(error.localizedDescription)), using default name")
        }

        return "vllm-mlx"
    }
}
