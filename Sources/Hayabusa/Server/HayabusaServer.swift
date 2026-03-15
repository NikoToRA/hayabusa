import Foundation
import Hummingbird

struct HayabusaServer {
    let engine: any InferenceEngine
    let port: Int
    let bindAddress: String
    let clusterManager: ClusterManager?

    init(engine: any InferenceEngine, port: Int, bindAddress: String = "127.0.0.1", clusterManager: ClusterManager? = nil) {
        self.engine = engine
        self.port = port
        self.bindAddress = bindAddress
        self.clusterManager = clusterManager
    }

    func run() async throws {
        let router = Router()
        let engine = self.engine
        let clusterManager = self.clusterManager

        // GET /health
        router.get("health") { _, _ -> String in
            "{\"status\":\"ok\"}"
        }

        // POST /v1/chat/completions
        router.post("v1/chat/completions") { request, context in
            let chatRequest = try await context.requestDecoder.decode(
                ChatRequest.self, from: request, context: context
            )

            let result = try await engine.generate(
                messages: chatRequest.messages,
                maxTokens: chatRequest.max_tokens ?? 2048,
                temperature: chatRequest.temperature ?? 0.7,
                priority: SlotPriority(string: chatRequest.priority)
            )

            let response = ChatResponse(
                id: "hayabusa-\(UUID().uuidString.prefix(8))",
                model: chatRequest.model ?? "local",
                content: result.text,
                promptTokens: result.promptTokens,
                completionTokens: result.completionTokens
            )

            let jsonData = try JSONEncoder().encode(response)
            let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"
            return jsonString
        }

        // GET /slots — diagnostic endpoint
        router.get("slots") { _, _ -> String in
            let summary = engine.slotSummary()
            let slots = summary.map { slot in
                "{\"index\":\(slot.index),\"state\":\"\(slot.state)\",\"priority\":\"\(slot.priority)\",\"pos\":\(slot.pos)}"
            }
            return "[\(slots.joined(separator: ","))]"
        }

        // GET /v1/memory — memory status (available for any backend)
        router.get("v1/memory") { _, _ -> String in
            if let info = engine.memoryInfo() {
                return """
                {"totalPhysical":\(info.totalPhysical),"rssBytes":\(info.rssBytes),\
                "freeEstimate":\(info.freeEstimate),"activeSlots":\(info.activeSlots),\
                "pressure":"\(info.pressure)","slots":\(engine.slotCount)}
                """
            }
            return "{\"pressure\":\"unknown\"}"
        }

        // GET /v1/cluster/status — cluster node listing with memory info
        router.get("v1/cluster/status") { _, _ -> String in
            // Update local node memory before responding
            if let cm = clusterManager, let info = engine.memoryInfo() {
                cm.updateLocalMemory(info)
            }

            guard let cm = clusterManager else {
                return "{\"cluster\":false}"
            }
            let nodes = cm.allNodes()
            let nodesJson = nodes.map { node in
                """
                {"id":"\(node.id)","host":"\(node.host)","port":\(node.port),\
                "backend":"\(node.backend)","model":"\(node.model)","slots":\(node.slots),\
                "isLocal":\(node.isLocal),"isHealthy":\(node.isHealthy),\
                "consecutiveFailures":\(node.consecutiveFailures),\
                "totalMemory":\(node.totalMemory),"rssBytes":\(node.rssBytes),\
                "freeMemory":\(node.freeMemory),"memoryPressure":"\(node.memoryPressure)"}
                """
            }
            let bandwidthJson = cm.bandwidthSnapshots().map { s in
                """
                {"nodeId":"\(s.nodeId)","isLocal":\(s.isLocal),\
                "ewmaTokPerSec":\(String(format: "%.1f", s.ewmaTokPerSec)),\
                "activeRequests":\(s.activeRequests),\
                "totalRequests":\(s.totalRequests),"totalTokens":\(s.totalTokens)}
                """
            }
            return """
            {"cluster":true,"routing":"uzu",\
            "nodes":[\(nodesJson.joined(separator: ","))],\
            "bandwidth":[\(bandwidthJson.joined(separator: ","))]}
            """
        }

        let app = Application(
            router: router,
            configuration: .init(address: .hostname(bindAddress, port: port))
        )
        try await app.runService()
    }
}
