import Foundation
import Network

// MARK: - ClusterNode

struct ClusterNode: Sendable {
    let id: String          // "host:port"
    let host: String
    let port: Int
    let backend: String
    let model: String
    var slots: Int
    let isLocal: Bool
    var isHealthy: Bool
    var lastSeen: Date
    var consecutiveFailures: Int

    // Memory info (updated periodically for local node)
    var totalMemory: UInt64 = 0
    var rssBytes: UInt64 = 0
    var freeMemory: UInt64 = 0
    var memoryPressure: String = "unknown"

    var baseURL: String { "http://\(host):\(port)" }
}

// MARK: - ClusterManager

/// Manages Bonjour-based LAN peer discovery for Hayabusa cluster mode.
///
/// Advertises the local node via `NWListener` and discovers peers via `NWBrowser`.
/// Provides round-robin node selection with failure tracking.
final class ClusterManager: @unchecked Sendable {
    private let httpPort: Int
    private let backend: String
    private let model: String
    private let slots: Int

    private let lock = NSLock()
    private var nodes: [String: ClusterNode] = [:]
    private var roundRobinIndex = 0

    private var listener: NWListener?
    private var browser: NWBrowser?

    private let serviceType = "_hayabusa._tcp"

    init(httpPort: Int, backend: String, model: String, slots: Int) {
        self.httpPort = httpPort
        self.backend = backend
        self.model = model
        self.slots = slots
    }

    // MARK: - Start / Stop

    func start() {
        startListener()
        startBrowser()
    }

    func stop() {
        listener?.cancel()
        browser?.cancel()
    }

    // MARK: - Bonjour Advertising

    private func startListener() {
        do {
            // Use a TCP listener on an ephemeral port (port 0) for Bonjour advertisement.
            // This avoids conflicting with the HTTP server port.
            let params = NWParameters.tcp
            let listener = try NWListener(using: params, on: .any)

            let txtRecord = NWTXTRecord([
                "port": "\(httpPort)",
                "backend": backend,
                "model": model,
                "slots": "\(slots)",
            ])

            listener.service = NWListener.Service(
                name: nil,  // auto-generated unique name
                type: serviceType,
                txtRecord: txtRecord
            )

            listener.newConnectionHandler = { connection in
                // Accept and immediately cancel — this listener is only for Bonjour advertisement
                connection.cancel()
            }

            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    if let port = listener.port {
                        print("[Cluster] Bonjour advertising on port \(port) (HTTP: \(self.httpPort))")
                    }
                case .failed(let error):
                    print("[Cluster] Listener failed: \(error)")
                default:
                    break
                }
            }

            listener.start(queue: .global(qos: .utility))
            self.listener = listener
        } catch {
            print("[Cluster] Failed to create listener: \(error)")
        }
    }

    // MARK: - Peer Discovery

    private func startBrowser() {
        let browser = NWBrowser(
            for: .bonjour(type: serviceType, domain: "local."),
            using: .tcp
        )

        browser.browseResultsChangedHandler = { results, changes in
            for change in changes {
                switch change {
                case .added(let result):
                    self.handlePeerAdded(result)
                case .removed(let result):
                    self.handlePeerRemoved(result)
                default:
                    break
                }
            }
        }

        browser.stateUpdateHandler = { state in
            switch state {
            case .ready:
                print("[Cluster] Browsing for peers...")
            case .failed(let error):
                print("[Cluster] Browser failed: \(error)")
            default:
                break
            }
        }

        browser.start(queue: .global(qos: .utility))
        self.browser = browser
    }

    private func handlePeerAdded(_ result: NWBrowser.Result) {
        // Resolve the endpoint to get TXT record data
        guard case .service = result.endpoint else { return }

        // Extract TXT record
        var port = 0
        var backend = ""
        var model = ""
        var slots = 0

        if case .bonjour(let txt) = result.metadata {
            port = Int(txt["port"] ?? "") ?? 0
            backend = txt["backend"] ?? ""
            model = txt["model"] ?? ""
            slots = Int(txt["slots"] ?? "") ?? 0
        }

        guard port > 0 else { return }

        // Resolve the host — use the service name + .local for now
        // NWBrowser gives us the endpoint, but for HTTP we need host:port
        let connection = NWConnection(to: result.endpoint, using: .tcp)
        connection.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            if case .ready = state {
                if let path = connection.currentPath,
                   let endpoint = path.remoteEndpoint,
                   case .hostPort(let host, _) = endpoint {
                    let hostStr = "\(host)"
                    let nodeId = "\(hostStr):\(port)"
                    let isLocal = port == self.httpPort && self.isLocalAddress(hostStr)

                    let node = ClusterNode(
                        id: nodeId,
                        host: hostStr,
                        port: port,
                        backend: backend,
                        model: model,
                        slots: slots,
                        isLocal: isLocal,
                        isHealthy: true,
                        lastSeen: Date(),
                        consecutiveFailures: 0
                    )

                    self.lock.lock()
                    self.nodes[nodeId] = node
                    self.lock.unlock()
                    print("[Cluster] Peer added: \(nodeId) (backend: \(backend), local: \(isLocal))")
                }
                connection.cancel()
            } else if case .failed = state {
                connection.cancel()
            }
        }
        connection.start(queue: .global(qos: .utility))
    }

    private func handlePeerRemoved(_ result: NWBrowser.Result) {
        // Try to find and remove the matching node
        if case .bonjour(let txt) = result.metadata {
            let port = Int(txt["port"] ?? "") ?? 0
            if port > 0 {
                lock.lock()
                // Remove nodes matching this port (may match multiple if host is known)
                let toRemove = nodes.filter { $0.value.port == port && !$0.value.isLocal }
                for key in toRemove.keys {
                    nodes.removeValue(forKey: key)
                    print("[Cluster] Peer removed: \(key)")
                }
                lock.unlock()
            }
        }
    }

    private func isLocalAddress(_ host: String) -> Bool {
        host == "127.0.0.1" || host == "::1" || host == "localhost"
            || host.hasPrefix("fe80::") || host == "0.0.0.0"
    }

    // MARK: - Round-Robin Node Selection

    /// Returns the next healthy node in round-robin order.
    /// Returns `nil` only if no nodes are known at all.
    func nextNode() -> ClusterNode? {
        lock.lock()
        defer { lock.unlock() }

        let healthyNodes = nodes.values.filter { node in
            if node.isLocal { return true }
            if !node.isHealthy { return false }
            if node.consecutiveFailures >= 3 {
                // Cooldown: 30 seconds after 3 consecutive failures
                return Date().timeIntervalSince(node.lastSeen) > 30
            }
            return true
        }.sorted { $0.id < $1.id }

        guard !healthyNodes.isEmpty else { return nil }

        roundRobinIndex = roundRobinIndex % healthyNodes.count
        let node = healthyNodes[roundRobinIndex]
        roundRobinIndex += 1
        return node
    }

    /// Mark a node as failed. After 3 consecutive failures, enters 30s cooldown.
    func markFailed(nodeId: String) {
        lock.lock()
        if var node = nodes[nodeId] {
            node.consecutiveFailures += 1
            if node.consecutiveFailures >= 3 {
                node.isHealthy = false
                node.lastSeen = Date()  // cooldown timer starts now
                print("[Cluster] Node \(nodeId) marked unhealthy (failures: \(node.consecutiveFailures))")
            }
            nodes[nodeId] = node
        }
        lock.unlock()
    }

    /// Mark a node as healthy after successful request.
    func markHealthy(nodeId: String) {
        lock.lock()
        if var node = nodes[nodeId] {
            node.consecutiveFailures = 0
            node.isHealthy = true
            node.lastSeen = Date()
            nodes[nodeId] = node
        }
        lock.unlock()
    }

    // MARK: - Memory Updates

    /// Update the local node's memory info and slot count (called periodically).
    func updateLocalMemory(_ info: EngineMemoryInfo) {
        lock.lock()
        for key in nodes.keys {
            if nodes[key]!.isLocal {
                nodes[key]!.totalMemory = info.totalPhysical
                nodes[key]!.rssBytes = info.rssBytes
                nodes[key]!.freeMemory = info.freeEstimate
                nodes[key]!.memoryPressure = info.pressure
                nodes[key]!.slots = info.activeSlots
            }
        }
        lock.unlock()
    }

    // MARK: - Status

    /// Returns all known nodes for the status endpoint.
    func allNodes() -> [ClusterNode] {
        lock.lock()
        defer { lock.unlock() }
        return Array(nodes.values).sorted { $0.id < $1.id }
    }
}
