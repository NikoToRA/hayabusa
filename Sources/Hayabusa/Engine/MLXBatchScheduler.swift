import Foundation
import MLX
import MLXLMCommon

// MARK: - MLXGenerationJob

final class MLXGenerationJob: @unchecked Sendable {
    let messages: [[String: String]]
    let maxTokens: Int
    let temperature: Float
    let priority: SlotPriority
    let continuation: CheckedContinuation<GenerationResult, Error>

    init(
        messages: [[String: String]],
        maxTokens: Int,
        temperature: Float,
        priority: SlotPriority,
        continuation: CheckedContinuation<GenerationResult, Error>
    ) {
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.priority = priority
        self.continuation = continuation
    }
}

// MARK: - AsyncSemaphore

/// Async-aware semaphore for slot admission control.
/// Supports dynamic adjustment of the total permit count.
final class AsyncSemaphore: @unchecked Sendable {
    private let lock = NSLock()
    private var count: Int
    private var totalPermits: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(count: Int) {
        self.count = count
        self.totalPermits = count
    }

    /// Dynamically adjust total permits. If increasing, wakes waiters.
    /// If decreasing, permits are reclaimed as jobs complete.
    func adjustPermits(to newTotal: Int) {
        lock.lock()
        let delta = newTotal - totalPermits
        totalPermits = newTotal
        if delta > 0 {
            // Release extra permits — wake waiters first, then add to count
            var toWake = min(delta, waiters.count)
            var woken: [CheckedContinuation<Void, Never>] = []
            while toWake > 0 {
                woken.append(waiters.removeFirst())
                toWake -= 1
            }
            let remaining = delta - woken.count
            count += remaining
            lock.unlock()
            for w in woken { w.resume() }
        } else if delta < 0 {
            // Reclaim permits: reduce available count (may go negative temporarily,
            // which means returning jobs won't signal until balance is restored)
            count += delta  // delta is negative
            lock.unlock()
        } else {
            lock.unlock()
        }
    }

    var currentTotal: Int {
        lock.lock()
        defer { lock.unlock() }
        return totalPermits
    }

    func wait() async {
        // Fast path: slot available
        if tryAcquire() { return }

        // Slow path: park until a slot frees up
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            self.parkOrResume(c)
        }
    }

    private nonisolated func tryAcquire() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if count > 0 {
            count -= 1
            return true
        }
        return false
    }

    private nonisolated func parkOrResume(_ c: CheckedContinuation<Void, Never>) {
        lock.lock()
        if count > 0 {
            count -= 1
            lock.unlock()
            c.resume()
        } else {
            waiters.append(c)
            lock.unlock()
        }
    }

    func signal() {
        lock.lock()
        if !waiters.isEmpty {
            let waiter = waiters.removeFirst()
            lock.unlock()
            waiter.resume()
        } else {
            count += 1
            lock.unlock()
        }
    }
}

// MARK: - MLXBatchScheduler

/// Scheduler that provides admission control for concurrent MLX generation.
///
/// Uses `modelContainer.generate()` which holds the model lock only during
/// prefill, then runs generation concurrently outside the lock. Multiple
/// generation streams overlap naturally on the Metal command queue, keeping
/// the GPU pipeline full.
final class MLXBatchScheduler: @unchecked Sendable {
    private let modelContainer: ModelContainer
    private var maxSlots: Int
    private let maxContext: Int?
    let semaphore: AsyncSemaphore

    // Diagnostics (protected by lock)
    private let lock = NSLock()
    private var slotStates: [SlotState]

    /// Hard minimum — never go below 1 slot.
    static let minimumSlots = 1
    /// Hard maximum — cap at 8 concurrent MLX streams.
    static let maximumSlots = 8

    init(modelContainer: ModelContainer, slotCount: Int, maxContext: Int? = nil) {
        self.modelContainer = modelContainer
        self.maxSlots = slotCount
        self.maxContext = maxContext
        self.slotStates = Array(repeating: .idle, count: slotCount)
        self.semaphore = AsyncSemaphore(count: slotCount)
    }

    // MARK: - Public API

    func submit(_ job: MLXGenerationJob) {
        Task {
            await self.runJob(job)
        }
    }

    /// Current number of configured slots.
    var currentSlotCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return slotStates.count
    }

    /// Number of slots currently busy (not idle).
    var activeSlotCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return slotStates.filter { $0 != .idle }.count
    }

    /// Dynamically adjust the slot count. Safe to call while jobs are running.
    func adjustSlots(to newCount: Int) {
        let clamped = max(Self.minimumSlots, min(Self.maximumSlots, newCount))

        lock.lock()
        let oldCount = slotStates.count
        guard clamped != oldCount else {
            lock.unlock()
            return
        }

        if clamped > oldCount {
            // Add idle slots
            slotStates.append(contentsOf: Array(repeating: SlotState.idle, count: clamped - oldCount))
        } else {
            // Shrink: remove trailing idle slots only
            var removed = 0
            while slotStates.count > clamped && removed < (oldCount - clamped) {
                if let lastIdleIdx = slotStates.lastIndex(of: .idle) {
                    slotStates.remove(at: lastIdleIdx)
                    removed += 1
                } else {
                    break  // All remaining slots are busy, can't shrink further now
                }
            }
        }
        let actualNew = slotStates.count
        maxSlots = actualNew
        lock.unlock()

        // Adjust semaphore permits to match
        semaphore.adjustPermits(to: actualNew)
        print("[MLX] Slots adjusted: \(oldCount) → \(actualNew)")
    }

    func slotSummary() -> [(index: Int, state: String, priority: String, pos: Int32)] {
        lock.lock()
        let states = slotStates
        lock.unlock()
        return states.enumerated().map { (i, state) in
            (index: i, state: state.rawValue, priority: "low", pos: 0)
        }
    }

    // MARK: - Job Execution

    private func runJob(_ job: MLXGenerationJob) async {
        // Wait for a slot (parks if all slots busy)
        await semaphore.wait()
        let slotIndex = acquireSlotIndex()

        do {
            // Prepare input (brief lock acquisition)
            setSlot(slotIndex, .promptEval)
            let userInput = UserInput(messages: job.messages)
            let lmInput = try await modelContainer.prepare(input: userInput)

            // Generate — holds model lock only during prefill,
            // then generation stream runs concurrently outside the lock
            setSlot(slotIndex, .generating)
            let params = GenerateParameters(
                maxTokens: job.maxTokens,
                maxKVSize: maxContext,
                temperature: job.temperature
            )
            let stream = try await modelContainer.generate(
                input: lmInput,
                parameters: params
            )

            // Consume stream (runs concurrently with other streams)
            var text = ""
            var promptTokens = 0
            var completionTokens = 0

            for try await generation in stream {
                switch generation {
                case .chunk(let chunk):
                    text += chunk
                case .info(let info):
                    promptTokens = info.promptTokenCount
                    completionTokens = info.generationTokenCount
                case .toolCall:
                    break
                }
            }

            setSlot(slotIndex, .idle)
            semaphore.signal()
            Memory.clearCache()

            // Strip Gemma 4 thinking channel tokens from output
            let cleanedText = Self.stripThinkingChannel(text)

            job.continuation.resume(returning: GenerationResult(
                text: cleanedText,
                promptTokens: promptTokens,
                completionTokens: completionTokens
            ))
        } catch {
            setSlot(slotIndex, .idle)
            semaphore.signal()
            Memory.clearCache()
            job.continuation.resume(throwing: error)
        }
    }

    // MARK: - Slot Management

    private func acquireSlotIndex() -> Int {
        lock.lock()
        defer { lock.unlock() }
        if let idx = slotStates.firstIndex(of: .idle) {
            slotStates[idx] = .promptEval
            return idx
        }
        // Shouldn't happen (semaphore guards), but fallback
        return 0
    }

    private nonisolated func setSlot(_ index: Int, _ state: SlotState) {
        lock.lock()
        if index < slotStates.count { slotStates[index] = state }
        lock.unlock()
    }

    /// Strip Gemma 4 thinking channel markers from generated text.
    /// Gemma 4 models emit `<|channel>thought\n...<channel|>` blocks before the actual response.
    private static func stripThinkingChannel(_ text: String) -> String {
        guard text.contains("<|channel>") else { return text }

        var result = text
        while let startRange = result.range(of: "<|channel>") {
            if let endRange = result.range(of: "<channel|>", range: startRange.upperBound..<result.endIndex) {
                result.removeSubrange(startRange.lowerBound..<endRange.upperBound)
            } else {
                result.removeSubrange(startRange.lowerBound..<result.endIndex)
                break
            }
        }
        // Trim leading whitespace/newlines only when channel markers were stripped
        return result.drop(while: { $0.isWhitespace || $0.isNewline }).description
    }
}
