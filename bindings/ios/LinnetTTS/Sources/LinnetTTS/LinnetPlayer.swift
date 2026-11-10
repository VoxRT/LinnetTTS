import AVFoundation
import Foundation

/// Minimal streaming playback helper: an `AVAudioEngine` with one `AVAudioPlayerNode` that
/// accepts the chunks of a ``LinnetSession`` as they are produced.
///
/// Usage, on a background queue:
/// ```swift
/// let session = try model.session(voice: voice)
/// let player = try LinnetPlayer(sampleRate: model.sampleRate)
/// try session.prime(text: text)
/// while let chunk = try session.step() { try player.write(chunk) }
/// player.drain()
/// ```
/// ``write(_:)`` blocks while more than `bufferMillis` of audio is queued. The default keeps 8 s ahead,
/// so a typical utterance is synthesised in one burst at full speed and playback drains behind it. Do not
/// pace synthesis to real time with a small buffer: a thread that sleeps 70 ms of every 80 ms is clocked
/// down and moved to efficiency cores by iOS, and each step then takes 3-4 times longer (measured 22 ms
/// instead of 6.7 ms on an A13). The player is not thread-safe; use it from the thread that drives the
/// session (``stop()`` may be called from any thread).
///
/// - Parameters:
///   - sampleRate: output rate, ``LinnetModel/sampleRate``.
///   - bufferMillis: audio kept queued ahead of playback (default 8000). Smaller values only make
///     ``stop()`` drop less audio; they do not reduce latency to the first chunk.
public final class LinnetPlayer {
    /// When the first buffer of audio went through the output, for latency measurements.
    /// All fields are `nil` until the corresponding event has happened.
    public struct PlaybackTiming {
        /// The engine and the player node were running and the first buffer was scheduled.
        public var firstBufferScheduled: Date?
        /// Samples in the first buffer.
        public var firstBufferSamples = 0
        /// The first buffer's `.dataPlayedBack` callback: its *last* sample has left the device
        /// (Apple accounts for the downstream and device latency in this callback).
        public var firstBufferPlayedBack: Date?
        /// Start of the first buffer at the device, from the player node's render clock plus
        /// `AVAudioSession.outputLatency` (iOS; on macOS no latency is added).
        public var onsetFromClock: Date?
        /// `firstBufferPlayedBack` minus the first buffer's duration: the other estimate of the onset.
        public var onsetBySubtraction: Date?
        /// `true` when the engine was already running before the first `write` (``prewarm()``).
        public var routeWasOpen = false
    }

    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private let sampleRate: Int
    private let maxQueuedSamples: Int
    private let lock = NSCondition()
    private var queuedSamples = 0
    private var generation = 0
    private var started = false
    private var scheduledSamples = 0
    private var firstRealBufferPending = true
    private var _timing = PlaybackTiming()

    /// Timing of the first buffer written through ``write(_:)`` (see ``PlaybackTiming``).
    public var timing: PlaybackTiming {
        lock.lock()
        defer { lock.unlock() }
        return _timing
    }

    public init(sampleRate: Int, bufferMillis: Int = 8000) throws {
        guard let f = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
                                    interleaved: false)
        else { throw LinnetError.invalidArgument("unsupported sample rate \(sampleRate)") }
        format = f
        self.sampleRate = sampleRate
        maxQueuedSamples = max(1, bufferMillis) * sampleRate / 1000
        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .spokenAudio)
            try session.setActive(true)
        #endif
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
    }

    /// Start the engine and the player node and play 50 ms of silence, so that the output route is
    /// open before the first ``write(_:)``; otherwise the first `write` opens it. For latency
    /// measurements of both cases (see ``timing``).
    public func prewarm() throws {
        try start()
        try schedule([Float](repeating: 0, count: sampleRate / 20), real: false)
    }

    private func start() throws {
        if !started {
            try engine.start()
            node.play()
            started = true
        }
    }

    /// Queue `chunk` (blocking while the queue is full). Playback starts on the first call.
    public func write(_ chunk: [Float]) throws {
        if chunk.isEmpty { return }
        let routeWasOpen = started
        try start()
        try schedule(chunk, real: true, routeWasOpen: routeWasOpen)
    }

    private func schedule(_ chunk: [Float], real: Bool, routeWasOpen: Bool = false) throws {
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(chunk.count)) else {
            throw LinnetError.internalError("AVAudioPCMBuffer allocation failed")
        }
        buf.frameLength = AVAudioFrameCount(chunk.count)
        chunk.withUnsafeBufferPointer { src in
            buf.floatChannelData![0].update(from: src.baseAddress!, count: chunk.count)
        }
        lock.lock()
        while started, queuedSamples > 0, queuedSamples + chunk.count > maxQueuedSamples {
            lock.wait()
        }
        if !started {  // stopped while waiting: drop the chunk
            lock.unlock()
            return
        }
        let myGeneration = generation
        queuedSamples += chunk.count
        let startSample = scheduledSamples
        scheduledSamples += chunk.count
        let measure = real && firstRealBufferPending
        if measure {
            firstRealBufferPending = false
            _timing.firstBufferScheduled = Date()
            _timing.firstBufferSamples = chunk.count
            _timing.routeWasOpen = routeWasOpen
        }
        lock.unlock()
        node.scheduleBuffer(buf, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            guard let self else { return }
            let playedBack = Date()
            // Render clock: sample `startSample` of this player left the node at host time
            // `hostTime(render) - (sampleTime(render) - startSample) / rate`; add the output latency.
            var onset: Date?
            if measure, let nodeTime = self.node.lastRenderTime, let playerTime = self.node.playerTime(forNodeTime: nodeTime),
               playerTime.isSampleTimeValid, playerTime.isHostTimeValid {
                let renderSeconds = AVAudioTime.seconds(forHostTime: playerTime.hostTime)
                let nowSeconds = AVAudioTime.seconds(forHostTime: mach_absolute_time())
                let startSeconds = renderSeconds - (Double(playerTime.sampleTime) - Double(startSample)) / playerTime.sampleRate
                var latency = 0.0
                #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
                    latency = AVAudioSession.sharedInstance().outputLatency
                #endif
                onset = playedBack.addingTimeInterval(startSeconds + latency - nowSeconds)
            }
            self.lock.lock()
            if measure {
                self._timing.firstBufferPlayedBack = playedBack
                self._timing.onsetFromClock = onset
                self._timing.onsetBySubtraction = playedBack.addingTimeInterval(-Double(chunk.count) / Double(self.sampleRate))
            }
            if self.generation == myGeneration {
                self.queuedSamples = max(0, self.queuedSamples - chunk.count)
                self.lock.broadcast()
            }
            self.lock.unlock()
        }
    }

    /// Block until everything queued so far has been played, then stop the node.
    public func drain() {
        lock.lock()
        let deadline = Date(timeIntervalSinceNow: Double(queuedSamples) / Double(sampleRate) + 1.0)
        while started, queuedSamples > 0, lock.wait(until: deadline) {}
        lock.unlock()
        stop()
    }

    /// Drop whatever is queued and stop immediately. May be called from any thread.
    public func stop() {
        lock.lock()
        generation += 1
        queuedSamples = 0
        let wasStarted = started
        started = false
        lock.broadcast()
        lock.unlock()
        if wasStarted {
            node.stop()
            engine.stop()
        }
    }

    deinit {
        stop()
    }
}
