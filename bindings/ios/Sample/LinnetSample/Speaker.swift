import Foundation
import LinnetTTS

/// Loads the model from the app's Documents directory and streams utterances into a
/// `LinnetPlayer` on a background queue. All published state changes land on the main thread.
///
/// Launch arguments (for driving the app without touching the screen, e.g. from
/// `xcrun simctl launch` / `xcrun devicectl device process launch`):
///   `-autospeak "<text>"`   speak the text as soon as the model is loaded, print the timings
///   `-nopace` / `-bufferms N` do not block on playback (pure compute timing) / queue N ms ahead (default 8000)
///                           to the log and exit;
///   `-threads N`            thread count for the auto run (default 0 = automatic);
///   `-repeat N`             repeat the utterance N times (default 1) and report the sum.
final class Speaker: ObservableObject {
    @Published var status = "loading..."
    @Published var ready = false
    @Published var busy = false
    @Published var threads = 0

    /// Synthesis queue; `-qos interactive|initiated|utility|background` selects its QoS class (default: user-initiated).
    private lazy var queue: DispatchQueue = {
        let q: DispatchQoS
        switch UserDefaults.standard.string(forKey: "qos") ?? "initiated" {
        case "interactive": q = .userInteractive
        case "utility": q = .utility
        case "background": q = .background
        default: q = .userInitiated
        }
        return DispatchQueue(label: "linnet.speaker", qos: q)
    }()
    /// `-quiet`: no per-step status updates (isolates UI work from synthesis timing).
    private var quiet = false
    /// `-noaudio`: synthesize without creating the audio engine at all (pure compute timing).
    private var noAudio = false
    /// Audio kept queued ahead of playback; `-nopace` makes it effectively unlimited (`-bufferms N` sets it).
    private var bufferMillis = 8000
    private var model: LinnetModel?
    private var voice: Voice?
    private var player: LinnetPlayer?
    private var cancelled = false
    private var started = false

    static let files = ["linnet-en-35m.fp16.safetensors", "tokenizer.model", "voice.npy"]

    static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    func start() {
        guard !started else { return }
        started = true
        let args = UserDefaults.standard
        let auto = args.string(forKey: "autospeak")
        if let t = args.object(forKey: "threads") as? String, let n = Int(t) { threads = n }
        let repeats = max(1, args.integer(forKey: "repeat"))
        let flags = Set(CommandLine.arguments)                                   // bare flags are not key/value pairs
        quiet = flags.contains("-quiet")
        noAudio = flags.contains("-noaudio")
        if flags.contains("-nopace") { bufferMillis = 600_000 }                 // never block on the player: measures pure compute
        else if let b = args.object(forKey: "bufferms") as? String, let n = Int(b), n > 0 { bufferMillis = n }
        queue.async { [self] in
            let dir = Self.documents
            let missing = Self.files.filter { !FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path) }
            if !missing.isEmpty {
                post("missing in Documents: \(missing.joined(separator: ", "))\n\(dir.path)")
                log("missing model files: \(missing)")
                if auto != nil { exit(2) }
                return
            }
            do {
                let t0 = Date()
                let m = try LinnetModel(weightsURL: dir.appendingPathComponent(Self.files[0]),
                                        tokenizerURL: dir.appendingPathComponent(Self.files[1]))
                let v = try m.loadVoice(url: dir.appendingPathComponent(Self.files[2]))
                let ms = Date().timeIntervalSince(t0) * 1000
                model = m
                voice = v
                let msg = String(format: "model %@ loaded in %.0f ms (%d Hz, %d samples/step, %d threads)",
                                 LinnetModel.version, ms, m.sampleRate, m.samplesPerStep, m.threads)
                log(msg)
                DispatchQueue.main.async { self.ready = true; self.status = msg }
                if let text = auto {
                    let thr = self.threads
                    DispatchQueue.main.async { self.busy = true }
                    var lines: [String] = []
                    for _ in 0..<repeats {
                        lines.append(try run(text: text, threads: thr))
                    }
                    log("autospeak done\n" + lines.joined(separator: "\n"))
                    DispatchQueue.main.async { self.busy = false }
                    exit(0)
                }
            } catch {
                post("load failed: \(error)")
                log("load failed: \(error)")
                if auto != nil { exit(1) }
            }
        }
    }

    func speak(_ text: String) {
        guard ready, !busy else { return }
        busy = true
        cancelled = false
        let thr = threads
        queue.async { [self] in
            do {
                let line = try run(text: text, threads: thr)
                post(line)
            } catch {
                post("error: \(error)")
            }
            DispatchQueue.main.async { self.busy = false }
        }
    }

    func stop() {
        cancelled = true
        player?.stop()
    }

    /// Streams one utterance through the player; returns the status line.
    private func run(text: String, threads: Int) throws -> String {
        guard let model, let voice else { return "model not loaded" }
        let player: LinnetPlayer? = noAudio ? nil : try LinnetPlayer(sampleRate: model.sampleRate, bufferMillis: bufferMillis)
        self.player = player
        let session = try model.session(voice: voice, params: SynthParams(seed: 0, threads: threads))
        let t0 = Date()
        try session.prime(text: text)
        let primeMs = Date().timeIntervalSince(t0) * 1000
        var firstMs = -1.0
        var worst = 0.0
        var stepTotal = 0.0
        var steps = 0
        var samples = 0
        while !cancelled {
            let ts = Date()
            guard let chunk = try session.step() else { break }
            let dt = Date().timeIntervalSince(ts) * 1000
            if firstMs < 0 { firstMs = Date().timeIntervalSince(t0) * 1000 }
            worst = max(worst, dt)
            stepTotal += dt
            if chunk.count == model.samplesPerStep { steps += 1 }
            samples += chunk.count
            if !quiet { post(String(format: "speaking... %d steps, step %.1f ms", steps, dt)) }
            try player?.write(chunk)
        }
        if let player { if cancelled { player.stop() } else { player.drain() } }
        let audioMs = Double(samples) / Double(model.sampleRate) * 1000
        let line = String(
            format: "%@qos %@%@ threads %d: prime %.0f ms, first chunk %.0f ms, step mean %.1f ms, worst %.1f ms (80 ms budget)\n%d steps, %.0f ms audio in %.0f ms: RTF %.1f %% (playback pacing excluded)",
            cancelled ? "stopped. " : "", UserDefaults.standard.string(forKey: "qos") ?? "initiated", (noAudio ? " noaudio" : "") + (quiet ? " quiet" : "") + (bufferMillis > 10_000 ? " nopace" : ""), model.threads, primeMs, firstMs, stepTotal / Double(max(1, steps + 1)), worst,
            steps, audioMs, primeMs + stepTotal, 100 * (primeMs + stepTotal) / max(1, audioMs))
        log(line)
        return line
    }

    private func post(_ s: String) {
        DispatchQueue.main.async { self.status = s }
    }

    private func log(_ s: String) {
        NSLog("[linnet-sample] %@", s)
        print("[linnet-sample] " + s)
    }
}
