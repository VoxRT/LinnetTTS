import CryptoKit
import Foundation
import XCTest
import LinnetTTS

/// Tests against the real model files. They are looked up, in order, in `$LINNET_ASSETS`
/// (`TEST_RUNNER_LINNET_ASSETS=... xcodebuild test` forwards it to the test process), in the
/// test bundle's `Assets/` resource folder (copy the files into Tests/LinnetTTSTests/Assets/ for
/// a device run), and in `<repo>/local/` next to this source tree (host and simulator runs).
/// Without the files the model tests are skipped, not failed.
///
/// The same file is compiled into `LinnetSampleTests` of ../../Sample (hosted by the sample
/// app), because a Swift package test bundle cannot run on a physical device by itself.
final class LinnetTTSTests: XCTestCase {
    struct Assets {
        let weights: String
        let tokenizer: String
        let voice: String
    }

    static let assets: Assets? = {
        var dirs: [String] = []
        if let env = ProcessInfo.processInfo.environment["LINNET_ASSETS"], !env.isEmpty { dirs.append(env) }
        #if SWIFT_PACKAGE
            let bundle = Bundle.module
        #else
            let bundle = Bundle(for: LinnetTTSTests.self)  // hosted by the sample app (device runs)
        #endif
        if let res = bundle.resourceURL?.appendingPathComponent("Assets").path { dirs.append(res) }
        let here = URL(fileURLWithPath: #filePath)
        dirs.append(here.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("local").path)
        let fm = FileManager.default
        for d in dirs {
            let w = d + "/linnet-en-35m.fp16.safetensors"
            let t = d + "/tokenizer.model"
            guard fm.fileExists(atPath: w), fm.fileExists(atPath: t) else { continue }
            for v in ["voice.npy", "voices/linnet-en-f1.npy"] where fm.fileExists(atPath: d + "/" + v) {
                return Assets(weights: w, tokenizer: t, voice: d + "/" + v)
            }
        }
        return nil
    }()

    /// One shared model per process (loading takes a few hundred ms).
    static var sharedModel: LinnetModel?

    func loadModel() throws -> (LinnetModel, Voice) {
        guard let a = Self.assets else {
            throw XCTSkip("model assets not found (set LINNET_ASSETS or copy them to Tests/LinnetTTSTests/Assets/)")
        }
        if Self.sharedModel == nil {
            Self.sharedModel = try LinnetModel(weightsPath: a.weights, tokenizerPath: a.tokenizer)
        }
        let model = Self.sharedModel!
        return (model, try model.loadVoice(path: a.voice))
    }

    let text = "Hello there, this is a streaming test."

    func report(_ s: String) {
        print("[linnet] " + s)
        NSLog("[linnet] %@", s)
    }

    func md5(_ audio: [Float]) -> String {
        audio.withUnsafeBufferPointer { buf in
            Insecure.MD5.hash(data: Data(buffer: buf)).map { String(format: "%02x", $0) }.joined()
        }
    }

    func testVersionAndErrorOnMissingFile() {
        XCTAssertFalse(LinnetModel.version.isEmpty)
        XCTAssertThrowsError(try LinnetModel(weightsPath: "/nonexistent/weights.safetensors", tokenizerPath: "/nonexistent/tok.model")) { e in
            guard let le = e as? LinnetError else { return XCTFail("not a LinnetError: \(e)") }
            XCTAssertEqual(le.code, -2)
            if case .io = le {} else { XCTFail("expected .io, got \(le)") }
        }
        XCTAssertThrowsError(try Voice.load(path: "/nonexistent/voice.npy")) { e in
            XCTAssertEqual((e as? LinnetError)?.code, -2)
        }
        XCTAssertThrowsError(try Voice(latents: [Float](repeating: 0, count: 10), frames: 1, dim: 48)) { e in
            XCTAssertEqual((e as? LinnetError)?.code, -1)
        }
    }

    func testConstantsAndVoiceRoundTrip() throws {
        let (model, voice) = try loadModel()
        XCTAssertEqual(model.sampleRate, 24000)
        XCTAssertEqual(model.samplesPerStep, 1920)
        XCTAssertEqual(voice.dim, 48)
        XCTAssertGreaterThan(voice.frames, 0)
        // Rebuild the voice from the .npy payload and check the audio is identical.
        let data = try Data(contentsOf: URL(fileURLWithPath: Self.assets!.voice))
        let headerLen = Int(data[8]) | Int(data[9]) << 8
        let payload = data.subdata(in: (10 + headerLen)..<data.count)
        let latents: [Float] = payload.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        XCTAssertEqual(latents.count, voice.frames * voice.dim)
        let v2 = try Voice(latents: latents, frames: voice.frames)
        let p = SynthParams(maxFrames: 40, seed: 3)
        let a = try model.synthesize(text: text, voice: voice, params: p)
        let b = try model.synthesize(text: text, voice: v2, params: p)
        XCTAssertEqual(a, b)
    }

    func testSynthesizeLengthAndStreamIdentity() throws {
        let (model, voice) = try loadModel()
        let p = SynthParams(seed: 0)
        let offline = try model.synthesize(text: text, voice: voice, params: p)
        XCTAssertGreaterThan(offline.count, 0)
        XCTAssertEqual(offline.count % model.samplesPerStep, 0, "length must be steps * samplesPerStep")

        let s = try model.session(voice: voice, params: p)
        try s.prime(text: text)
        var chunks: [[Float]] = []
        while let c = try s.step() { chunks.append(c) }
        XCTAssertTrue(s.finished)
        XCTAssertNil(try s.step())
        XCTAssertEqual(chunks.first?.count, model.samplesPerStep - model.outputDelay)
        XCTAssertEqual(chunks.last?.count, model.outputDelay)
        XCTAssertTrue(chunks.last!.allSatisfy { $0 == 0 })
        for c in chunks.dropFirst().dropLast() { XCTAssertEqual(c.count, model.samplesPerStep) }
        let streamed = chunks.flatMap { $0 }
        XCTAssertEqual(streamed.count, offline.count)
        XCTAssertEqual(streamed, offline, "stream concatenation must equal the offline result bit for bit")
        XCTAssertEqual(offline.count, (chunks.count - 1) * model.samplesPerStep)
        report("steps \(chunks.count - 1), \(offline.count) samples, md5 \(md5(offline))")

        // Prime again on the same session: same result.
        try s.prime(text: text)
        XCTAssertEqual(try s.remaining(), offline)
    }

    func testThreadsAreBitExactAndSeedIsDeterministic() throws {
        let (model, voice) = try loadModel()
        let t1 = try model.synthesize(text: text, voice: voice, params: SynthParams(seed: 7, threads: 1))
        XCTAssertEqual(model.threads, 1)
        let t4 = try model.synthesize(text: text, voice: voice, params: SynthParams(seed: 7, threads: 4))
        XCTAssertEqual(model.threads, 4)
        let tAuto = try model.synthesize(text: text, voice: voice, params: SynthParams(seed: 7, threads: 0))
        XCTAssertGreaterThan(model.threads, 0, "automatic resolves to a positive thread count")
        XCTAssertEqual(t1, t4, "threads 1 vs 4 must be bit-exact")
        XCTAssertEqual(t1, tAuto, "threads 1 vs auto must be bit-exact")
        let again = try model.synthesize(text: text, voice: voice, params: SynthParams(seed: 7))
        XCTAssertEqual(t1, again, "same seed must give the same audio")
        let other = try model.synthesize(text: text, voice: voice, params: SynthParams(seed: 8))
        XCTAssertNotEqual(t1, other, "a different seed must change the audio")
        model.threads = 2
        XCTAssertEqual(model.threads, 2)
        model.threads = 0
        XCTAssertLessThanOrEqual(model.threads, 4)
    }

    /// `loadVoice` prepares the voice; a voice loaded without the model does not get prepared until
    /// `prepare(voice:)`. Prepared, unprepared and re-primed sessions give the same bytes.
    func testPreparedVoiceIsBitExact() throws {
        let (model, voice) = try loadModel()
        XCTAssertTrue(model.isPrepared(voice: voice))
        let raw = try Voice.load(path: Self.assets!.voice)
        XCTAssertFalse(model.isPrepared(voice: raw))
        let params = SynthParams(seed: 4)
        let plain = try model.synthesize(text: text, voice: raw, params: params)
        XCTAssertEqual(try model.synthesize(text: text, voice: voice, params: params), plain)
        let s = try model.session(voice: voice, params: params)
        try s.prime(text: "Hello there.")
        _ = try s.remaining()
        try s.prime(text: text)
        XCTAssertEqual(try s.remaining(), plain, "re-primed session")
        try model.prepare(voice: raw)
        XCTAssertTrue(model.isPrepared(voice: raw))
        XCTAssertEqual(try model.synthesize(text: text, voice: raw, params: params), plain, "prepared later")
        let other = try LinnetModel(weightsPath: Self.assets!.weights, tokenizerPath: Self.assets!.tokenizer)
        XCTAssertFalse(model.isPrepared(voice: voice) && other.isPrepared(voice: voice), "a state belongs to one model")
        XCTAssertEqual(try other.synthesize(text: text, voice: voice, params: params), plain, "another model falls back")
    }

    /// Timing per thread count: prime, first chunk, mean and worst step, RTF. Printed, not asserted.
    func testPerformanceReport() throws {
        let (model, voice) = try loadModel()
        let long = "The quick brown fox jumps over the lazy dog, then settles down for a nap in the afternoon sun."
        var reference: [Float]?
        for threads in [1, 0, 4] {
            let s = try model.session(voice: voice, params: SynthParams(seed: 0, threads: threads))
            let t0 = Date()
            try s.prime(text: long)
            let primeMs = Date().timeIntervalSince(t0) * 1000
            var firstMs = -1.0
            var worst = 0.0
            var stepMs: [Double] = []
            var audio: [Float] = []
            while true {
                let ts = Date()
                guard let c = try s.step() else { break }
                let dt = Date().timeIntervalSince(ts) * 1000
                if firstMs < 0 { firstMs = Date().timeIntervalSince(t0) * 1000 }
                if c.count == model.samplesPerStep { stepMs.append(dt) }  // full steps only
                worst = max(worst, dt)
                audio.append(contentsOf: c)
            }
            let total = Date().timeIntervalSince(t0) * 1000
            let audioMs = Double(audio.count) / Double(model.sampleRate) * 1000
            let mean = stepMs.reduce(0, +) / Double(max(1, stepMs.count))
            report(String(format: "threads %d (effective %d): prime %.1f ms, first chunk %.1f ms, step mean %.2f ms worst %.2f ms over %d steps, RTF %.1f %% (%.0f ms for %.0f ms audio)",
                          threads, model.threads == 0 ? -1 : model.threads, primeMs, firstMs, mean, worst, stepMs.count, 100 * total / audioMs, total, audioMs))
            if let r = reference { XCTAssertEqual(r, audio) } else { reference = audio }
        }
        report("perf md5 \(md5(reference!))")
    }

    /// Ten utterances in a row (seed i): sustained speed and a checksum over all audio.
    func testSustainedTenUtterances() throws {
        let (model, voice) = try loadModel()
        let sentences = [
            "Good morning, and welcome to the show.",
            "The train leaves at half past seven.",
            "Please remember to water the plants on Friday.",
            "Streaming synthesis keeps every operator causal.",
            "A short sentence.",
            "Mountains rise beyond the quiet valley, and rivers run to the sea.",
            "Numbers like twelve, forty, and one hundred are spelled out.",
            "The meeting has been moved to the second floor.",
            "Thank you for listening; that is all for today.",
            "One more utterance closes the sustained run.",
        ]
        var hasher = Insecure.MD5()
        var totalMs = 0.0
        var audioSamples = 0
        var steps = 0
        let t0 = Date()
        for (i, sentence) in sentences.enumerated() {
            let ts = Date()
            let a = try model.synthesize(text: sentence, voice: voice, params: SynthParams(seed: UInt64(i)))
            totalMs += Date().timeIntervalSince(ts) * 1000
            audioSamples += a.count
            steps += a.count / model.samplesPerStep
            a.withUnsafeBufferPointer { hasher.update(data: Data(buffer: $0)) }
        }
        let wall = Date().timeIntervalSince(t0) * 1000
        let audioMs = Double(audioSamples) / Double(model.sampleRate) * 1000
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        report(String(format: "sustained 10 utterances (threads auto): %d steps, %.1f s audio in %.2f s (%.2f ms/step, RTF %.1f %%), wall %.2f s, md5 %@",
                      steps, audioMs / 1000, totalMs / 1000, totalMs / Double(steps), 100 * totalMs / audioMs, wall / 1000, digest))
    }
}
