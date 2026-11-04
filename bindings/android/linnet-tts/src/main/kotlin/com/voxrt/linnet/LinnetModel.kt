package com.voxrt.linnet

import java.io.File

/**
 * Loaded weights and tokenizer.
 *
 * The model is immutable and may be shared by any number of threads and sessions; the only
 * rule is not to call [close] while another thread is still using it (sessions created from
 * the model keep the weights alive on their own, so closing the model before its sessions is
 * safe). Loading takes a few hundred milliseconds and should happen off the main thread.
 *
 * Audio is mono float32 at [sampleRate] (24 kHz), in `[-1, 1]`.
 */
class LinnetModel private constructor(handle: Long) : AutoCloseable {
    /** Load `weightsPath` (safetensors, fp16 or fp32) and `tokenizerPath` (SentencePiece `.model`). */
    constructor(weightsPath: String, tokenizerPath: String) : this(LinnetNative.modelLoad(weightsPath, tokenizerPath))

    constructor(weights: File, tokenizer: File) : this(weights.absolutePath, tokenizer.absolutePath)

    @Volatile
    private var handle: Long = handle

    internal val nativeHandle: Long
        get() = handle.also { check(it != 0L) { "LinnetModel is closed" } }

    /** Output sample rate in Hz. */
    val sampleRate: Int = LinnetNative.sampleRate(nativeHandle)

    /** Samples produced per LM step (two codec frames, 1920 = 80 ms). Largest chunk a session returns. */
    val samplesPerStep: Int = LinnetNative.samplesPerStep(nativeHandle)

    /** Width of one LM latent and of a voice row (48). */
    val latentDim: Int = LinnetNative.latentDim(nativeHandle)

    /** Delay compensated by the streaming session, in samples (720): the first chunk is this much shorter. */
    val outputDelay: Int = LinnetNative.outputDelay(nativeHandle)

    /** SentencePiece token ids of [text]. */
    fun tokenize(text: String): IntArray = LinnetNative.tokenize(nativeHandle, text)

    /** Load `[T, latentDim]` latents from a `.npy` file and check the width against the model. */
    fun loadVoice(path: String): Voice = checkVoice(Voice.load(path))

    fun loadVoice(file: File): Voice = loadVoice(file.absolutePath)

    /** Parse a `.npy` image from memory and check the width against the model. */
    fun voiceFromNpy(npy: ByteArray): Voice = checkVoice(Voice.fromNpy(npy))

    private fun checkVoice(v: Voice): Voice {
        if (v.dim != latentDim) {
            val d = v.dim
            v.close()
            throw LinnetException(LinnetException.INVALID_ARG, "voice latents have width $d, the model expects $latentDim")
        }
        return v
    }

    /**
     * Whole utterance in one call. Blocks for the full synthesis; the result is
     * `frames * samplesPerStep` samples and equals the concatenation of the chunks a
     * [LinnetSession] with the same [params] produces.
     */
    @JvmOverloads
    fun synthesize(text: String, voice: Voice, params: SynthParams = SynthParams()): FloatArray =
        LinnetNative.synthesize(
            nativeHandle, voice.nativeHandle, text,
            params.temp, params.cfg, params.maxFrames, params.eosThreshold, params.seed, params.threads,
        )

    /** Streaming session bound to [voice]. Call [LinnetSession.prime] before stepping. */
    @JvmOverloads
    fun session(voice: Voice, params: SynthParams = SynthParams()): LinnetSession =
        LinnetSession(
            LinnetNative.sessionNew(
                nativeHandle, voice.nativeHandle,
                params.temp, params.cfg, params.maxFrames, params.eosThreshold, params.seed, params.threads,
            ),
            samplesPerStep,
        )

    /** Release the weights. Sessions already created stay usable. Idempotent. */
    override fun close() {
        val h = handle
        handle = 0L
        if (h != 0L) LinnetNative.modelFree(h)
    }

    override fun toString() =
        if (handle == 0L) "LinnetModel(closed)"
        else "LinnetModel(sampleRate=$sampleRate, samplesPerStep=$samplesPerStep, latentDim=$latentDim)"

    companion object {
        /** Version of the native runtime. */
        @JvmStatic
        val version: String
            get() = LinnetNative.version()

        /**
         * Build a model from in-memory images, e.g. `assets.open("linnet-en-35m.fp16.safetensors").readBytes()`.
         * The weights are copied once more on the native side; for the 83 MB fp16 file prefer copying the
         * assets to `filesDir` on first run and using the path constructor.
         */
        @JvmStatic
        fun fromBytes(weights: ByteArray, tokenizer: ByteArray): LinnetModel =
            LinnetModel(LinnetNative.modelFromBytes(weights, tokenizer))
    }
}
