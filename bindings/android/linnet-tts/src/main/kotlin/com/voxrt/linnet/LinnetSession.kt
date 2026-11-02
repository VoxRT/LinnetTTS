package com.voxrt.linnet

/**
 * One utterance in progress: streaming synthesis one LM step (80 ms) at a time.
 *
 * Not thread-safe: drive a session from one thread at a time (typically a dedicated
 * synthesis thread that writes to an [AudioTrackPlayer]). Different sessions of the same model
 * may run on different threads concurrently. The session keeps the model weights alive until
 * it is closed, so close sessions when done to free the KV caches (a few MB each).
 *
 * Chunk layout (see docs/streaming-contract.md): the first chunk is
 * `samplesPerStep - outputDelay` samples (1200), later ones `samplesPerStep` (1920), the last one
 * `outputDelay` (720) zeros; concatenated they equal [LinnetModel.synthesize] with the same
 * parameters sample for sample.
 */
class LinnetSession internal constructor(handle: Long, val samplesPerStep: Int) : AutoCloseable {
    @Volatile
    private var handle: Long = handle

    private val buffer = FloatArray(samplesPerStep)

    private val nativeHandle: Long
        get() = handle.also { check(it != 0L) { "LinnetSession is closed" } }

    /** Tokenize [text] and run the prefix. Resets any previous utterance; may be called again for a new text. */
    fun prime(text: String) = LinnetNative.sessionPrime(nativeHandle, text)

    /**
     * Next chunk of 24 kHz float32 audio as a fresh array, or `null` once the utterance is over
     * (after the final zero chunk). Allocation-free alternative: [stepInto].
     */
    fun step(): FloatArray? {
        val n = LinnetNative.sessionStep(nativeHandle, buffer)
        return if (n < 0) null else buffer.copyOf(n)
    }

    /**
     * Next chunk into [out] (`out.size >= samplesPerStep`). Returns the number of samples
     * written, or -1 when the utterance is over.
     */
    fun stepInto(out: FloatArray): Int {
        require(out.size >= samplesPerStep) { "out must hold at least $samplesPerStep samples" }
        return LinnetNative.sessionStep(nativeHandle, out)
    }

    /** LM steps emitted so far. */
    val frames: Int get() = LinnetNative.sessionFrames(nativeHandle)

    /** True once [step] has returned `null`. */
    val finished: Boolean get() = LinnetNative.sessionFinished(nativeHandle)

    /** Codec-space latent of the most recent step (`latentDim` values). */
    fun latent(): FloatArray = LinnetNative.sessionLatent(nativeHandle)

    /** Iterate the remaining chunks. */
    fun chunks(): Sequence<FloatArray> = generateSequence { step() }

    override fun close() {
        val h = handle
        handle = 0L
        if (h != 0L) LinnetNative.sessionFree(h)
    }
}
