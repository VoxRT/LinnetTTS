package com.voxrt.linnet

/**
 * Raw JNI surface of `liblinnet_tts_jni.so`. Handles are opaque `Long`s owned by the
 * wrapper classes in this package; every function that can fail throws [LinnetException].
 *
 * These are instance methods of the Kotlin object (not `@JvmStatic`), which is what the
 * native symbols `Java_com_voxrt_linnet_LinnetNative_*` expect. Keep rules: see
 * `consumer-rules.pro`.
 */
internal object LinnetNative {
    init {
        System.loadLibrary("linnet_tts_jni")
    }

    external fun version(): String

    // Model
    external fun modelLoad(weightsPath: String, tokenizerPath: String): Long
    external fun modelFromBytes(weights: ByteArray, tokenizer: ByteArray): Long
    external fun modelFree(model: Long)
    external fun sampleRate(model: Long): Int
    external fun samplesPerStep(model: Long): Int
    external fun latentDim(model: Long): Int
    external fun outputDelay(model: Long): Int
    external fun tokenize(model: Long, text: String): IntArray
    external fun synthesize(
        model: Long,
        voice: Long,
        text: String,
        temp: Float,
        cfg: Float,
        maxFrames: Int,
        eosThreshold: Float,
        seed: Long,
        threads: Int,
    ): FloatArray

    // Voice
    external fun voiceLoad(path: String): Long
    external fun voiceFromNpy(npy: ByteArray): Long
    external fun voiceFromLatents(latents: FloatArray, frames: Int, dim: Int): Long
    external fun voiceFree(voice: Long)
    external fun voiceFrames(voice: Long): Int
    external fun voiceDim(voice: Long): Int
    external fun voiceLatents(voice: Long): FloatArray

    // Session
    external fun sessionNew(
        model: Long,
        voice: Long,
        temp: Float,
        cfg: Float,
        maxFrames: Int,
        eosThreshold: Float,
        seed: Long,
        threads: Int,
    ): Long
    external fun sessionFree(session: Long)
    external fun sessionPrime(session: Long, text: String)
    /** Writes the next chunk into [out]; returns the sample count, or -1 when the utterance is over. */
    external fun sessionStep(session: Long, out: FloatArray): Int
    external fun sessionFrames(session: Long): Int
    external fun sessionFinished(session: Long): Boolean
    external fun sessionLatent(session: Long): FloatArray
}
