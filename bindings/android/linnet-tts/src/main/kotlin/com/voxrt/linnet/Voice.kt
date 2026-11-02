package com.voxrt.linnet

import java.io.File

/**
 * Voice prompt latents, a `[frames, dim]` float32 matrix (dim = 48 for linnet-en-35m).
 *
 * Immutable and thread-safe once created. A session copies the latents at creation, so a
 * voice may be closed as soon as the sessions that need it exist. Closing twice is a no-op.
 */
class Voice private constructor(handle: Long) : AutoCloseable {
    @Volatile
    private var handle: Long = handle

    internal val nativeHandle: Long
        get() = handle.also { check(it != 0L) { "Voice is closed" } }

    /** Number of 80 ms frames. */
    val frames: Int get() = LinnetNative.voiceFrames(nativeHandle)

    /** Latent width. */
    val dim: Int get() = LinnetNative.voiceDim(nativeHandle)

    /** Copy of the latents, row-major `[frames * dim]`. */
    fun latents(): FloatArray = LinnetNative.voiceLatents(nativeHandle)

    override fun close() {
        val h = handle
        handle = 0L
        if (h != 0L) LinnetNative.voiceFree(h)
    }

    override fun toString() = if (handle == 0L) "Voice(closed)" else "Voice(frames=$frames, dim=$dim)"

    companion object {
        /** Load a NumPy `.npy` file (float32, C order). */
        @JvmStatic
        fun load(path: String): Voice = Voice(LinnetNative.voiceLoad(path))

        @JvmStatic
        fun load(file: File): Voice = load(file.absolutePath)

        /** Parse a `.npy` image held in memory (e.g. read from the app's assets). */
        @JvmStatic
        fun fromNpy(npy: ByteArray): Voice = Voice(LinnetNative.voiceFromNpy(npy))

        /** Wrap `[frames, dim]` latents given row-major in [latents] (`latents.size == frames * dim`). */
        @JvmStatic
        @JvmOverloads
        fun fromLatents(latents: FloatArray, frames: Int, dim: Int = 48): Voice {
            require(frames > 0 && dim > 0 && latents.size == frames * dim) {
                "latents has ${latents.size} values, expected frames * dim = $frames * $dim"
            }
            return Voice(LinnetNative.voiceFromLatents(latents, frames, dim))
        }
    }
}
