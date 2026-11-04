package com.voxrt.linnet

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack

/**
 * Minimal streaming playback helper: a mono float32 [AudioTrack] in streaming mode that
 * accepts the chunks of a [LinnetSession] as they are produced.
 *
 * Usage, on a background thread:
 * ```
 * model.session(voice).use { s ->
 *     AudioTrackPlayer(model.sampleRate).use { player ->
 *         s.prime(text)
 *         while (true) { val chunk = s.step() ?: break; player.write(chunk) }
 *         player.drain()
 *     }
 * }
 * ```
 * [write] blocks while the track buffer is full. The default buffer holds 8 s, so a typical utterance is
 * synthesised in one burst at full speed and playback drains behind it. Do not pace synthesis to real time
 * with a small buffer: a thread that sleeps most of every 80 ms step is clocked down and moved to little
 * cores by the kernel's frequency governor, and each step then takes several times longer (measured on
 * iOS: 22 ms instead of 6.7 ms per step; the same mechanism applies to Android). The player is not
 * thread-safe; use it from the thread that drives the session.
 *
 * @param sampleRate output rate, [LinnetModel.sampleRate].
 * @param bufferMillis size of the track buffer (default 8000 ms, 768 KB at 24 kHz float); smaller values only
 *   make [stopNow] drop less audio, they do not reduce latency to the first chunk.
 */
class AudioTrackPlayer(private val sampleRate: Int, bufferMillis: Int = 8000) : AutoCloseable {
    private val track: AudioTrack
    private var started = false
    private var framesWritten = 0L

    init {
        val minBytes = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val wantedBytes = sampleRate * bufferMillis / 1000 * 4
        track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            )
            .setBufferSizeInBytes(maxOf(minBytes, wantedBytes))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        check(track.state == AudioTrack.STATE_INITIALIZED) { "AudioTrack failed to initialize" }
    }

    /** Queue [count] samples of [chunk] (blocking). Playback starts on the first call. */
    @JvmOverloads
    fun write(chunk: FloatArray, count: Int = chunk.size) {
        if (count <= 0) return
        if (!started) {
            track.play()
            started = true
        }
        var off = 0
        while (off < count) {
            val n = track.write(chunk, off, count - off, AudioTrack.WRITE_BLOCKING)
            if (n < 0) throw IllegalStateException("AudioTrack.write failed: $n")
            off += n
        }
        framesWritten += count
    }

    /** Block until everything queued so far has been played, then stop. */
    fun drain() {
        if (!started) return
        // A short silent pad so the tail is not cut by the output latency. Then stop(): in streaming mode the
        // track keeps playing what was written, and the playback head advances until the last frame has been
        // rendered. Some devices never report the head reaching the exact number of frames written, so instead
        // of waiting for that count, wait until the head stops moving (bounded by the queued duration plus a
        // margin).
        write(FloatArray(sampleRate / 10))
        val total = framesWritten
        val deadline = System.nanoTime() + (total * 1_000_000_000L / sampleRate) + 1_000_000_000L
        track.stop()
        var last = -1L
        var stableSince = System.nanoTime()
        while (System.nanoTime() < deadline) {
            val head = track.playbackHeadPosition.toLong() and 0xffffffffL
            if (head >= total) break
            if (head != last) {
                last = head
                stableSince = System.nanoTime()
            } else if (System.nanoTime() - stableSince > 150_000_000L) {
                break // the head has stopped advancing: everything audible has been rendered
            }
            Thread.sleep(10)
        }
        started = false
        framesWritten = 0L
    }

    /** Drop whatever is queued and stop immediately. */
    fun stopNow() {
        if (started) {
            track.pause()
            track.flush()
            track.stop()
            started = false
            framesWritten = 0L
        }
    }

    override fun close() {
        stopNow()
        track.release()
    }
}
