package com.voxrt.linnet

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTimestamp
import android.media.AudioTrack
import android.os.Build

/**
 * Minimal streaming playback helper: a mono float32 [AudioTrack] in streaming mode that accepts the
 * chunks of a [LinnetSession] as they are produced.
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
 * [write] queues the chunk and returns; a feeder thread hands the queue to the track. [write] blocks only
 * while more than [bufferMillis] of audio is queued (default 8 s), so a typical utterance is synthesised in
 * one burst at full speed and playback drains behind it. Do not pace synthesis to real time with a small
 * queue: a thread that sleeps most of every 80 ms step is clocked down and moved to little cores by the
 * kernel's frequency governor, and each step then takes several times longer (measured on iOS: 22 ms
 * instead of 6.7 ms per step; the same mechanism applies to Android).
 *
 * The track itself is small (about 160 ms) and asks for the low-latency performance mode, so it runs on
 * the mixer's fast path and starts as soon as the first chunk is in. A large track buffer must not be used
 * here: Qualcomm's audio policy routes PCM tracks with a buffer of several seconds to a "direct PCM" output
 * that waits for the whole buffer to fill before it plays, so short utterances never come out (observed on
 * a Snapdragon 662: the 0.1.x player, with an 8 s track buffer, played nothing there).
 *
 * The player is not thread-safe; use it from the thread that drives the session ([stopNow] may be called
 * from any thread).
 *
 * @param sampleRate output rate, [LinnetModel.sampleRate].
 * @param bufferMillis audio kept queued ahead of the track (default 8000 ms); smaller values only make
 *   [stopNow] drop less audio, they do not reduce latency to the first chunk.
 */
class AudioTrackPlayer(private val sampleRate: Int, bufferMillis: Int = 8000) : AutoCloseable {
    /**
     * When the first chunk written through [write] went through the output, for latency measurements.
     * Times are `System.nanoTime()` values. [onsetNanos] and [firstChunkPlayedNanos] are extrapolated from
     * the first [AudioTimestamp] the track reported (a frame position and the time it was presented at the
     * device), so they are only available once [timestampNanos] is set, which happens during playback;
     * [headPassedFirstFrameNanos] / [headPassedFirstChunkNanos] come from polling the playback head every
     * millisecond and are the mixer-side view, without the device's output latency.
     */
    class PlaybackTiming internal constructor(
        /** `System.nanoTime()` when the first chunk was handed to the track (after `play()` opened the route). */
        val firstWriteNanos: Long,
        /** Frame index of the first chunk's first sample (after any prewarm silence). */
        val firstFrame: Long,
        /** Samples in the first chunk. */
        val firstChunkFrames: Int,
        /** `true` when the track was already playing before the first [write] ([prewarm]). */
        val routeWasOpen: Boolean,
        private val sampleRate: Int,
    ) {
        /** Frame position and presentation time of the first timestamp the track reported (`-1` so far). */
        @Volatile var timestampFrame: Long = -1
            internal set
        @Volatile var timestampNanos: Long = -1
            internal set
        @Volatile var headPassedFirstFrameNanos: Long? = null
            internal set
        @Volatile var headPassedFirstChunkNanos: Long? = null
            internal set

        /** `System.nanoTime()` at which [frame] was (or will be) presented, from the first timestamp; `null` without one. */
        fun nanosAtFrame(frame: Long): Long? =
            if (timestampNanos < 0) null else timestampNanos + (frame - timestampFrame) * 1_000_000_000L / sampleRate

        /** When the first chunk's first sample reached the output. */
        val onsetNanos: Long? get() = nanosAtFrame(firstFrame)

        /** When the first chunk's last sample had left the output. */
        val firstChunkPlayedNanos: Long? get() = nanosAtFrame(firstFrame + firstChunkFrames)
    }

    private class Item(val data: FloatArray, val count: Int, val real: Boolean)

    private val track: AudioTrack
    private val maxQueuedSamples = sampleRate.toLong() * bufferMillis.coerceAtLeast(1) / 1000
    private val lock = Object()
    private val queue = ArrayDeque<Item>()
    private var queuedSamples = 0L
    private var closed = false
    private var generation = 0
    private var framesWritten = 0L
    private var started = false
    private var feederError: Throwable? = null
    private val feeder: Thread

    /** Timing of the first chunk written through [write], or `null` before it. */
    @Volatile
    var timing: PlaybackTiming? = null
        private set

    init {
        val minBytes = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val wantedBytes = sampleRate * TRACK_BUFFER_MILLIS / 1000 * 4
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
            .apply { if (Build.VERSION.SDK_INT >= 26) setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY) }
            .build()
        check(track.state == AudioTrack.STATE_INITIALIZED) { "AudioTrack failed to initialize" }
        feeder = Thread(::feed, "linnet-audio-feeder").apply { isDaemon = true; start() }
    }

    /**
     * Start the track and play 50 ms of silence, so that the output route is open before the first [write];
     * otherwise the first `write` opens it. For latency measurements of both cases (see [timing]).
     */
    fun prewarm() = enqueue(FloatArray(sampleRate / 20), sampleRate / 20, real = false)

    /** Queue [count] samples of [chunk]; blocks only while more than `bufferMillis` of audio is queued. */
    @JvmOverloads
    fun write(chunk: FloatArray, count: Int = chunk.size) {
        if (count <= 0) return
        enqueue(chunk.copyOf(count), count, real = true)
    }

    private fun enqueue(data: FloatArray, count: Int, real: Boolean) {
        synchronized(lock) {
            while (!closed && queuedSamples > 0 && queuedSamples + count > maxQueuedSamples) lock.wait()
            feederError?.let { throw IllegalStateException("AudioTrack feeder failed", it) }
            check(!closed) { "AudioTrackPlayer is closed" }
            queue.addLast(Item(data, count, real))
            queuedSamples += count
            lock.notifyAll()
        }
    }

    /** The feeder thread: hands queued chunks to the track, blocking on the track's small buffer. */
    private fun feed() {
        while (true) {
            val item: Item
            val gen: Int
            synchronized(lock) {
                while (queue.isEmpty() && !closed) lock.wait()
                if (closed && queue.isEmpty()) return
                item = queue.removeFirst()
                gen = generation
            }
            try {
                val routeWasOpen = started
                if (!started) {
                    track.play()
                    started = true
                }
                if (item.real && timing == null) {
                    val t = PlaybackTiming(System.nanoTime(), framesWritten, item.count, routeWasOpen, sampleRate)
                    timing = t
                    pollHead(t)
                }
                var off = 0
                while (off < item.count) {
                    val n = track.write(item.data, off, item.count - off, AudioTrack.WRITE_BLOCKING)
                    if (n < 0) throw IllegalStateException("AudioTrack.write failed: $n")
                    if (n == 0) break // paused or stopped underneath us: drop the rest
                    off += n
                    synchronized(lock) { if (generation != gen) off = item.count }
                }
                pollTimestamp()
            } catch (e: Throwable) {
                synchronized(lock) { feederError = e; lock.notifyAll() }
                return
            } finally {
                synchronized(lock) {
                    if (generation == gen) framesWritten += item.count
                    queuedSamples -= item.count
                    lock.notifyAll()
                }
            }
        }
    }

    /** Watch the playback head cross the first chunk, on a short-lived thread (at most two seconds). */
    private fun pollHead(t: PlaybackTiming) {
        val end = t.firstFrame + t.firstChunkFrames
        Thread {
            val deadline = System.nanoTime() + 2_000_000_000L
            while (System.nanoTime() < deadline && t.headPassedFirstChunkNanos == null) {
                val head = try {
                    track.playbackHeadPosition.toLong() and 0xffffffffL
                } catch (_: IllegalStateException) {
                    return@Thread // the track was released
                }
                val now = System.nanoTime()
                if (t.headPassedFirstFrameNanos == null && head > t.firstFrame) t.headPassedFirstFrameNanos = now
                if (head >= end) t.headPassedFirstChunkNanos = now
                try { Thread.sleep(1) } catch (_: InterruptedException) { return@Thread }
            }
        }.apply { isDaemon = true; name = "linnet-head-poll" }.start()
    }

    /** Record the first timestamp the track reports (frame position + presentation time). */
    private fun pollTimestamp() {
        val t = timing ?: return
        if (t.timestampNanos >= 0) return
        val ts = AudioTimestamp()
        try {
            if (track.getTimestamp(ts) && ts.framePosition > 0) {
                t.timestampFrame = ts.framePosition
                t.timestampNanos = ts.nanoTime
            }
        } catch (_: IllegalStateException) {
        }
    }

    /** Block until everything queued so far has been played, then stop. */
    fun drain() {
        synchronized(lock) {
            while (queuedSamples > 0 && feederError == null && !closed) lock.wait()
            feederError?.let { throw IllegalStateException("AudioTrack feeder failed", it) }
        }
        if (!started) return
        // A short silent pad so the tail is not cut by the output latency, then wait for the playback head
        // to reach the last frame (or to stop advancing: some devices never report the exact count).
        val total: Long
        synchronized(lock) { total = framesWritten }
        val pad = sampleRate / 10
        enqueue(FloatArray(pad), pad, real = false)
        synchronized(lock) { while (queuedSamples > 0 && feederError == null) lock.wait() }
        val deadline = System.nanoTime() + ((total + pad) * 1_000_000_000L / sampleRate) + 1_000_000_000L
        var last = -1L
        var stableSince = System.nanoTime()
        while (System.nanoTime() < deadline) {
            pollTimestamp()
            val head = track.playbackHeadPosition.toLong() and 0xffffffffL
            if (head >= total) break
            if (head != last) {
                last = head
                stableSince = System.nanoTime()
            } else if (System.nanoTime() - stableSince > 150_000_000L) {
                break
            }
            Thread.sleep(5)
        }
        track.stop()
        started = false
        synchronized(lock) { framesWritten = 0L }
        timing = null
    }

    /** Drop whatever is queued and stop immediately. May be called from any thread. */
    fun stopNow() {
        synchronized(lock) {
            generation++
            // The feeder subtracts the item it is writing when it returns; drop the rest here.
            queuedSamples -= queue.sumOf { it.count.toLong() }
            queue.clear()
            framesWritten = 0L
            lock.notifyAll()
        }
        if (started) {
            track.pause()
            track.flush()
            track.stop()
            started = false
        }
        timing = null
    }

    override fun close() {
        stopNow()
        synchronized(lock) { closed = true; lock.notifyAll() }
        feeder.join(2000)
        track.release()
    }

    private companion object {
        /** Buffer of the AudioTrack itself: two 80 ms chunks; the rest of the queue lives in [queue]. */
        const val TRACK_BUFFER_MILLIS = 160
    }
}
