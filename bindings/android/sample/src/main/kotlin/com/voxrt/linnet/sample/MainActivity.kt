package com.voxrt.linnet.sample

import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.voxrt.linnet.AudioTrackPlayer
import com.voxrt.linnet.LinnetModel
import com.voxrt.linnet.SynthParams
import com.voxrt.linnet.Voice
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * One screen: a text field, Speak, Stop, and a status line. Speak runs a streaming session on a
 * background thread and plays each 80 ms chunk through an AudioTrack as soon as it is ready.
 *
 * The model files are read from the app's private files directory (not bundled). Push them with:
 *
 *   adb push linnet-en-35m.fp16.safetensors tokenizer.model voice.npy /data/local/tmp/
 *   adb shell run-as com.voxrt.linnet.sample sh -c \
 *     'mkdir -p files && cp /data/local/tmp/linnet-en-35m.fp16.safetensors /data/local/tmp/tokenizer.model /data/local/tmp/voice.npy files/'
 *
 * (`voice.npy` is any `[T, 48]` voice file, e.g. voices/linnet-en-f1.npy renamed. `run-as` works
 * for debug builds; for a release build push to a world-readable directory and change [modelDir].)
 *
 * Driving it without the screen: `adb shell am start -n com.voxrt.linnet.sample/.MainActivity --es autospeak
 * "<text>" [--ez prewarm true] [--ei repeat N]` speaks on start, logs the timings (`adb logcat -s linnet-sample`)
 * and exits. `prewarm` opens the output route before the timer starts; the log line shows when the first sample
 * reached the output (onset) and when the first chunk had fully played.
 */
class MainActivity : AppCompatActivity() {
    private val executor = Executors.newSingleThreadExecutor()
    private val cancel = AtomicBoolean(false)

    private var model: LinnetModel? = null
    private var voice: Voice? = null

    private lateinit var textField: EditText
    private lateinit var speak: Button
    private lateinit var stop: Button
    private lateinit var status: TextView

    private val modelDir: File get() = filesDir
    private val weightsFile get() = File(modelDir, "linnet-en-35m.fp16.safetensors")
    private val tokenizerFile get() = File(modelDir, "tokenizer.model")
    private val voiceFile get() = File(modelDir, "voice.npy")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        textField = findViewById(R.id.text)
        speak = findViewById(R.id.speak)
        stop = findViewById(R.id.stop)
        status = findViewById(R.id.status)

        val present = listOf(weightsFile, tokenizerFile, voiceFile).filter { it.exists() }.map { it.name }
        status.text = getString(R.string.status_idle, if (present.size == 3) "found in ${modelDir.path}" else "missing (${3 - present.size} of 3) in ${modelDir.path}")

        speak.setOnClickListener { speak(textField.text.toString()) }
        stop.setOnClickListener { cancel.set(true) }

        intent.getStringExtra("autospeak")?.let { text ->
            val prewarm = intent.getBooleanExtra("prewarm", false)
            val repeat = intent.getIntExtra("repeat", 1).coerceAtLeast(1)
            textField.setText(text)
            executor.execute {
                try {
                    repeat(repeat) { runSynthesis(text, prewarm) }
                    Log.i(TAG, "autospeak done")
                } catch (e: Exception) {
                    Log.e(TAG, "autospeak failed: ${e.message}")
                }
                runOnUiThread { finish() }
            }
        }
    }

    private fun speak(text: String) {
        if (text.isBlank()) return
        speak.isEnabled = false
        stop.isEnabled = true
        cancel.set(false)
        executor.execute {
            try {
                runSynthesis(text)
            } catch (e: Exception) {
                post("error: ${e.message}")
            } finally {
                runOnUiThread {
                    speak.isEnabled = true
                    stop.isEnabled = false
                }
            }
        }
    }

    private fun ensureModel(): Pair<LinnetModel, Voice> {
        model?.let { m -> voice?.let { v -> return m to v } }
        post("loading model (runtime ${LinnetModel.version})...")
        val t0 = SystemClock.elapsedRealtime()
        val m = LinnetModel(weightsFile, tokenizerFile)
        val tWeights = SystemClock.elapsedRealtime() - t0
        val v = m.loadVoice(voiceFile) // also prepares the voice: the prime pass then runs only the text
        val tVoice = SystemClock.elapsedRealtime() - t0 - tWeights
        val loaded = "model loaded in ${tWeights + tVoice} ms (weights $tWeights ms, voice $tVoice ms)"
        Log.i(TAG, loaded)
        post(loaded)
        model = m
        voice = v
        return m to v
    }

    private fun runSynthesis(text: String, prewarm: Boolean = false) {
        val (m, v) = ensureModel()
        m.session(v, SynthParams(temp = 0.3f, cfg = 2.0f, seed = 0)).use { session ->
            AudioTrackPlayer(m.sampleRate).use { player ->
                if (prewarm) player.prewarm()
                val t0 = SystemClock.elapsedRealtime()
                val t0Nanos = System.nanoTime()
                session.prime(text)
                val tPrime = SystemClock.elapsedRealtime() - t0
                var firstChunkMs = -1L
                var worstStepMs = 0L
                var synthMs = 0L
                var samples = 0L
                val buf = FloatArray(session.samplesPerStep)
                while (!cancel.get()) {
                    val tStep = SystemClock.elapsedRealtime()
                    val n = session.stepInto(buf)
                    if (n < 0) break
                    val dt = SystemClock.elapsedRealtime() - tStep
                    if (firstChunkMs < 0) firstChunkMs = SystemClock.elapsedRealtime() - t0
                    if (dt > worstStepMs) worstStepMs = dt
                    synthMs += dt
                    samples += n
                    player.write(buf, n)
                    if (session.frames % 5 == 0) {
                        post("speaking: ${session.frames} steps, ${samples * 1000 / m.sampleRate} ms of audio")
                    }
                }
                if (cancel.get()) {
                    player.stopNow()
                    post("stopped after ${session.frames} steps")
                } else {
                    val timing = player.timing
                    player.drain()
                    val audioMs = samples * 1000 / m.sampleRate
                    val ms = { n: Long? -> if (n == null) "n/a" else "${(n - t0Nanos) / 1_000_000}" }
                    val playback = if (timing == null) "" else
                        "\nplayback (route ${if (timing.routeWasOpen) "prewarmed" else "opened by the first chunk"}): first chunk " +
                            "${timing.firstChunkFrames} samples written at ${ms(timing.firstWriteNanos)} ms, onset ${ms(timing.onsetNanos)} ms, " +
                            "first chunk fully played at ${ms(timing.firstChunkPlayedNanos)} ms (AudioTrack timestamp); " +
                            "playback head passed the first sample at ${ms(timing.headPassedFirstFrameNanos)} ms and the chunk's end at " +
                            "${ms(timing.headPassedFirstChunkNanos)} ms (mixer side, output latency not included)"
                    val line = "done: ${session.frames} steps, ${audioMs} ms audio\n" +
                        "prime ${tPrime} ms, first chunk ${firstChunkMs} ms, worst step ${worstStepMs} ms (80 ms budget)\n" +
                        "step time ${synthMs} ms for ${audioMs} ms of audio: RTF ${"%.1f".format(100.0 * synthMs / audioMs)} % (playback pacing excluded)" +
                        playback
                    Log.i(TAG, line)
                    post(line)
                }
            }
        }
    }

    private fun post(msg: String) = runOnUiThread { status.text = msg }

    private companion object {
        const val TAG = "linnet-sample"
    }

    override fun onDestroy() {
        super.onDestroy()
        cancel.set(true)
        executor.shutdown()
        voice?.close()
        model?.close()
    }
}
