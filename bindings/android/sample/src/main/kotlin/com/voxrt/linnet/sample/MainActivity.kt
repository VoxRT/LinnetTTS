package com.voxrt.linnet.sample

import android.os.Bundle
import android.os.SystemClock
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
 * (`voice.npy` is any `[T, 48]` voice file, e.g. voice_hifitts2_val0.npy renamed. `run-as` works
 * for debug builds; for a release build push to a world-readable directory and change [modelDir].)
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
        val v = m.loadVoice(voiceFile)
        post("model loaded in ${SystemClock.elapsedRealtime() - t0} ms")
        model = m
        voice = v
        return m to v
    }

    private fun runSynthesis(text: String) {
        val (m, v) = ensureModel()
        m.session(v, SynthParams(temp = 0.3f, cfg = 2.0f, seed = 0)).use { session ->
            AudioTrackPlayer(m.sampleRate).use { player ->
                val t0 = SystemClock.elapsedRealtime()
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
                    player.drain()
                    val audioMs = samples * 1000 / m.sampleRate
                    post(
                        "done: ${session.frames} steps, ${audioMs} ms audio\n" +
                            "prime ${tPrime} ms, first chunk ${firstChunkMs} ms, worst step ${worstStepMs} ms (80 ms budget)\n" +
                            "step time ${synthMs} ms for ${audioMs} ms of audio: RTF ${"%.1f".format(100.0 * synthMs / audioMs)} % (playback pacing excluded)",
                    )
                }
            }
        }
    }

    private fun post(msg: String) = runOnUiThread { status.text = msg }

    override fun onDestroy() {
        super.onDestroy()
        cancel.set(true)
        executor.shutdown()
        voice?.close()
        model?.close()
    }
}
