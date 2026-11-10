package com.voxrt.linnet

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Instrumented smoke test against the real model files. They are not bundled; push them first:
 *
 *   adb shell mkdir -p /data/local/tmp/linnet
 *   adb push linnet-en-35m.fp16.safetensors tokenizer.model voice.npy /data/local/tmp/linnet/
 *
 * (`voice.npy` = any `[T, 48]` voice file, e.g. voices/linnet-en-f1.npy renamed.) Without the
 * files the tests are skipped, not failed.
 */
@RunWith(AndroidJUnit4::class)
class SmokeTest {
    private val dir = File("/data/local/tmp/linnet")
    private val weights = File(dir, "linnet-en-35m.fp16.safetensors")
    private val tokenizer = File(dir, "tokenizer.model")
    private val voiceFile = File(dir, "voice.npy")

    private fun assumeAssets() {
        assumeTrue("model assets not found under $dir", weights.exists() && tokenizer.exists() && voiceFile.exists())
    }

    @Test
    fun nativeLibraryLoadsAndReportsVersion() {
        InstrumentationRegistry.getInstrumentation().targetContext
        assertTrue(LinnetModel.version.isNotEmpty())
    }

    @Test
    fun constantsAndVoiceRoundTrip() {
        assumeAssets()
        LinnetModel(weights, tokenizer).use { model ->
            assertEquals(24000, model.sampleRate)
            assertEquals(1920, model.samplesPerStep)
            assertEquals(720, model.outputDelay)
            assertEquals(48, model.latentDim)
            assertTrue(model.tokenize("Hello there.").isNotEmpty())
            model.loadVoice(voiceFile).use { voice ->
                val lat = voice.latents()
                assertEquals(voice.frames * 48, lat.size)
                Voice.fromLatents(lat, voice.frames).use { v2 ->
                    assertEquals(voice.frames, v2.frames)
                    val a = model.synthesize("Hello there.", voice, SynthParams(seed = 3))
                    val b = model.synthesize("Hello there.", v2, SynthParams(seed = 3))
                    assertArrayEquals(a, b, 0f)
                }
            }
        }
    }

    @Test
    fun streamConcatEqualsSynthesize() {
        assumeAssets()
        val text = "The quick brown fox jumps over the lazy dog."
        val params = SynthParams(seed = 7)
        LinnetModel(weights, tokenizer).use { model ->
            model.loadVoice(voiceFile).use { voice ->
                val offline = model.synthesize(text, voice, params)
                model.session(voice, params).use { s ->
                    s.prime(text)
                    val chunks = s.chunks().toList()
                    assertTrue(s.finished)
                    assertNull(s.step())
                    assertEquals(s.frames + 1, chunks.size)
                    assertEquals(model.samplesPerStep - model.outputDelay, chunks.first().size)
                    assertEquals(model.outputDelay, chunks.last().size)
                    assertTrue(chunks.last().all { it == 0f })
                    val streamed = FloatArray(chunks.sumOf { it.size })
                    var off = 0
                    for (c in chunks) { c.copyInto(streamed, off); off += c.size }
                    assertEquals(offline.size, streamed.size)
                    assertEquals(s.frames * model.samplesPerStep, offline.size)
                    assertArrayEquals(offline, streamed, 0f)
                }
            }
        }
    }

    /** `loadVoice` prepares the voice, `Voice.load` does not; both paths and a re-primed session give the same bytes. */
    @Test
    fun preparedVoiceIsBitExact() {
        assumeAssets()
        val text = "The meeting starts at nine."
        val params = SynthParams(seed = 4)
        LinnetModel(weights, tokenizer).use { model ->
            model.loadVoice(voiceFile).use { voice ->
                assertTrue(model.isPrepared(voice))
                Voice.load(voiceFile).use { raw ->
                    assertTrue(!model.isPrepared(raw))
                    val plain = model.synthesize(text, raw, params)
                    assertArrayEquals(plain, model.synthesize(text, voice, params), 0f)
                    model.session(voice, params).use { s ->
                        s.prime("Hello there.")
                        s.chunks().count()
                        s.prime(text)
                        val chunks = s.chunks().toList()
                        val streamed = FloatArray(chunks.sumOf { it.size })
                        var off = 0
                        for (c in chunks) { c.copyInto(streamed, off); off += c.size }
                        assertArrayEquals(plain, streamed, 0f)
                    }
                    model.prepare(raw)
                    assertTrue(model.isPrepared(raw))
                    assertArrayEquals(plain, model.synthesize(text, raw, params), 0f)
                    LinnetModel(weights, tokenizer).use { other ->
                        assertTrue(!other.isPrepared(voice))
                        assertArrayEquals(plain, other.synthesize(text, voice, params), 0f)
                    }
                }
            }
        }
    }

    @Test
    fun sessionOutlivesModelAndErrorsAreTyped() {
        assumeAssets()
        val model = LinnetModel(weights, tokenizer)
        val voice = model.loadVoice(voiceFile)
        val s = model.session(voice, SynthParams(maxFrames = 3))
        voice.close()
        model.close()
        s.prime("Hello.")
        assertTrue(s.chunks().count() >= 2)
        s.close()
        try {
            LinnetModel(File(dir, "missing.safetensors"), tokenizer)
            throw AssertionError("expected LinnetException")
        } catch (e: LinnetException) {
            assertEquals(LinnetException.IO, e.code)
        }
    }
}
