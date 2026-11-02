package com.voxrt.linnet

/**
 * Sampling parameters of one utterance (same defaults as the Python and C APIs).
 *
 * @property temp variance of the flow start noise; 0 is deterministic.
 * @property cfg classifier-free guidance weight; 1 disables the unconditional stream (faster).
 * @property maxFrames upper bound on LM steps (80 ms each); 300 = 24 s.
 * @property eosThreshold stop when the end-of-speech logit exceeds this value.
 * @property seed seed of the noise generator; the same seed gives the same audio.
 */
data class SynthParams(
    val temp: Float = 0.3f,
    val cfg: Float = 2.0f,
    val maxFrames: Int = 300,
    val eosThreshold: Float = -1.0f,
    val seed: Long = 0L,
) {
    init {
        require(maxFrames >= 1) { "maxFrames must be >= 1" }
        require(temp >= 0f && temp.isFinite()) { "temp must be finite and >= 0" }
        require(cfg.isFinite()) { "cfg must be finite" }
    }
}
