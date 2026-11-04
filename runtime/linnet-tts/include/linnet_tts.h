/* LinnetTTS C API. See runtime/linnet-tts/src/c_api.rs for the contract.
 *
 * Every fallible function returns a status (LINNET_OK = 0, negative on error).
 * Handles come from *_load / *_new and go back through *_free; the
 * library never returns memory the caller has to free. A session keeps a
 * pointer to its model: free sessions before the model. Handles are not
 * thread-safe individually; one model may serve many sessions on
 * different threads.
 */
#ifndef LINNET_TTS_H
#define LINNET_TTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LINNET_OK 0
#define LINNET_ERR_INVALID_ARG (-1)
#define LINNET_ERR_IO (-2)
#define LINNET_ERR_FORMAT (-3)
#define LINNET_ERR_SHAPE (-4)
#define LINNET_ERR_BUFFER_TOO_SMALL (-5)
#define LINNET_ERR_INTERNAL (-6)

typedef struct linnet_model linnet_model;
typedef struct linnet_voice linnet_voice;
typedef struct linnet_session linnet_session;

/* Sampling parameters. Pass NULL for the defaults:
 * temp 0.3, cfg 2.0, eos_threshold -1.0, max_frames 300, seed 0. */
typedef struct linnet_params {
  float temp;          /* variance of the flow start noise; 0 = deterministic */
  float cfg;           /* classifier-free guidance weight; 1 = off */
  float eos_threshold; /* stop when the end-of-speech logit exceeds this */
  uint32_t max_frames; /* upper bound on LM steps (80 ms each), at least 1 */
  uint64_t seed;       /* noise generator seed */
} linnet_params;

/* Static version string; do not free. */
const char *linnet_version(void);
/* Message of the most recent error on the calling thread; do not free. */
const char *linnet_last_error(void);

/* Model: safetensors weights + SentencePiece .model tokenizer. */
int linnet_model_load(const char *weights_path, const char *tokenizer_path, linnet_model **out_model);
void linnet_model_free(linnet_model *model);
uint32_t linnet_sample_rate(const linnet_model *model);
/* Largest chunk linnet_session_step can write (samples per LM step; 1920 for linnet-en-35m). */
size_t linnet_samples_per_step(const linnet_model *model);
/* Threads the model uses: 1 = single-threaded, 0 = automatic (the fast cores,
 * at most 4 on Linux/Android, 1 on iOS/macOS); explicit values are capped at 32.
 * The default is automatic. The audio is identical for every value; the
 * setting applies to existing and future sessions. */
int linnet_set_threads(linnet_model *model, int threads);
int linnet_threads(const linnet_model *model);

/* Voice: [T, 48] float32 latents, .npy file or in-memory buffer. The width is
 * validated against the model by linnet_session_new / linnet_synthesize
 * (LINNET_ERR_INVALID_ARG), not here. */
int linnet_voice_load(const char *path, linnet_voice **out_voice);
int linnet_voice_from_latents(const float *latents, size_t frames, size_t dim, linnet_voice **out_voice);
void linnet_voice_free(linnet_voice *voice);

/* Session: streaming synthesis. */
int linnet_session_new(const linnet_model *model, const linnet_voice *voice, const linnet_params *params,
                       linnet_session **out_session);
void linnet_session_free(linnet_session *session);
/* Tokenize text (UTF-8) and run the prefix; may be called again for a new text.
 * Text that produces no tokens (empty or whitespace) is LINNET_ERR_INVALID_ARG. */
int linnet_session_prime(linnet_session *session, const char *text);
/* Next chunk of 24 kHz float32 audio; cap must be >= linnet_samples_per_step
 * (checked before the step runs). *out_written == 0 means finished, or not
 * primed yet. The first chunk is samples_per_step - output delay long (1200
 * for linnet-en-35m), later ones samples_per_step, the final one the output
 * delay (720) zeros; an utterance that ends on its first step yields no chunk
 * at all (see docs/streaming-contract.md). */
int linnet_session_step(linnet_session *session, float *out_pcm, size_t cap, size_t *out_written);

/* Whole utterance in one call. *out_written always receives the full length;
 * if it exceeds cap the status is LINNET_ERR_BUFFER_TOO_SMALL and nothing is
 * written, so out_pcm = NULL, cap = 0 queries the size. */
int linnet_synthesize(const linnet_model *model, const linnet_voice *voice, const char *text,
                      const linnet_params *params, float *out_pcm, size_t cap, size_t *out_written);

#ifdef __cplusplus
}
#endif

#endif /* LINNET_TTS_H */
