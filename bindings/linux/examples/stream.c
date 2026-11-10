/* Streaming synthesis: prepare the voice, prime a session and write each 80 ms chunk to the
 * WAV as it arrives. Reports the voice preparation, time to first audio and per-step latency.
 *
 *   ./stream <weights.safetensors> <tokenizer.model> <voice.npy> "text" out.wav [seed]
 */
#define _POSIX_C_SOURCE 199309L /* clock_gettime */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "linnet_tts.h"
#include "wav.h"

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int fail(const char *what, int rc) {
  fprintf(stderr, "%s failed (%d): %s\n", what, rc, linnet_last_error());
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: stream <weights> <tokenizer.model> <voice.npy> \"text\" out.wav [seed]\n");
    return 2;
  }
  linnet_model *model = NULL;
  linnet_voice *voice = NULL;
  linnet_session *session = NULL;
  int rc;

  if ((rc = linnet_model_load(argv[1], argv[2], &model)) != LINNET_OK) return fail("model load", rc);
  if ((rc = linnet_voice_load(argv[3], &voice)) != LINNET_OK) return fail("voice load", rc);
  /* Once per voice: sessions then skip the voice part of the prime pass. */
  double tv = now_ms();
  if ((rc = linnet_voice_prepare(model, voice)) != LINNET_OK) return fail("voice prepare", rc);
  double prepare_ms = now_ms() - tv;

  linnet_params params = {0.3f, 2.0f, -1.0f, 300, 0};
  if (argc > 6) params.seed = strtoull(argv[6], NULL, 10);
  if ((rc = linnet_session_new(model, voice, &params, &session)) != LINNET_OK) return fail("session", rc);

  uint32_t rate = linnet_sample_rate(model);
  size_t per_step = linnet_samples_per_step(model);
  float *chunk = malloc(per_step * sizeof(float));

  FILE *f = wav_open(argv[5], rate, 0);
  if (!f) { perror(argv[5]); return 1; }

  double t0 = now_ms();
  if ((rc = linnet_session_prime(session, argv[4])) != LINNET_OK) return fail("prime", rc);
  double prime_ms = now_ms() - t0;

  size_t total = 0, steps = 0, written;
  double first_audio_ms = 0, max_step_ms = 0;
  for (;;) {
    double ts = now_ms();
    if ((rc = linnet_session_step(session, chunk, per_step, &written)) != LINNET_OK) return fail("step", rc);
    double ms = now_ms() - ts;
    if (written == 0) break; /* finished */
    if (total == 0) first_audio_ms = prime_ms + ms;
    if (written == per_step) {
      steps++;
      if (ms > max_step_ms) max_step_ms = ms;
    }
    wav_write(f, chunk, written);
    total += written;
  }
  double total_ms = now_ms() - t0;
  wav_close(f, (uint32_t)total);

  double secs = (double)total / rate;
  printf("linnet-tts %s: voice prepared in %.1f ms (once); prime %.1f ms, first audio %.1f ms, %zu steps, "
         "%.3f s audio in %.0f ms (RTF %.1f %%, worst step %.1f ms of %.0f ms budget) -> %s\n",
         linnet_version(), prepare_ms, prime_ms, first_audio_ms, total / per_step, secs, total_ms,
         100.0 * total_ms / 1e3 / secs, max_step_ms, 1e3 * per_step / rate, argv[5]);

  free(chunk);
  linnet_session_free(session);
  linnet_voice_free(voice);
  linnet_model_free(model);
  return 0;
}
