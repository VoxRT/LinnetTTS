/* Offline synthesis: load model + voice, synthesize one sentence, write a 16-bit WAV.
 *
 *   ./synth <weights.safetensors> <tokenizer.model> <voice.npy> "text" out.wav [seed]
 */
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
    fprintf(stderr, "usage: synth <weights> <tokenizer.model> <voice.npy> \"text\" out.wav [seed]\n");
    return 2;
  }
  linnet_model *model = NULL;
  linnet_voice *voice = NULL;
  int rc;

  double t0 = now_ms();
  if ((rc = linnet_model_load(argv[1], argv[2], &model)) != LINNET_OK) return fail("model load", rc);
  if ((rc = linnet_voice_load(argv[3], &voice)) != LINNET_OK) return fail("voice load", rc);
  double load_ms = now_ms() - t0;

  linnet_params params = {0.3f, 2.0f, -1.0f, 300, 0};
  if (argc > 6) params.seed = strtoull(argv[6], NULL, 10);

  uint32_t rate = linnet_sample_rate(model);
  size_t per_step = linnet_samples_per_step(model);
  size_t cap = params.max_frames * per_step;
  float *pcm = malloc(cap * sizeof(float));
  size_t n = 0;

  double t1 = now_ms();
  rc = linnet_synthesize(model, voice, argv[4], &params, pcm, cap, &n);
  double synth_ms = now_ms() - t1;
  if (rc != LINNET_OK) return fail("synthesize", rc);

  FILE *f = wav_open(argv[5], rate, (uint32_t)n);
  if (!f) { perror(argv[5]); return 1; }
  wav_write(f, pcm, n);
  wav_close(f, (uint32_t)n);

  double secs = (double)n / rate;
  printf("linnet-tts %s: load %.0f ms, %zu steps, %.3f s audio in %.0f ms (RTF %.1f %%) -> %s\n",
         linnet_version(), load_ms, n / per_step, secs, synth_ms, 100.0 * synth_ms / 1e3 / secs, argv[5]);

  free(pcm);
  linnet_voice_free(voice);
  linnet_model_free(model);
  return 0;
}
