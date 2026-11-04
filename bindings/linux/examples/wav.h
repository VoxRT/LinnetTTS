/* Minimal 16-bit PCM WAV writer shared by the examples. */
#ifndef LINNET_EXAMPLE_WAV_H
#define LINNET_EXAMPLE_WAV_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void put_u16(FILE *f, uint16_t v) { uint8_t b[2] = {v & 0xff, v >> 8}; fwrite(b, 1, 2, f); }
static void put_u32(FILE *f, uint32_t v) {
  uint8_t b[4] = {v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, v >> 24};
  fwrite(b, 1, 4, f);
}

/* Open a WAV file and write a header for `frames` mono 16-bit samples at `rate`. */
static FILE *wav_open(const char *path, uint32_t rate, uint32_t frames) {
  FILE *f = fopen(path, "wb");
  if (!f) return NULL;
  uint32_t data_len = frames * 2;
  fwrite("RIFF", 1, 4, f);
  put_u32(f, 36 + data_len);
  fwrite("WAVEfmt ", 1, 8, f);
  put_u32(f, 16);
  put_u16(f, 1); /* PCM */
  put_u16(f, 1); /* mono */
  put_u32(f, rate);
  put_u32(f, rate * 2);
  put_u16(f, 2);
  put_u16(f, 16);
  fwrite("data", 1, 4, f);
  put_u32(f, data_len);
  return f;
}

/* Append float samples as 16-bit PCM (clamp to [-1, 1], scale by 32767, round half away from zero: the same
 * conversion as the runtime's `synth` example, so the WAV matches the reference MD5 of the weights release). */
static void wav_write(FILE *f, const float *pcm, size_t n) {
  int16_t *buf = (int16_t *)malloc(n * sizeof(int16_t));
  for (size_t i = 0; i < n; i++) {
    float s = pcm[i] < -1.f ? -1.f : pcm[i] > 1.f ? 1.f : pcm[i];
    buf[i] = (int16_t)roundf(s * 32767.f);
  }
  fwrite(buf, sizeof(int16_t), n, f);
  free(buf);
}

/* Patch the sizes in the header once the total is known (streaming use), then close. */
static void wav_close(FILE *f, uint32_t frames) {
  uint32_t data_len = frames * 2;
  fseek(f, 4, SEEK_SET);
  put_u32(f, 36 + data_len);
  fseek(f, 40, SEEK_SET);
  put_u32(f, data_len);
  fclose(f);
}

#endif
