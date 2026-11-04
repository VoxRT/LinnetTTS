# Example audio

Four utterances synthesized with the released `linnet-en-35m` weights (fp16 file, Rust runtime, default parameters:
`temp` 0.3, `cfg` 2.0, `seed` 0, one thread), 24 kHz 16-bit WAV. They are reproducible with the `synth` example:

```sh
cd runtime && cargo run --release --example synth -- ../local/linnet-en-35m.fp16.safetensors ../local/tokenizer.model \
    ../local/voices/linnet-en-f1.npy "<text>" out.wav
```

| file | voice | text |
|---|---|---|
| [`linnet-en-f1_plain.wav`](linnet-en-f1_plain.wav) | `linnet-en-f1` | Good morning, and welcome back. The meeting starts at nine, and the coffee is already on the table. |
| [`linnet-en-f1_numbers.wav`](linnet-en-f1_numbers.wav) | `linnet-en-f1` | There are 24 students in room 12, and the class starts at 9:30. |
| [`linnet-en-m1_plain.wav`](linnet-en-m1_plain.wav) | `linnet-en-m1` | Good morning, and welcome back. The meeting starts at nine, and the coffee is already on the table. |
| [`linnet-en-m1_numbers.wav`](linnet-en-m1_numbers.wav) | `linnet-en-m1` | There are 24 students in room 12, and the class starts at 9:30. |

The note on raw digits, dates and similar text is in the root README ("Examples").

The files are model Outputs in the sense of `WEIGHTS_LICENSE.md`; see `tests/goldens/NOTICE.md` for how generated audio in
this repository may be used.
