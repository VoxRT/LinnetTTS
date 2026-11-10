# Written permissions under the Linnet Weights License

Section 15.2 of the Linnet Weights License 1.0 (`WEIGHTS_LICENSE.md`) makes written permissions given by the Licensor part
of the license. The permissions below are given by Wiseport Ltd. (trading as VoxRT) to every licensee of the Linnet weights,
for every weights release designated under version 1.0 of the license. They add to the license; they take nothing away.

## 1. Stored prepared voice state (10 November 2026)

The LinnetTTS runtime can run a Voice File through the Weights once and keep the result, the attention keys and values of
the voice prefix (about 2 MB per voice: `Linnet::voice_state` in Rust, `linnet_voice_prepare` in C, `prepare_voice` in
Python, `prepare` in Kotlin and Swift), so that later utterances skip that part of the prime pass. This *prepared voice
state* is a transformed form of the Voice File under Section 3 of the license, which permits it in memory only.

In addition to Section 3, a Product may store the prepared voice state of a Voice File that it is licensed to use, on the
device or server where the Product runs, and reuse it across runs of that Product, provided that:

1. the state is used only by that Product, with the same Weights that produced it;
2. the state is not distributed, exported or made available outside the Product (Section 5(d) applies to it as it applies
   to the Licensed Assets), and is removed with the Product;
3. for every other purpose of the license the state is treated as the Voice File it was derived from: in particular
   Sections 4.1, 4.3 and 4.5 (Custom Voice Creation, Model Training, reproduction of a voice identity) apply to it, and it
   grants no right to the Voice File beyond those the Product already holds.

This permission does not cover the Weights or the tokenizer themselves: a stored quantised, converted or compiled copy of
the Weights remains outside Section 3.
