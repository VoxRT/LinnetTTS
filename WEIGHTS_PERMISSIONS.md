# Written permissions under the Linnet Weights License

Section 15.2 of the Linnet Weights License, version 1.0 (`WEIGHTS_LICENSE.md`), makes written permissions given by the
Licensor part of the license. The permissions below are given by Wiseport Ltd. (trading as VoxRT), the Licensor, to
everyone exercising rights under version 1.0 of the license ("You"), for every weights release designated under that
version. They add to the license and take nothing away from it. Capitalised terms have the meaning the license gives
them. The authoritative copy of this file is
https://github.com/VoxRT/LinnetTTS/blob/main/WEIGHTS_PERMISSIONS.md; a later change to it applies only to weights
releases published after the change, and a permission already given is not withdrawn for releases designated under
version 1.0.

## 1. Stored prepared voice state (10 November 2026)

**What it is.** The LinnetTTS runtime can run a Voice File through the Weights once and keep the result, so that later
utterances skip that part of the prime pass: `Linnet::voice_state` in Rust, `linnet_voice_prepare` in C,
`Model.prepare_voice` in Python, `LinnetModel.prepare` in Kotlin and Swift. The *prepared voice state* is what the
runtime's `VoiceState` holds for a Voice File: the attention keys and values of the voice prefix, the two-row prefix
of the unconditional stream and its output (about 2 MB per voice). The runtime keeps it in memory and offers no way
to write it out; the code is Apache-2.0, so a Product may add one.

**How the Licensor reads the license.** The prepared voice state encodes the Voice File and allows it to be
recovered, so the Licensor treats it as a transformed form of the Voice File under Section 3 of the license, not as
an Output under Section 1. Section 3 permits such a form in memory only. This permission is given on that reading.

**Permission.** In addition to Section 3, You may store the prepared voice state of a Voice File that You are
licensed to use with the Weights (a built-in voice, or a voice the Licensor has authorised in writing under
Section 1) and reuse it, provided that:

1. the state is used only within a Product of Yours, with the same Weights that produced it;
2. the state is stored only where that Product runs or is installed: on the devices and servers that run it, in the
   Product's installation package, images and backups under Section 5.1, and in the incidental operating-system
   backup and synchronisation of the Product's own storage; it is removed with the Product;
3. the state is not distributed, exported or made available outside the Product (Section 5(d) applies to it as it
   applies to the Licensed Assets), and the Product provides no feature whose purpose is to export it;
4. for every other purpose of the license the state is treated as the Voice File it was derived from: in
   particular Sections 4.1, 4.3 and 4.5 (Custom Voice Creation, Model Training, reproduction of a voice identity)
   apply to it, and it grants no right in the Voice File beyond those You already hold.

Section 4.4 is unaffected: a General-Purpose TTS Service is not a Product, so nothing here lets one store such
states. Section 8 is unaffected: a Product that stores a prepared voice state shows the notices as before.

**Not covered.** The Weights and the Tokenizer themselves. A stored quantised, converted or compiled copy of them is a
Modification reserved under Section 4.2; Section 3 does not permit it, and neither does this permission.
