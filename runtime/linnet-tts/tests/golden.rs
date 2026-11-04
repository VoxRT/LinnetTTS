//! Parity against the Python package: the goldens in `tests/goldens/`
//! store the token ids, the Gaussian draws, the latents and the audio of
//! two short utterances. The fp32 weights must reproduce the latents to
//! 1e-4, the decoder alone to 1e-4 and the end-to-end audio to 1e-2 (SNR
//! >= 40 dB); the fp16 export is allowed to drift (reported, bounded loosely).
//!
//! Inputs are looked up under `local/` of the repository (the unpacked
//! weights release: `linnet-en-35m.safetensors`, `linnet-en-35m.fp16.safetensors`,
//! `tokenizer.model`, `voices/linnet-en-f1.npy`) and may be overridden with
//! `LINNET_WEIGHTS`, `LINNET_WEIGHTS_FP16`, `LINNET_TOKENIZER`, `LINNET_VOICE`.
//! A missing file skips the test instead of failing it and prints a
//! `SKIPPED: ...` line on stderr; the test harness hides that output for
//! passing tests, so run `cargo test -- --nocapture` to see whether the
//! goldens were actually checked.

use std::path::{Path, PathBuf};

use linnet_tts::{Linnet, Session, SynthParams, Tokenizer, VoiceLatents};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn input(env: &str, default_rel: &str) -> Option<PathBuf> {
    let p = std::env::var(env).map(PathBuf::from).unwrap_or_else(|_| repo_root().join(default_rel));
    if p.exists() {
        Some(p)
    } else {
        // Visible only with `cargo test -- --nocapture`: the harness captures
        // stderr of passing tests.
        eprintln!("SKIPPED: {} not found (set {env}); run with -- --nocapture to see this", p.display());
        None
    }
}

fn weights() -> Option<PathBuf> {
    input("LINNET_WEIGHTS", "local/linnet-en-35m.safetensors")
}
fn weights_fp16() -> Option<PathBuf> {
    input("LINNET_WEIGHTS_FP16", "local/linnet-en-35m.fp16.safetensors")
}
fn tokenizer() -> Option<PathBuf> {
    input("LINNET_TOKENIZER", "local/tokenizer.model")
}
fn voice() -> Option<PathBuf> {
    input("LINNET_VOICE", "local/voices/linnet-en-f1.npy")
}

struct Golden {
    name: String,
    text: String,
    tokens: Vec<u32>,
    params: SynthParams,
    n_steps: usize,
    eos_fired: bool,
    noise: Vec<f32>,
    latents: Vec<f32>,
    wav: Vec<f32>,
}

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn goldens() -> Vec<Golden> {
    let dir = repo_root().join("tests/goldens");
    let mut out = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).collect();
    entries.sort();
    for d in entries {
        let meta_path = d.join("meta.json");
        if !meta_path.exists() {
            continue;
        }
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
        let n_steps = meta["n_steps"].as_u64().unwrap() as usize;
        let ld = meta["latent_dim"].as_u64().unwrap() as usize;
        let noise = read_f32(&d.join("noise.bin"));
        let latents = read_f32(&d.join("latents.bin"));
        let wav = read_f32(&d.join("wav.bin"));
        assert_eq!(noise.len(), meta["n_noise"].as_u64().unwrap() as usize * ld);
        assert_eq!(latents.len(), n_steps * ld);
        assert_eq!(wav.len(), meta["wav_len"].as_u64().unwrap() as usize);
        out.push(Golden {
            name: d.file_name().unwrap().to_string_lossy().into_owned(),
            text: meta["text"].as_str().unwrap().to_string(),
            tokens: meta["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect(),
            params: SynthParams {
                temp: meta["temp"].as_f64().unwrap() as f32,
                cfg: meta["cfg"].as_f64().unwrap() as f32,
                max_frames: meta["max_frames"].as_u64().unwrap() as usize,
                eos_threshold: meta["eos_threshold"].as_f64().unwrap() as f32,
                seed: meta["seed"].as_u64().unwrap(),
                threads: 1,
            },
            n_steps,
            eos_fired: meta["eos_fired"].as_bool().unwrap(),
            noise,
            latents,
            wav,
        });
    }
    assert!(!out.is_empty(), "no goldens under {}", dir.display());
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// Signal-to-noise ratio of `test` against `reference`, in dB.
fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    let sig: f64 = reference.iter().map(|x| (*x as f64).powi(2)).sum();
    let err: f64 = reference.iter().zip(test).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum();
    10.0 * (sig / err.max(1e-30)).log10()
}

struct Run {
    latents: Vec<f32>,
    wav: Vec<f32>,
    n_steps: usize,
    noise_used: usize,
}

/// Drive a session with the golden noise instead of the RNG.
fn run_with_noise(model: &Linnet, voice: &VoiceLatents, g: &Golden) -> Run {
    run_with_noise_params(model, voice, g, g.params)
}

fn run_with_noise_params(model: &Linnet, voice: &VoiceLatents, g: &Golden, params: SynthParams) -> Run {
    let ld = model.latent_dim();
    let mut s = Session::new(model, voice, params).unwrap();
    s.prime(&g.text).unwrap();
    let mut latents = Vec::new();
    let mut wav = Vec::new();
    let mut i = 0usize;
    let mut n_steps = 0usize;
    loop {
        let before = s.frames();
        let noise = if i < g.noise.len() / ld { &g.noise[i * ld..(i + 1) * ld] } else { &g.noise[..ld] };
        let chunk = match s.step_with_noise(noise) {
            Some(c) => c.to_vec(),
            None => break,
        };
        if s.frames() > before {
            // a real step consumed one noise vector and produced a latent
            i += 1;
            n_steps += 1;
            latents.extend_from_slice(s.latent());
        } else if !s.is_finished() {
            // end detected: the EOS step also consumed one noise vector
            i += 1;
        }
        wav.extend_from_slice(&chunk);
    }
    if s.frames() == n_steps && i == n_steps && g.eos_fired {
        // EOS detected on a step that produced no audio tail (frames == 0)
        i += 1;
    }
    Run { latents, wav, n_steps, noise_used: i }
}

#[test]
fn tokenizer_matches_sentencepiece() {
    let Some(tok) = tokenizer() else { return };
    let tok = Tokenizer::load(tok).unwrap();
    assert_eq!(tok.vocab_size(), 6000);
    // Reference ids produced by the sentencepiece library on the same model file.
    let cases: &[(&str, &[u32])] = &[
        ("Hello there.", &[345, 395, 440, 350, 282]),
        ("The meeting is at nine.", &[311, 1675, 298, 313, 1643, 282]),
        ("  double  space", &[286, 286, 2809, 286, 2012]),
        ("naïve café 日本", &[1724, 218, 198, 443, 1825, 543, 5989, 286, 253, 174, 188, 253, 179, 195]),
        ("ab.c", &[287, 448, 282, 521]),
        ("Call 911 now", &[897, 395, 286, 476, 414, 414, 391]),
        ("<sep> hi <spk>", &[286, 5994, 281, 1873, 5997, 2013, 286, 5994, 895, 577, 5997]),
        ("tab\there", &[286, 302, 1886, 32, 700, 376]),
        ("x\ny", &[286, 679, 33, 322]),
        ("émigré ȝ", &[286, 5989, 1030, 976, 5989, 286, 223, 180]),
        ("", &[]),
    ];
    for (text, want) in cases {
        assert_eq!(tok.encode(text), *want, "text {text:?}");
    }
    for g in goldens() {
        assert_eq!(tok.encode(&g.text), g.tokens, "golden {}", g.name);
    }
}

#[test]
fn fp32_matches_goldens() {
    let (Some(w), Some(t), Some(v)) = (weights(), tokenizer(), voice()) else { return };
    let model = Linnet::load(w, t).unwrap();
    let voice = VoiceLatents::load(v).unwrap();
    for g in goldens() {
        assert_eq!(model.tokenize(&g.text), g.tokens, "{}: tokens", g.name);
        let r = run_with_noise(&model, &voice, &g);
        assert_eq!(r.n_steps, g.n_steps, "{}: step count", g.name);
        assert_eq!(r.noise_used * model.latent_dim(), g.noise.len(), "{}: noise consumption", g.name);
        let d_lat = max_abs_diff(&r.latents, &g.latents);
        let d_wav = max_abs_diff(&r.wav, &g.wav);
        let snr = snr_db(&g.wav, &r.wav);
        println!(
            "fp32 {}: {} steps, latents max|d| = {d_lat:.3e}, wav max|d| = {d_wav:.3e}, wav SNR = {snr:.1} dB",
            g.name, r.n_steps
        );
        assert!(d_lat <= 1e-4, "{}: latents drift {d_lat:.3e}", g.name);
        // The decoder amplifies latent differences by one to two orders of
        // magnitude (a 1e-5 perturbation of the reference latents moves the
        // reference waveform by ~1e-3), so the end-to-end waveform is gated
        // more loosely than the latents and the decoder alone.
        assert!(d_wav <= 1e-2, "{}: wav drift {d_wav:.3e}", g.name);
        assert!(snr >= 40.0, "{}: wav SNR {snr:.1} dB", g.name);
    }
}

#[test]
fn decoder_matches_golden_wav() {
    let (Some(w), Some(t)) = (weights(), tokenizer()) else { return };
    let model = Linnet::load(w, t).unwrap();
    for g in goldens() {
        let wav = model.decode(&g.latents).unwrap();
        let d = max_abs_diff(&wav, &g.wav);
        println!("decoder {}: wav max|d| = {d:.3e}", g.name);
        assert!(d <= 1e-4, "{}: decoder drift {d:.3e}", g.name);
    }
}

/// The decoder takes the two frames of a step through the network at
/// once; that must not change a single bit against frame-by-frame
/// decoding, for both weight precisions and for the offline entry point.
#[test]
fn decoder_frame_batching_is_bit_exact() {
    let Some(t) = tokenizer() else { return };
    for w in [weights(), weights_fp16()].into_iter().flatten() {
        let model = Linnet::load(&w, &t).unwrap();
        let dec = model.decoder();
        let frames_per_step = model.latent_dim() / dec.config().latent_dim;
        assert_eq!(frames_per_step, 2);
        let pool = linnet_tts::ThreadPool::new(1);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for g in goldens() {
            let one = dec.decode(&g.latents, 1, &pool).unwrap();
            let two = dec.decode(&g.latents, 2, &pool).unwrap();
            assert_eq!(bits(&one), bits(&two), "{}: {} frame batching changed the bytes", g.name, w.display());
            // Frame by frame through the streaming entry point as well.
            let ld = dec.config().latent_dim;
            let spf = dec.samples_per_frame();
            let mut st = dec.new_state();
            let mut sc = dec.new_scratch(1);
            let mut streamed = vec![0.0f32; one.len()];
            for j in 0..g.latents.len() / ld {
                dec.stream_step(&g.latents[j * ld..(j + 1) * ld], &mut st, &mut sc, &mut streamed[j * spf..(j + 1) * spf], &pool).unwrap();
            }
            assert_eq!(bits(&one), bits(&streamed), "{}: streamed single frames", g.name);
            // Linnet::decode is the batched decode with the output delay applied.
            let delayed = model.decode(&g.latents).unwrap();
            let delay = model.config().output_delay;
            assert_eq!(bits(&delayed[..one.len() - delay]), bits(&one[delay..]), "{}: output delay", g.name);
            assert!(delayed[one.len() - delay..].iter().all(|v| *v == 0.0));
        }
    }
}

/// The thread count must not change a single bit of the latents, the
/// streamed audio or the offline decode, for either weight precision.
#[test]
fn thread_count_does_not_change_the_output() {
    let (Some(t), Some(v)) = (tokenizer(), voice()) else { return };
    let voice = VoiceLatents::load(v).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for w in [weights(), weights_fp16()].into_iter().flatten() {
        let model = Linnet::load(&w, &t).unwrap();
        for g in goldens() {
            let one = run_with_noise_params(&model, &voice, &g, SynthParams { threads: 1, ..g.params });
            assert_eq!(model.threads(), 1);
            let dec_one = model.decode(&g.latents).unwrap();
            for threads in [2usize, 3, 4, 8] {
                let r = run_with_noise_params(&model, &voice, &g, SynthParams { threads, ..g.params });
                assert_eq!(model.threads(), threads);
                assert_eq!(r.n_steps, one.n_steps, "{}: threads {threads} step count", g.name);
                assert_eq!(bits(&r.latents), bits(&one.latents), "{}: threads {threads} latents", g.name);
                assert_eq!(bits(&r.wav), bits(&one.wav), "{}: threads {threads} wav ({})", g.name, w.display());
                let dec = model.decode(&g.latents).unwrap();
                assert_eq!(bits(&dec), bits(&dec_one), "{}: threads {threads} decode", g.name);
            }
            // Automatic selection and an explicit setter on the model.
            model.set_threads(0);
            assert!((1..=4).contains(&model.threads()));
            let dec = model.decode(&g.latents).unwrap();
            assert_eq!(bits(&dec), bits(&dec_one), "{}: auto threads decode", g.name);
        }
    }
}

#[test]
fn fp16_drift_is_bounded() {
    let (Some(w), Some(t), Some(v)) = (weights_fp16(), tokenizer(), voice()) else { return };
    let model = Linnet::load(w, t).unwrap();
    let voice = VoiceLatents::load(v).unwrap();
    for g in goldens() {
        let r = run_with_noise(&model, &voice, &g);
        let n = r.n_steps.min(g.n_steps);
        assert!(n > 0, "{}: no steps", g.name);
        let ld = model.latent_dim();
        let sps = model.samples_per_step();
        let d_lat = max_abs_diff(&r.latents[..n * ld], &g.latents[..n * ld]);
        let d_wav = max_abs_diff(&r.wav[..n * sps], &g.wav[..n * sps]);
        let snr = snr_db(&g.wav[..n * sps], &r.wav[..n * sps]);
        // Decoder alone on the reference latents: isolates the fp16 rounding
        // of the decoder from the autoregressive drift of the LM.
        let dec_only = model.decode(&g.latents).unwrap();
        let d_dec = max_abs_diff(&dec_only, &g.wav);
        let snr_dec = snr_db(&g.wav, &dec_only);
        println!(
            "fp16 {}: {} steps (golden {}), latents max|d| = {d_lat:.3e}, end-to-end wav max|d| = {d_wav:.3e} (SNR {snr:.1} dB), decoder-only wav max|d| = {d_dec:.3e} (SNR {snr_dec:.1} dB)",
            g.name, r.n_steps, g.n_steps
        );
        assert_eq!(r.n_steps, g.n_steps, "{}: fp16 changed the step count", g.name);
        assert!(d_lat <= 1e-1, "{}: fp16 latents drift {d_lat:.3e}", g.name);
        assert!(d_dec <= 5e-2, "{}: fp16 decoder drift {d_dec:.3e}", g.name);
        // The end-to-end waveform is dominated by the decoder's sensitivity
        // to the fp16 latent drift (phase-level differences); bounded loosely.
        assert!(d_wav <= 5e-1, "{}: fp16 wav drift {d_wav:.3e}", g.name);
    }
}

#[test]
fn seeded_sessions_are_deterministic_and_offline_equals_stream() {
    let (Some(w), Some(t), Some(v)) = (weights(), tokenizer(), voice()) else { return };
    let model = Linnet::load(w, t).unwrap();
    let voice = VoiceLatents::load(v).unwrap();
    let params = SynthParams { max_frames: 6, seed: 3, ..SynthParams::default() };
    let a = model.synthesize("Hello there.", &voice, params).unwrap();
    let mut s = Session::new(&model, &voice, params).unwrap();
    s.prime("Hello there.").unwrap();
    let mut b = Vec::new();
    let mut latents = Vec::new();
    while let Some(c) = s.step().map(|c| c.to_vec()) {
        if !s.is_finished() {
            latents.extend_from_slice(s.latent());
        }
        b.extend_from_slice(&c);
    }
    assert_eq!(a, b);
    assert_eq!(a.len(), s.frames() * model.samples_per_step());
    let offline = model.decode(&latents).unwrap();
    assert_eq!(offline, b, "offline decode differs from the streamed chunks");
    // without guidance a single stream runs
    let c = model
        .synthesize("Hello there.", &voice, SynthParams { cfg: 1.0, max_frames: 3, ..params })
        .unwrap();
    assert_eq!(c.len(), 3 * model.samples_per_step());
}
