//! Command-line synthesis with timing report.
//!
//! ```text
//! cargo run --release --example synth -- <weights.safetensors> <tokenizer.model> <voice.npy> "text" [out.wav] [temp] [cfg] [seed]
//! ```

use std::io::Write;
use std::time::Instant;

use linnet_tts::{Linnet, Session, SynthParams, VoiceLatents};

fn write_wav(path: &str, pcm: &[f32], rate: u32) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let data_len = (pcm.len() * 2) as u32;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&rate.to_le_bytes())?;
    f.write_all(&(rate * 2).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    let mut buf = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: synth <weights> <tokenizer.model> <voice.npy> \"text\" [out.wav] [temp] [cfg] [seed]");
        std::process::exit(2);
    }
    let t0 = Instant::now();
    let model = Linnet::load(&args[1], &args[2]).unwrap_or_else(|e| {
        eprintln!("load: {e}");
        std::process::exit(1)
    });
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let voice = VoiceLatents::load(&args[3]).unwrap_or_else(|e| {
        eprintln!("voice: {e}");
        std::process::exit(1)
    });
    let mut params = SynthParams::default();
    if let Some(t) = args.get(6) {
        params.temp = t.parse().expect("temp");
    }
    if let Some(c) = args.get(7) {
        params.cfg = c.parse().expect("cfg");
    }
    if let Some(s) = args.get(8) {
        params.seed = s.parse().expect("seed");
    }

    let mut session = Session::new(&model, &voice, params).expect("session");
    let t1 = Instant::now();
    session.prime(&args[4]).expect("prime");
    let prime_ms = t1.elapsed().as_secs_f64() * 1e3;

    let mut pcm = Vec::new();
    let mut latents = Vec::new();
    let mut step_ms = Vec::new();
    let mut first_audio_ms = None;
    loop {
        let ts = Instant::now();
        let Some(chunk) = session.step().map(|c| c.to_vec()) else { break };
        let ms = ts.elapsed().as_secs_f64() * 1e3;
        if first_audio_ms.is_none() {
            first_audio_ms = Some(prime_ms + ms);
        }
        if chunk.len() == model.samples_per_step() || pcm.is_empty() {
            step_ms.push(ms);
        }
        if !session.is_finished() {
            latents.extend_from_slice(session.latent());
        }
        pcm.extend_from_slice(&chunk);
    }
    let total_ms = t1.elapsed().as_secs_f64() * 1e3;
    let audio_s = pcm.len() as f64 / model.sample_rate() as f64;
    let n = step_ms.len().max(1) as f64;
    let mean_step = step_ms.iter().sum::<f64>() / n;
    let min_step = step_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_step = step_ms.iter().cloned().fold(0.0, f64::max);

    println!("tokens: {:?}", model.tokenize(&args[4]));
    println!("load {load_ms:.1} ms, prime {prime_ms:.2} ms, first audio {:.2} ms", first_audio_ms.unwrap_or(0.0));
    println!(
        "{} steps, {:.3} s audio, synthesis {total_ms:.1} ms, RTF {:.2} %",
        session.frames(),
        audio_s,
        100.0 * total_ms / 1e3 / audio_s.max(1e-9)
    );
    println!("per step (LM + 2 decoder frames): mean {mean_step:.2} ms, min {min_step:.2}, max {max_step:.2}");
    // Decoder alone on the same latents: isolates its share of the step.
    let n_frames = latents.len() / model.config().decoder.latent_dim;
    let td = Instant::now();
    let _ = model.decode(&latents).expect("decode");
    let dec_ms_per_frame = td.elapsed().as_secs_f64() * 1e3 / n_frames.max(1) as f64;
    println!(
        "decoder {dec_ms_per_frame:.2} ms per codec frame (40 ms audio); LM alone ~{:.2} ms per step",
        mean_step - 2.0 * dec_ms_per_frame
    );
    if let Some(out) = args.get(5).filter(|p| !p.is_empty()) {
        write_wav(out, &pcm, model.sample_rate()).expect("wav");
        println!("wrote {out}");
    }
}
