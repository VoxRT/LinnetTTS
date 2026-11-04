//! C ABI. Opaque handles for the model, a voice and a session; status
//! codes instead of exceptions; caller-owned output buffers. The header
//! `include/linnet_tts.h` mirrors these declarations.
//!
//! Ownership: every `*_load` / `*_new` returns a handle that the matching
//! `*_free` releases; the library never hands out memory the caller must
//! free. A session keeps a pointer to the model it was created from; the
//! model must outlive the session. Handles are not thread-safe
//! individually; distinct handles may be used from different threads, and
//! one model may serve any number of sessions concurrently.
//!
//! Panics never cross the boundary: each fallible entry point catches
//! them and returns `LINNET_ERR_INTERNAL`. The message of the most recent error on
//! the calling thread is available through `linnet_last_error`.

#![allow(clippy::missing_safety_doc)]
#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_int};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::error::Error;
use crate::session::{Linnet, Session, SynthParams};
use crate::voice::VoiceLatents;

pub const LINNET_OK: c_int = 0;
/// A pointer argument was null or a value out of range.
pub const LINNET_ERR_INVALID_ARG: c_int = -1;
/// A file could not be read.
pub const LINNET_ERR_IO: c_int = -2;
/// The file is not a container this runtime understands.
pub const LINNET_ERR_FORMAT: c_int = -3;
/// The container parsed but the model topology is wrong.
pub const LINNET_ERR_SHAPE: c_int = -4;
/// The output buffer is too small; `*out_written` holds the required size.
pub const LINNET_ERR_BUFFER_TOO_SMALL: c_int = -5;
/// Internal failure (caught panic).
pub const LINNET_ERR_INTERNAL: c_int = -6;

/// Sampling parameters; pass null for the defaults.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct linnet_params {
    pub temp: f32,
    pub cfg: f32,
    pub eos_threshold: f32,
    pub max_frames: u32,
    pub seed: u64,
}

/// Opaque model handle.
pub struct linnet_model {
    inner: Linnet,
}

/// Opaque voice handle.
pub struct linnet_voice {
    inner: VoiceLatents,
}

/// Opaque session handle. The `'static` lifetime is a promise made by the
/// caller: the model outlives the session.
pub struct linnet_session {
    inner: Session<'static>,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: &str) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

fn status_of(e: &Error) -> c_int {
    set_error(&e.to_string());
    match e {
        Error::Io(_) => LINNET_ERR_IO,
        Error::Format(_) => LINNET_ERR_FORMAT,
        Error::Shape(_) => LINNET_ERR_SHAPE,
        Error::Invalid(_) => LINNET_ERR_INVALID_ARG,
    }
}

fn guard(f: impl FnOnce() -> c_int) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(s) => s,
        Err(_) => {
            set_error("internal error");
            LINNET_ERR_INTERNAL
        }
    }
}

unsafe fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok()
}

/// The C parameters carry no thread count: a session keeps the model's
/// current setting (`linnet_set_threads`).
unsafe fn params_or_default(p: *const linnet_params, model: &Linnet) -> SynthParams {
    let threads = model.threads();
    if p.is_null() {
        return SynthParams { threads, ..SynthParams::default() };
    }
    let p = &*p;
    SynthParams {
        temp: p.temp,
        cfg: p.cfg,
        max_frames: p.max_frames as usize,
        eos_threshold: p.eos_threshold,
        seed: p.seed,
        threads,
    }
}

/// Version string of the library (static, do not free).
#[no_mangle]
pub extern "C" fn linnet_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Message of the most recent error on this thread (static until the
/// next call on the same thread; do not free).
#[no_mangle]
pub extern "C" fn linnet_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Load weights (safetensors) and tokenizer (SentencePiece `.model`).
#[no_mangle]
pub unsafe extern "C" fn linnet_model_load(
    weights_path: *const c_char,
    tokenizer_path: *const c_char,
    out_model: *mut *mut linnet_model,
) -> c_int {
    guard(|| {
        let (Some(w), Some(t)) = (cstr(weights_path), cstr(tokenizer_path)) else {
            set_error("null or non-UTF-8 path");
            return LINNET_ERR_INVALID_ARG;
        };
        if out_model.is_null() {
            set_error("out_model is null");
            return LINNET_ERR_INVALID_ARG;
        }
        match Linnet::load(w, t) {
            Ok(m) => {
                *out_model = Box::into_raw(Box::new(linnet_model { inner: m }));
                LINNET_OK
            }
            Err(e) => status_of(&e),
        }
    })
}

/// Release a model. Null is ignored. All sessions created from the model
/// must have been freed first.
#[no_mangle]
pub unsafe extern "C" fn linnet_model_free(model: *mut linnet_model) {
    if !model.is_null() {
        drop(Box::from_raw(model));
    }
}

/// Output sample rate in Hz (0 on a null handle).
#[no_mangle]
pub unsafe extern "C" fn linnet_sample_rate(model: *const linnet_model) -> u32 {
    if model.is_null() {
        return 0;
    }
    (*model).inner.sample_rate()
}

/// Threads the model uses from now on: 1 = single-threaded, 0 = automatic
/// (fast cores, at most 4). Applies to sessions created before and after
/// the call; the audio is identical for every value. Negative is invalid.
#[no_mangle]
pub unsafe extern "C" fn linnet_set_threads(model: *mut linnet_model, threads: c_int) -> c_int {
    guard(|| {
        if model.is_null() || threads < 0 {
            set_error("model is null or threads < 0");
            return LINNET_ERR_INVALID_ARG;
        }
        (*model).inner.set_threads(threads as usize);
        LINNET_OK
    })
}

/// Threads the model currently uses (0 on a null handle).
#[no_mangle]
pub unsafe extern "C" fn linnet_threads(model: *const linnet_model) -> c_int {
    if model.is_null() {
        return 0;
    }
    (*model).inner.threads() as c_int
}

/// Largest chunk `linnet_session_step` can write (samples per LM step).
#[no_mangle]
pub unsafe extern "C" fn linnet_samples_per_step(model: *const linnet_model) -> usize {
    if model.is_null() {
        return 0;
    }
    (*model).inner.samples_per_step()
}

/// Load `[T, 48]` voice latents from a `.npy` file.
#[no_mangle]
pub unsafe extern "C" fn linnet_voice_load(path: *const c_char, out_voice: *mut *mut linnet_voice) -> c_int {
    guard(|| {
        let Some(p) = cstr(path) else {
            set_error("null or non-UTF-8 path");
            return LINNET_ERR_INVALID_ARG;
        };
        if out_voice.is_null() {
            set_error("out_voice is null");
            return LINNET_ERR_INVALID_ARG;
        }
        match VoiceLatents::load(p) {
            Ok(v) => {
                *out_voice = Box::into_raw(Box::new(linnet_voice { inner: v }));
                LINNET_OK
            }
            Err(e) => status_of(&e),
        }
    })
}

/// Build a voice from an in-memory `[frames × dim]` f32 buffer. The width
/// is checked against the model when a session is created (a voice is not
/// tied to a model here).
#[no_mangle]
pub unsafe extern "C" fn linnet_voice_from_latents(
    latents: *const f32,
    frames: usize,
    dim: usize,
    out_voice: *mut *mut linnet_voice,
) -> c_int {
    guard(|| {
        if latents.is_null() || out_voice.is_null() || frames == 0 || dim == 0 {
            set_error("latents/out_voice is null or frames/dim is 0");
            return LINNET_ERR_INVALID_ARG;
        }
        let Some(n) = frames.checked_mul(dim) else {
            set_error("frames * dim overflows");
            return LINNET_ERR_INVALID_ARG;
        };
        let data = core::slice::from_raw_parts(latents, n).to_vec();
        match VoiceLatents::from_vec(data, dim) {
            Ok(v) => {
                *out_voice = Box::into_raw(Box::new(linnet_voice { inner: v }));
                LINNET_OK
            }
            Err(e) => status_of(&e),
        }
    })
}

/// Release a voice. Null is ignored.
#[no_mangle]
pub unsafe extern "C" fn linnet_voice_free(voice: *mut linnet_voice) {
    if !voice.is_null() {
        drop(Box::from_raw(voice));
    }
}

/// Create a session for `voice`. `params` may be null (defaults). The
/// model must stay alive until `linnet_session_free`.
#[no_mangle]
pub unsafe extern "C" fn linnet_session_new(
    model: *const linnet_model,
    voice: *const linnet_voice,
    params: *const linnet_params,
    out_session: *mut *mut linnet_session,
) -> c_int {
    guard(|| {
        if model.is_null() || voice.is_null() || out_session.is_null() {
            set_error("model, voice or out_session is null");
            return LINNET_ERR_INVALID_ARG;
        }
        let m: &'static Linnet = &(*model).inner;
        let p = params_or_default(params, m);
        match Session::new(m, &(*voice).inner, p) {
            Ok(s) => {
                *out_session = Box::into_raw(Box::new(linnet_session { inner: s }));
                LINNET_OK
            }
            Err(e) => status_of(&e),
        }
    })
}

/// Release a session. Null is ignored.
#[no_mangle]
pub unsafe extern "C" fn linnet_session_free(session: *mut linnet_session) {
    if !session.is_null() {
        drop(Box::from_raw(session));
    }
}

/// Start an utterance: tokenize `text` (UTF-8, NUL-terminated) and run
/// the prefix. May be called again on the same session for a new text.
#[no_mangle]
pub unsafe extern "C" fn linnet_session_prime(session: *mut linnet_session, text: *const c_char) -> c_int {
    guard(|| {
        if session.is_null() {
            set_error("session is null");
            return LINNET_ERR_INVALID_ARG;
        }
        let Some(t) = cstr(text) else {
            set_error("text is null or not UTF-8");
            return LINNET_ERR_INVALID_ARG;
        };
        match (*session).inner.prime(t) {
            Ok(()) => LINNET_OK,
            Err(e) => status_of(&e),
        }
    })
}

/// Next chunk of audio into `out_pcm` (capacity `cap` samples, at least
/// `linnet_samples_per_step`, checked before the step runs so that no
/// chunk is lost). `*out_written` receives the number of samples
/// written; 0 means the utterance is finished (or the session was never
/// primed).
#[no_mangle]
pub unsafe extern "C" fn linnet_session_step(
    session: *mut linnet_session,
    out_pcm: *mut f32,
    cap: usize,
    out_written: *mut usize,
) -> c_int {
    guard(|| {
        if session.is_null() || out_written.is_null() || (out_pcm.is_null() && cap > 0) {
            set_error("session/out_written is null, or out_pcm is null with cap > 0");
            return LINNET_ERR_INVALID_ARG;
        }
        *out_written = 0;
        let s = &mut (*session).inner;
        if cap < s.samples_per_step() {
            set_error("output buffer must hold at least linnet_samples_per_step samples");
            return LINNET_ERR_INVALID_ARG;
        }
        let Some(chunk) = s.step() else {
            return LINNET_OK;
        };
        debug_assert!(chunk.len() <= cap);
        core::slice::from_raw_parts_mut(out_pcm, chunk.len()).copy_from_slice(chunk);
        *out_written = chunk.len();
        LINNET_OK
    })
}

/// Whole utterance into `out_pcm` (capacity `cap`). `*out_written` always
/// receives the full length; when it exceeds `cap` the status is
/// `LINNET_ERR_BUFFER_TOO_SMALL` and nothing is written, so
/// `out_pcm = NULL, cap = 0` only reports the size (the audio is still
/// synthesized).
#[no_mangle]
pub unsafe extern "C" fn linnet_synthesize(
    model: *const linnet_model,
    voice: *const linnet_voice,
    text: *const c_char,
    params: *const linnet_params,
    out_pcm: *mut f32,
    cap: usize,
    out_written: *mut usize,
) -> c_int {
    guard(|| {
        if model.is_null() || voice.is_null() || out_written.is_null() || (out_pcm.is_null() && cap > 0) {
            set_error("model/voice/out_written is null, or out_pcm is null with cap > 0");
            return LINNET_ERR_INVALID_ARG;
        }
        let Some(t) = cstr(text) else {
            set_error("text is null or not UTF-8");
            return LINNET_ERR_INVALID_ARG;
        };
        let p = params_or_default(params, &(*model).inner);
        let wav = match (*model).inner.synthesize(t, &(*voice).inner, p) {
            Ok(w) => w,
            Err(e) => return status_of(&e),
        };
        *out_written = wav.len();
        if wav.len() > cap {
            set_error("output buffer too small");
            return LINNET_ERR_BUFFER_TOO_SMALL;
        }
        if !wav.is_empty() {
            core::slice::from_raw_parts_mut(out_pcm, wav.len()).copy_from_slice(&wav);
        }
        LINNET_OK
    })
}
