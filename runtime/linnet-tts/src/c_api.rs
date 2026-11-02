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
//! Panics never cross the boundary: each entry point catches them and
//! returns `LINNET_ERR_INTERNAL`. The message of the most recent error on
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

unsafe fn params_or_default(p: *const linnet_params) -> SynthParams {
    if p.is_null() {
        return SynthParams::default();
    }
    let p = &*p;
    SynthParams {
        temp: p.temp,
        cfg: p.cfg,
        max_frames: p.max_frames as usize,
        eos_threshold: p.eos_threshold,
        seed: p.seed,
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

/// Build a voice from an in-memory `[frames × dim]` f32 buffer.
#[no_mangle]
pub unsafe extern "C" fn linnet_voice_from_latents(
    latents: *const f32,
    frames: usize,
    dim: usize,
    out_voice: *mut *mut linnet_voice,
) -> c_int {
    guard(|| {
        if latents.is_null() || out_voice.is_null() || frames == 0 || dim == 0 {
            return LINNET_ERR_INVALID_ARG;
        }
        let data = core::slice::from_raw_parts(latents, frames * dim).to_vec();
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
            return LINNET_ERR_INVALID_ARG;
        }
        let m: &'static Linnet = &(*model).inner;
        let p = params_or_default(params);
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
/// `linnet_samples_per_step`). `*out_written` receives the number of
/// samples written; 0 means the utterance is finished.
#[no_mangle]
pub unsafe extern "C" fn linnet_session_step(
    session: *mut linnet_session,
    out_pcm: *mut f32,
    cap: usize,
    out_written: *mut usize,
) -> c_int {
    guard(|| {
        if session.is_null() || out_written.is_null() || (out_pcm.is_null() && cap > 0) {
            return LINNET_ERR_INVALID_ARG;
        }
        *out_written = 0;
        let s = &mut (*session).inner;
        let Some(chunk) = s.step() else {
            return LINNET_OK;
        };
        if chunk.len() > cap {
            *out_written = chunk.len();
            set_error("output buffer too small");
            return LINNET_ERR_BUFFER_TOO_SMALL;
        }
        core::slice::from_raw_parts_mut(out_pcm, chunk.len()).copy_from_slice(chunk);
        *out_written = chunk.len();
        LINNET_OK
    })
}

/// Whole utterance into `out_pcm` (capacity `cap`). `*out_written` always
/// receives the full length; with `out_pcm = NULL, cap = 0` the call only
/// reports the size (the audio is still synthesized).
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
            return LINNET_ERR_INVALID_ARG;
        }
        let Some(t) = cstr(text) else {
            set_error("text is null or not UTF-8");
            return LINNET_ERR_INVALID_ARG;
        };
        let p = params_or_default(params);
        let wav = match (*model).inner.synthesize(t, &(*voice).inner, p) {
            Ok(w) => w,
            Err(e) => return status_of(&e),
        };
        *out_written = wav.len();
        if wav.len() > cap {
            set_error("output buffer too small");
            return LINNET_ERR_BUFFER_TOO_SMALL;
        }
        core::slice::from_raw_parts_mut(out_pcm, wav.len()).copy_from_slice(&wav);
        LINNET_OK
    })
}
