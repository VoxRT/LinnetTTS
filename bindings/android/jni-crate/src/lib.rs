//! JNI layer of the LinnetTTS Android library.
//!
//! Every `Java_com_voxrt_linnet_LinnetNative_*` function below backs one
//! `external fun` of the Kotlin object `com.voxrt.linnet.LinnetNative`.
//! Handles cross the boundary as `jlong` (a `Box` pointer); the Kotlin
//! wrappers own them and release them through the matching `*Free`.
//!
//! Lifetimes: a model handle is an `Arc<Linnet>`; a session handle keeps
//! its own clone of that `Arc`, so a session stays valid even if the
//! Kotlin model object is closed first. Voices are copied into the
//! session at creation and may be closed right after `sessionNew`.
//!
//! Errors never cross as panics: each entry point catches unwinds and
//! turns runtime errors into a `com.voxrt.linnet.LinnetException`
//! (constructor `(int code, String message)`), with the same codes as the
//! C API (`LINNET_ERR_*`). The function then returns a neutral value
//! (0, null, -1) that the Kotlin side never sees because the exception is
//! pending.

#![allow(clippy::missing_safety_doc)]
#![allow(non_snake_case)]

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use jni::objects::{JByteArray, JFloatArray, JIntArray, JObject, JString, JValue};
use jni::sys::{jboolean, jfloat, jint, jlong, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;

use linnet_tts::{Error, Linnet, Session, SynthParams, VoiceLatents, VoiceState};

pub use linnet_tts::c_api::{
    LINNET_ERR_FORMAT, LINNET_ERR_INTERNAL, LINNET_ERR_INVALID_ARG, LINNET_ERR_IO, LINNET_ERR_SHAPE,
};

const EXCEPTION_CLASS: &str = "com/voxrt/linnet/LinnetException";

// ---------------------------------------------------------------------------
// Handles

/// Model handle: the shared, immutable model.
pub struct ModelHandle {
    model: Arc<Linnet>,
}

/// Voice handle: the latents plus, after `voicePrepare`, the trunk state
/// of the voice prefix for one model.
pub struct VoiceHandle {
    voice: Arc<VoiceLatents>,
    state: Mutex<Option<VoiceState>>,
}

impl VoiceHandle {
    pub fn new(voice: VoiceLatents) -> Self {
        VoiceHandle { voice: Arc::new(voice), state: Mutex::new(None) }
    }

    /// A session for `m`: from the prepared state when it is `m`'s, else
    /// from the latents (the audio is the same either way).
    pub fn session<'m>(&self, m: &'m Linnet, params: SynthParams) -> Result<Session<'m>, Error> {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match st.as_ref() {
            Some(s) if s.is_for(m) => Session::with_voice_state(m, s, params),
            _ => Session::new(m, &self.voice, params),
        }
    }
}

/// Session handle. Field order matters: `session` borrows the `Linnet`
/// behind `model` with an erased lifetime and is dropped first (Rust
/// drops fields in declaration order); the `Arc` pins the model's address.
pub struct SessionHandle {
    session: Session<'static>,
    #[allow(dead_code)]
    model: Arc<Linnet>,
}

impl SessionHandle {
    /// Create a session that keeps the model alive for its own lifetime.
    pub fn new(model: Arc<Linnet>, voice: &VoiceHandle, params: SynthParams) -> Result<Self, Error> {
        // SAFETY: `model` is heap-allocated behind an `Arc` owned by this
        // struct and dropped only after `session`, so the reference stays
        // valid for the session's whole life.
        let m: &'static Linnet = unsafe { &*Arc::as_ptr(&model) };
        let session = voice.session(m, params)?;
        Ok(SessionHandle { session, model })
    }
}

fn into_handle<T>(v: T) -> jlong {
    Box::into_raw(Box::new(v)) as jlong
}

/// Borrow a handle. Returns `None` for 0 (a closed wrapper).
unsafe fn handle_ref<'a, T>(h: jlong) -> Option<&'a T> {
    (h as *const T).as_ref()
}

unsafe fn handle_mut<'a, T>(h: jlong) -> Option<&'a mut T> {
    (h as *mut T).as_mut()
}

unsafe fn free_handle<T>(h: jlong) {
    if h != 0 {
        drop(Box::from_raw(h as *mut T));
    }
}

// ---------------------------------------------------------------------------
// Errors

/// Status code of a runtime error, same numbering as the C API.
pub fn code_of(e: &Error) -> jint {
    match e {
        Error::Io(_) => LINNET_ERR_IO,
        Error::Format(_) => LINNET_ERR_FORMAT,
        Error::Shape(_) => LINNET_ERR_SHAPE,
        Error::Invalid(_) => LINNET_ERR_INVALID_ARG,
    }
}

fn throw(env: &mut JNIEnv, code: jint, msg: &str) {
    if env.exception_check().unwrap_or(false) {
        return; // keep the first exception
    }
    let thrown = (|| -> jni::errors::Result<()> {
        let cls = env.find_class(EXCEPTION_CLASS)?;
        let jmsg = env.new_string(msg)?;
        let ex = env.new_object(cls, "(ILjava/lang/String;)V", &[JValue::Int(code), JValue::Object(&jmsg)])?;
        env.throw(jni::objects::JThrowable::from(ex))
    })();
    if thrown.is_err() {
        let _ = env.exception_clear();
        let _ = env.throw_new("java/lang/RuntimeException", format!("linnet error {code}: {msg}"));
    }
}

fn throw_error(env: &mut JNIEnv, e: &Error) {
    throw(env, code_of(e), &e.to_string());
}

fn throw_invalid(env: &mut JNIEnv, msg: &str) {
    throw(env, LINNET_ERR_INVALID_ARG, msg);
}

/// Run `f`, converting a panic into a pending `LinnetException` and the
/// fallback value `default`.
fn guarded<'l, T>(env: &mut JNIEnv<'l>, default: T, f: impl FnOnce(&mut JNIEnv<'l>) -> T) -> T {
    match catch_unwind(AssertUnwindSafe(|| f(env))) {
        Ok(v) => v,
        Err(_) => {
            throw(env, LINNET_ERR_INTERNAL, "internal error");
            default
        }
    }
}

// ---------------------------------------------------------------------------
// Conversions

/// Validate and convert the Kotlin-side parameters.
pub fn params_from(temp: f32, cfg: f32, max_frames: i32, eos_threshold: f32, seed: i64, threads: i32) -> Result<SynthParams, Error> {
    if max_frames <= 0 {
        return Err(Error::Invalid(format!("maxFrames {max_frames} must be >= 1")));
    }
    if threads < 0 {
        return Err(Error::Invalid(format!("threads {threads} must be >= 0")));
    }
    Ok(SynthParams {
        temp,
        cfg,
        max_frames: max_frames as usize,
        eos_threshold,
        seed: seed as u64,
        threads: threads as usize,
    })
}

fn jstring_to_rust(env: &mut JNIEnv, s: &JString) -> Option<String> {
    if s.is_null() {
        throw_invalid(env, "string argument is null");
        return None;
    }
    match env.get_string(s) {
        Ok(js) => Some(js.into()),
        Err(_) => {
            throw_invalid(env, "string argument is not valid UTF-8");
            None
        }
    }
}

fn bytes_to_vec(env: &mut JNIEnv, a: &JByteArray) -> Option<Vec<u8>> {
    if a.is_null() {
        throw_invalid(env, "byte array argument is null");
        return None;
    }
    env.convert_byte_array(a).ok()
}

fn floats_to_vec(env: &mut JNIEnv, a: &JFloatArray) -> Option<Vec<f32>> {
    if a.is_null() {
        throw_invalid(env, "float array argument is null");
        return None;
    }
    let n = env.get_array_length(a).ok()? as usize;
    let mut v = vec![0.0f32; n];
    env.get_float_array_region(a, 0, &mut v).ok()?;
    Some(v)
}

fn vec_to_floats<'l>(env: &mut JNIEnv<'l>, v: &[f32]) -> JFloatArray<'l> {
    let Ok(arr) = env.new_float_array(v.len() as i32) else {
        return JFloatArray::default();
    };
    if env.set_float_array_region(&arr, 0, v).is_err() {
        return JFloatArray::default();
    }
    arr
}

fn vec_to_ints<'l>(env: &mut JNIEnv<'l>, v: &[u32]) -> JIntArray<'l> {
    let Ok(arr) = env.new_int_array(v.len() as i32) else {
        return JIntArray::default();
    };
    let ints: Vec<jint> = v.iter().map(|&t| t as jint).collect();
    if env.set_int_array_region(&arr, 0, &ints).is_err() {
        return JIntArray::default();
    }
    arr
}

// ---------------------------------------------------------------------------
// Library

#[no_mangle]
pub extern "system" fn Java_com_voxrt_linnet_LinnetNative_version<'l>(env: JNIEnv<'l>, _this: JObject<'l>) -> JString<'l> {
    env.new_string(linnet_tts::VERSION).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Model

#[no_mangle]
pub extern "system" fn Java_com_voxrt_linnet_LinnetNative_modelLoad<'l>(
    mut env: JNIEnv<'l>,
    _this: JObject<'l>,
    weights_path: JString<'l>,
    tokenizer_path: JString<'l>,
) -> jlong {
    guarded(&mut env, 0, |env| {
        let (Some(w), Some(t)) = (jstring_to_rust(env, &weights_path), jstring_to_rust(env, &tokenizer_path)) else {
            return 0;
        };
        match Linnet::load(w, t) {
            Ok(m) => into_handle(ModelHandle { model: Arc::new(m) }),
            Err(e) => {
                throw_error(env, &e);
                0
            }
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_voxrt_linnet_LinnetNative_modelFromBytes<'l>(
    mut env: JNIEnv<'l>,
    _this: JObject<'l>,
    weights: JByteArray<'l>,
    tokenizer: JByteArray<'l>,
) -> jlong {
    guarded(&mut env, 0, |env| {
        let (Some(w), Some(t)) = (bytes_to_vec(env, &weights), bytes_to_vec(env, &tokenizer)) else {
            return 0;
        };
        match Linnet::from_bytes(w, &t) {
            Ok(m) => into_handle(ModelHandle { model: Arc::new(m) }),
            Err(e) => {
                throw_error(env, &e);
                0
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_modelFree(_env: JNIEnv, _this: JObject, model: jlong) {
    free_handle::<ModelHandle>(model);
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_sampleRate(_env: JNIEnv, _this: JObject, model: jlong) -> jint {
    handle_ref::<ModelHandle>(model).map_or(0, |m| m.model.sample_rate() as jint)
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_samplesPerStep(_env: JNIEnv, _this: JObject, model: jlong) -> jint {
    handle_ref::<ModelHandle>(model).map_or(0, |m| m.model.samples_per_step() as jint)
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_latentDim(_env: JNIEnv, _this: JObject, model: jlong) -> jint {
    handle_ref::<ModelHandle>(model).map_or(0, |m| m.model.latent_dim() as jint)
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_outputDelay(_env: JNIEnv, _this: JObject, model: jlong) -> jint {
    handle_ref::<ModelHandle>(model).map_or(0, |m| m.model.config().output_delay as jint)
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_tokenize<'l>(
    mut env: JNIEnv<'l>,
    _this: JObject<'l>,
    model: jlong,
    text: JString<'l>,
) -> JIntArray<'l> {
    guarded(&mut env, JIntArray::default(), |env| {
        let Some(m) = handle_ref::<ModelHandle>(model) else {
            throw_invalid(env, "model is closed");
            return JIntArray::default();
        };
        let Some(t) = jstring_to_rust(env, &text) else {
            return JIntArray::default();
        };
        let ids = m.model.tokenize(&t);
        vec_to_ints(env, &ids)
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_synthesize<'l>(
    mut env: JNIEnv<'l>,
    _this: JObject<'l>,
    model: jlong,
    voice: jlong,
    text: JString<'l>,
    temp: jfloat,
    cfg: jfloat,
    max_frames: jint,
    eos_threshold: jfloat,
    seed: jlong,
    threads: jint,
) -> JFloatArray<'l> {
    guarded(&mut env, JFloatArray::default(), |env| {
        let (Some(m), Some(v)) = (handle_ref::<ModelHandle>(model), handle_ref::<VoiceHandle>(voice)) else {
            throw_invalid(env, "model or voice is closed");
            return JFloatArray::default();
        };
        let Some(t) = jstring_to_rust(env, &text) else {
            return JFloatArray::default();
        };
        let params = match params_from(temp, cfg, max_frames, eos_threshold, seed, threads) {
            Ok(p) => p,
            Err(e) => {
                throw_error(env, &e);
                return JFloatArray::default();
            }
        };
        match v.session(&m.model, params).and_then(|mut s| {
            s.prime(&t)?;
            Ok(s.collect())
        }) {
            Ok(pcm) => vec_to_floats(env, &pcm),
            Err(e) => {
                throw_error(env, &e);
                JFloatArray::default()
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Voice

#[no_mangle]
pub extern "system" fn Java_com_voxrt_linnet_LinnetNative_voiceLoad<'l>(mut env: JNIEnv<'l>, _this: JObject<'l>, path: JString<'l>) -> jlong {
    guarded(&mut env, 0, |env| {
        let Some(p) = jstring_to_rust(env, &path) else {
            return 0;
        };
        match VoiceLatents::load(p) {
            Ok(v) => into_handle(VoiceHandle::new(v)),
            Err(e) => {
                throw_error(env, &e);
                0
            }
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_voxrt_linnet_LinnetNative_voiceFromNpy<'l>(mut env: JNIEnv<'l>, _this: JObject<'l>, npy: JByteArray<'l>) -> jlong {
    guarded(&mut env, 0, |env| {
        let Some(b) = bytes_to_vec(env, &npy) else {
            return 0;
        };
        match VoiceLatents::from_npy_bytes(&b) {
            Ok(v) => into_handle(VoiceHandle::new(v)),
            Err(e) => {
                throw_error(env, &e);
                0
            }
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_com_voxrt_linnet_LinnetNative_voiceFromLatents<'l>(
    mut env: JNIEnv<'l>,
    _this: JObject<'l>,
    latents: JFloatArray<'l>,
    frames: jint,
    dim: jint,
) -> jlong {
    guarded(&mut env, 0, |env| {
        let Some(data) = floats_to_vec(env, &latents) else {
            return 0;
        };
        if frames <= 0 || dim <= 0 || data.len() != frames as usize * dim as usize {
            throw_invalid(
                env,
                &format!("latents has {} values, expected frames * dim = {} * {}", data.len(), frames, dim),
            );
            return 0;
        }
        match VoiceLatents::from_vec(data, dim as usize) {
            Ok(v) => into_handle(VoiceHandle::new(v)),
            Err(e) => {
                throw_error(env, &e);
                0
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_voiceFree(_env: JNIEnv, _this: JObject, voice: jlong) {
    free_handle::<VoiceHandle>(voice);
}

/// Run the voice through the model once and keep the result in the voice
/// handle, so that sessions created from it for that model skip the voice
/// part of the prime pass. Tied to the model; preparing again replaces the
/// state.
#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_voicePrepare(mut env: JNIEnv, _this: JObject, model: jlong, voice: jlong) {
    guarded(&mut env, (), |env| {
        let (Some(m), Some(v)) = (handle_ref::<ModelHandle>(model), handle_ref::<VoiceHandle>(voice)) else {
            throw_invalid(env, "model or voice is closed");
            return;
        };
        match m.model.voice_state(&v.voice) {
            Ok(s) => *v.state.lock().unwrap_or_else(|e| e.into_inner()) = Some(s),
            Err(e) => throw_error(env, &e),
        }
    })
}

/// True when `voicePrepare` ran for this model.
#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_voiceIsPrepared(_env: JNIEnv, _this: JObject, model: jlong, voice: jlong) -> jboolean {
    let (Some(m), Some(v)) = (handle_ref::<ModelHandle>(model), handle_ref::<VoiceHandle>(voice)) else {
        return 0;
    };
    v.state.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|s| s.is_for(&m.model)) as jboolean
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_voiceFrames(_env: JNIEnv, _this: JObject, voice: jlong) -> jint {
    handle_ref::<VoiceHandle>(voice).map_or(0, |v| v.voice.frames() as jint)
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_voiceDim(_env: JNIEnv, _this: JObject, voice: jlong) -> jint {
    handle_ref::<VoiceHandle>(voice).map_or(0, |v| v.voice.dim() as jint)
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_voiceLatents<'l>(mut env: JNIEnv<'l>, _this: JObject<'l>, voice: jlong) -> JFloatArray<'l> {
    guarded(&mut env, JFloatArray::default(), |env| match handle_ref::<VoiceHandle>(voice) {
        Some(v) => vec_to_floats(env, v.voice.latents()),
        None => {
            throw_invalid(env, "voice is closed");
            JFloatArray::default()
        }
    })
}

// ---------------------------------------------------------------------------
// Session

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_sessionNew(
    mut env: JNIEnv,
    _this: JObject,
    model: jlong,
    voice: jlong,
    temp: jfloat,
    cfg: jfloat,
    max_frames: jint,
    eos_threshold: jfloat,
    seed: jlong,
    threads: jint,
) -> jlong {
    guarded(&mut env, 0, |env| {
        let (Some(m), Some(v)) = (handle_ref::<ModelHandle>(model), handle_ref::<VoiceHandle>(voice)) else {
            throw_invalid(env, "model or voice is closed");
            return 0;
        };
        let params = match params_from(temp, cfg, max_frames, eos_threshold, seed, threads) {
            Ok(p) => p,
            Err(e) => {
                throw_error(env, &e);
                return 0;
            }
        };
        match SessionHandle::new(Arc::clone(&m.model), v, params) {
            Ok(s) => into_handle(s),
            Err(e) => {
                throw_error(env, &e);
                0
            }
        }
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_sessionFree(_env: JNIEnv, _this: JObject, session: jlong) {
    free_handle::<SessionHandle>(session);
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_sessionPrime<'l>(
    mut env: JNIEnv<'l>,
    _this: JObject<'l>,
    session: jlong,
    text: JString<'l>,
) {
    guarded(&mut env, (), |env| {
        let Some(s) = handle_mut::<SessionHandle>(session) else {
            throw_invalid(env, "session is closed");
            return;
        };
        let Some(t) = jstring_to_rust(env, &text) else {
            return;
        };
        if let Err(e) = s.session.prime(&t) {
            throw_error(env, &e);
        }
    })
}

/// Next chunk into `out` (capacity `out.length`, at least `samplesPerStep`).
/// Returns the number of samples written, or -1 once the utterance is over.
#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_sessionStep<'l>(
    mut env: JNIEnv<'l>,
    _this: JObject<'l>,
    session: jlong,
    out: JFloatArray<'l>,
) -> jint {
    guarded(&mut env, -1, |env| {
        let Some(s) = handle_mut::<SessionHandle>(session) else {
            throw_invalid(env, "session is closed");
            return -1;
        };
        if out.is_null() {
            throw_invalid(env, "output buffer is null");
            return -1;
        }
        let cap = env.get_array_length(&out).unwrap_or(0) as usize;
        let Some(chunk) = s.session.step() else {
            return -1;
        };
        if chunk.len() > cap {
            throw_invalid(env, &format!("output buffer holds {cap} samples, the chunk has {}", chunk.len()));
            return -1;
        }
        if env.set_float_array_region(&out, 0, chunk).is_err() {
            return -1;
        }
        chunk.len() as jint
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_sessionFrames(_env: JNIEnv, _this: JObject, session: jlong) -> jint {
    handle_ref::<SessionHandle>(session).map_or(0, |s| s.session.frames() as jint)
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_sessionFinished(_env: JNIEnv, _this: JObject, session: jlong) -> jboolean {
    match handle_ref::<SessionHandle>(session) {
        Some(s) if s.session.is_finished() => JNI_TRUE,
        _ => JNI_FALSE,
    }
}

#[no_mangle]
pub unsafe extern "system" fn Java_com_voxrt_linnet_LinnetNative_sessionLatent<'l>(mut env: JNIEnv<'l>, _this: JObject<'l>, session: jlong) -> JFloatArray<'l> {
    guarded(&mut env, JFloatArray::default(), |env| match handle_ref::<SessionHandle>(session) {
        Some(s) => vec_to_floats(env, s.session.latent()),
        None => {
            throw_invalid(env, "session is closed");
            JFloatArray::default()
        }
    })
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_validate_max_frames() {
        assert!(params_from(0.3, 2.0, 0, -1.0, 0, 0).is_err());
        assert!(params_from(0.3, 2.0, -5, -1.0, 0, 0).is_err());
        let p = params_from(0.5, 1.0, 12, 0.25, -1, 0).unwrap();
        assert_eq!(p.max_frames, 12);
        assert_eq!(p.seed, u64::MAX); // Kotlin Long -1 wraps to the full u64 range
        assert_eq!(p.temp, 0.5);
        assert_eq!(p.cfg, 1.0);
        assert_eq!(p.eos_threshold, 0.25);
    }

    #[test]
    fn error_codes_match_c_api() {
        assert_eq!(code_of(&Error::Io(std::io::Error::other("x"))), LINNET_ERR_IO);
        assert_eq!(code_of(&Error::Format("x".into())), LINNET_ERR_FORMAT);
        assert_eq!(code_of(&Error::Shape("x".into())), LINNET_ERR_SHAPE);
        assert_eq!(code_of(&Error::Invalid("x".into())), LINNET_ERR_INVALID_ARG);
    }

    #[test]
    fn handles_round_trip_and_null_is_none() {
        let h = into_handle(VoiceHandle::new(VoiceLatents::from_vec(vec![0.0; 96], 48).unwrap()));
        assert_ne!(h, 0);
        unsafe {
            assert_eq!(handle_ref::<VoiceHandle>(h).unwrap().voice.frames(), 2);
            assert!(handle_ref::<VoiceHandle>(0).is_none());
            free_handle::<VoiceHandle>(h);
            free_handle::<VoiceHandle>(0);
        }
    }

    fn assets() -> Option<(String, String, String)> {
        let dir = std::env::var("LINNET_ASSETS").unwrap_or_else(|_| {
            format!("{}/../../../local", env!("CARGO_MANIFEST_DIR"))
        });
        let w = format!("{dir}/linnet-en-35m.fp16.safetensors");
        let t = format!("{dir}/tokenizer.model");
        let v = format!("{dir}/voices/linnet-en-f1.npy");
        if [&w, &t, &v].iter().all(|p| std::path::Path::new(p).exists()) {
            Some((w, t, v))
        } else {
            None
        }
    }

    /// The session handle outlives the model handle it was created from
    /// (the `Arc` clone keeps the weights alive) and streams per contract.
    #[test]
    fn session_survives_model_handle_free() {
        let Some((w, t, v)) = assets() else {
            eprintln!("model assets not found, skipping");
            return;
        };
        let model = Arc::new(Linnet::load(&w, &t).unwrap());
        let voice = VoiceLatents::load(&v).unwrap();
        let sps = model.samples_per_step();
        let delay = model.config().output_delay;
        let offline = model.synthesize("Hello there.", &voice, SynthParams::default()).unwrap();

        let mh = into_handle(ModelHandle { model });
        let vh = VoiceHandle::new(voice);
        let sh = unsafe {
            let m = handle_ref::<ModelHandle>(mh).unwrap();
            into_handle(SessionHandle::new(Arc::clone(&m.model), &vh, SynthParams::default()).unwrap())
        };
        unsafe { free_handle::<ModelHandle>(mh) };

        let s = unsafe { handle_mut::<SessionHandle>(sh).unwrap() };
        s.session.prime("Hello there.").unwrap();
        let mut streamed = Vec::new();
        let mut n = 0;
        while let Some(c) = s.session.step() {
            if n == 0 {
                assert_eq!(c.len(), sps - delay);
            }
            streamed.extend_from_slice(c);
            n += 1;
        }
        assert_eq!(streamed.len(), offline.len());
        assert_eq!(streamed, offline);
        unsafe { free_handle::<SessionHandle>(sh) };
    }
}

#[cfg(test)]
mod prepared_voice {
    use super::*;

    fn assets() -> Option<(String, String, String)> {
        let dir = std::env::var("LINNET_ASSETS").unwrap_or_else(|_| format!("{}/../../../local", env!("CARGO_MANIFEST_DIR")));
        let w = format!("{dir}/linnet-en-35m.fp16.safetensors");
        let t = format!("{dir}/tokenizer.model");
        let v = format!("{dir}/voices/linnet-en-f1.npy");
        [&w, &t, &v].iter().all(|p| std::path::Path::new(p).exists()).then_some((w, t, v))
    }

    /// A prepared voice gives the same audio as an unprepared one, through
    /// both the session and the one-shot path, and belongs to its model.
    #[test]
    fn prepared_voice_matches_unprepared() {
        let Some((w, t, v)) = assets() else {
            eprintln!("model assets not found, skipping");
            return;
        };
        let model = Arc::new(Linnet::load(&w, &t).unwrap());
        let vh = VoiceHandle::new(VoiceLatents::load(&v).unwrap());
        let params = SynthParams { max_frames: 8, ..SynthParams::default() };
        let plain = model.synthesize("Hello there.", &vh.voice, params).unwrap();
        *vh.state.lock().unwrap() = Some(model.voice_state(&vh.voice).unwrap());
        assert!(vh.state.lock().unwrap().as_ref().unwrap().is_for(&model));
        let mut s = vh.session(&model, params).unwrap();
        assert!(s.voice_kept());
        s.prime("Hello there.").unwrap();
        assert_eq!(s.collect(), plain);
        // Another model falls back to the latents.
        let other = Arc::new(Linnet::load(&w, &t).unwrap());
        let mut s = vh.session(&other, params).unwrap();
        assert!(!s.voice_kept());
        s.prime("Hello there.").unwrap();
        assert_eq!(s.collect(), plain);
    }
}
