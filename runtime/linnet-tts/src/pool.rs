//! Persistent worker pool for the parallel regions of a synthesis step.
//!
//! A region is a fixed list of chunks (`0..n`) handed out through an
//! atomic counter: whichever thread is free takes the next chunk, so
//! slow cores and throttling only move work around (dynamic scheduling,
//! no affinity, no priorities). The mapping from a chunk to the values it
//! produces never changes and every chunk is computed by exactly one
//! thread with the single-threaded code, so the result does not depend on
//! the thread count — a chunk is a set of output rows, heads, channels
//! or columns, never part of a reduction.
//!
//! Workers spin for a short while after a region (the gaps between the
//! regions of one step are a few microseconds), then park on a condition
//! variable until the next region is published. The calling thread takes
//! part in every region. With one thread there are no workers and
//! [`ThreadPool::run`] is a plain loop. On wasm32 the pool is compiled
//! without workers.

use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

/// Hard cap on the thread count any caller may ask for.
pub const MAX_THREADS: usize = 32;

/// One published region: the closure, the chunk count and the counters.
/// Lives on the stack of [`ThreadPool::run`]; workers only touch it while
/// `Shared::active` accounts for them.
struct Job<'a> {
    f: &'a (dyn Fn(usize) + Sync + 'a),
    n: usize,
    next: AtomicUsize,
    done: AtomicUsize,
}

struct Shared {
    job: AtomicPtr<Job<'static>>,
    /// Workers that have loaded a non-null job pointer and not yet let go.
    active: AtomicUsize,
    /// Workers waiting on the condition variable.
    parked: AtomicUsize,
    stop: AtomicBool,
    lock: Mutex<()>,
    cv: Condvar,
}

impl Shared {
    /// Take chunks of the current job until none are left. Returns
    /// whether a job was found at all.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn try_work(&self) -> bool {
        self.active.fetch_add(1, SeqCst);
        let p = self.job.load(SeqCst);
        if p.is_null() {
            self.active.fetch_sub(1, SeqCst);
            return false;
        }
        // SAFETY: `run` keeps the job alive until `active` drops to zero
        // after it has cleared the pointer, and we incremented `active`
        // before loading it.
        let job = unsafe { &*p };
        loop {
            let i = job.next.fetch_add(1, AcqRel);
            if i >= job.n {
                break;
            }
            (job.f)(i);
            job.done.fetch_add(1, Release);
        }
        self.active.fetch_sub(1, SeqCst);
        true
    }
}

/// The pool. `threads` counts the caller, so `threads - 1` workers exist.
pub struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
    threads: usize,
}

#[cfg(not(target_arch = "wasm32"))]
const SPIN_BEFORE_PARK: Duration = Duration::from_micros(300);

#[cfg(not(target_arch = "wasm32"))]
fn spawn_workers(shared: &Arc<Shared>, threads: usize) -> Vec<std::thread::JoinHandle<()>> {
    (1..threads)
        .map(|i| {
            let sh = Arc::clone(shared);
            std::thread::Builder::new()
                .name(format!("linnet-worker-{i}"))
                .spawn(move || worker(sh))
                .expect("spawn worker thread")
        })
        .collect()
}

/// No threads on wasm32: the pool runs every region inline.
#[cfg(target_arch = "wasm32")]
fn spawn_workers(_shared: &Arc<Shared>, _threads: usize) -> Vec<std::thread::JoinHandle<()>> {
    Vec::new()
}

#[cfg(not(target_arch = "wasm32"))]
fn worker(sh: Arc<Shared>) {
    let mut idle_since: Option<Instant> = None;
    loop {
        if sh.stop.load(Acquire) {
            return;
        }
        if sh.try_work() {
            idle_since = None;
            continue;
        }
        match idle_since {
            None => idle_since = Some(Instant::now()),
            Some(t0) if t0.elapsed() < SPIN_BEFORE_PARK => std::hint::spin_loop(),
            Some(_) => {
                sh.parked.fetch_add(1, SeqCst);
                let mut g = sh.lock.lock().unwrap();
                while sh.job.load(SeqCst).is_null() && !sh.stop.load(Acquire) {
                    g = sh.cv.wait(g).unwrap();
                }
                drop(g);
                sh.parked.fetch_sub(1, SeqCst);
                idle_since = None;
            }
        }
    }
}

impl ThreadPool {
    /// A pool of `threads` threads including the caller (0 and 1 both
    /// mean single-threaded; values above [`MAX_THREADS`] are capped).
    pub fn new(threads: usize) -> Self {
        let threads = threads.clamp(1, MAX_THREADS);
        let shared = Arc::new(Shared {
            job: AtomicPtr::new(null_mut()),
            active: AtomicUsize::new(0),
            parked: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        });
        let workers = spawn_workers(&shared, threads);
        let threads = workers.len() + 1;
        ThreadPool { shared, workers, threads }
    }

    /// Threads taking part in a region, caller included.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `f(0..n)` with every chunk executed exactly once by some thread
    /// of the pool (the caller included) and return once all are done.
    /// Regions do not nest.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if self.workers.is_empty() || n <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let job = Job { f, n, next: AtomicUsize::new(0), done: AtomicUsize::new(0) };
        let sh = &self.shared;
        sh.job.store(&job as *const Job<'_> as *mut Job<'static>, SeqCst);
        if sh.parked.load(SeqCst) > 0 {
            let _g = sh.lock.lock().unwrap();
            sh.cv.notify_all();
        }
        loop {
            let i = job.next.fetch_add(1, AcqRel);
            if i >= job.n {
                break;
            }
            (job.f)(i);
            job.done.fetch_add(1, Release);
        }
        while job.done.load(Acquire) < n {
            std::hint::spin_loop();
        }
        sh.job.store(null_mut(), SeqCst);
        // A worker may still hold the pointer it loaded before the store.
        while sh.active.load(SeqCst) != 0 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Release);
        {
            let _g = self.shared.lock.lock().unwrap();
            self.shared.cv.notify_all();
        }
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

/// A raw pointer that may be shared with a region's closure so that each
/// chunk can write its own, disjoint part of one output buffer.
#[derive(Clone, Copy)]
pub struct SendPtr<T>(*mut T);

// SAFETY: the pointer is only dereferenced through `slice`, whose caller
// guarantees that concurrent chunks touch disjoint ranges.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> Default for SendPtr<T> {
    fn default() -> Self {
        SendPtr(null_mut())
    }
}

impl<T> SendPtr<T> {
    pub fn new(s: &mut [T]) -> Self {
        SendPtr(s.as_mut_ptr())
    }

    /// `&mut buf[off..off + len]`.
    ///
    /// # Safety
    /// The range must lie inside the buffer the pointer was made from, and
    /// no other live reference (from any thread) may overlap it.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, off: usize, len: usize) -> &mut [T] {
        core::slice::from_raw_parts_mut(self.0.add(off), len)
    }
}

/// Thread count for `threads = 0`: the number of fast cores, at most 4.
///
/// Android / Linux: cores whose `cpufreq/cpuinfo_max_freq` equals the
/// highest one (the big cluster). iOS / macOS: `hw.perflevel0.physicalcpu`
/// (performance cores), else `hw.physicalcpu`. Elsewhere the available
/// parallelism. wasm32: 1.
pub fn auto_threads() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        return 1;
    }
    #[allow(unreachable_code)]
    {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if let Some(n) = big_cores_sysfs() {
            return n.clamp(1, 4);
        }
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        if let Some(n) = perf_cores_sysctl() {
            return n.clamp(1, 4);
        }
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 4)
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn big_cores_sysfs() -> Option<usize> {
    let mut freqs = Vec::new();
    for i in 0..256 {
        let dir = format!("/sys/devices/system/cpu/cpu{i}");
        if !std::path::Path::new(&dir).exists() {
            break;
        }
        if let Ok(s) = std::fs::read_to_string(format!("{dir}/cpufreq/cpuinfo_max_freq")) {
            if let Ok(f) = s.trim().parse::<u64>() {
                freqs.push(f);
            }
        }
    }
    let max = *freqs.iter().max()?;
    Some(freqs.iter().filter(|&&f| f == max).count())
}

#[cfg(any(target_os = "ios", target_os = "macos"))]
fn perf_cores_sysctl() -> Option<usize> {
    use core::ffi::{c_char, c_int, c_void};
    extern "C" {
        fn sysctlbyname(name: *const c_char, oldp: *mut c_void, oldlenp: *mut usize, newp: *const c_void, newlen: usize) -> c_int;
    }
    fn read(name: &[u8]) -> Option<usize> {
        let mut v: i32 = 0;
        let mut len = core::mem::size_of::<i32>();
        // SAFETY: NUL-terminated name, correctly sized output buffer.
        let rc = unsafe { sysctlbyname(name.as_ptr() as *const c_char, &mut v as *mut i32 as *mut c_void, &mut len, core::ptr::null(), 0) };
        (rc == 0 && v > 0).then_some(v as usize)
    }
    read(b"hw.perflevel0.physicalcpu\0").or_else(|| read(b"hw.physicalcpu\0"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn every_chunk_runs_once_for_any_thread_count() {
        for threads in [1usize, 2, 3, 4, 8] {
            let pool = ThreadPool::new(threads);
            assert_eq!(pool.threads(), if cfg!(target_arch = "wasm32") { 1 } else { threads });
            for n in [0usize, 1, 2, 5, 33, 1000] {
                let counts: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
                pool.run(n, &|i| {
                    counts[i].fetch_add(1, SeqCst);
                });
                assert!(counts.iter().all(|c| c.load(SeqCst) == 1), "threads={threads} n={n}");
            }
        }
    }

    #[test]
    fn disjoint_writes_through_send_ptr_are_complete() {
        let pool = ThreadPool::new(4);
        let n = 4096;
        let mut out = vec![0u32; n];
        let p = SendPtr::new(&mut out);
        for round in 0..200u32 {
            pool.run(n / 16, &|c| {
                // SAFETY: chunk c owns out[c*16..(c+1)*16].
                let s = unsafe { p.slice(c * 16, 16) };
                for (k, v) in s.iter_mut().enumerate() {
                    *v = round * 100_000 + (c * 16 + k) as u32;
                }
            });
            for (i, v) in out.iter().enumerate() {
                assert_eq!(*v, round * 100_000 + i as u32);
            }
        }
    }

    #[test]
    fn workers_park_and_wake() {
        let pool = ThreadPool::new(3);
        let hits = AtomicU32::new(0);
        pool.run(64, &|_| {
            hits.fetch_add(1, SeqCst);
        });
        // Long enough for the workers to park, then publish again.
        std::thread::sleep(std::time::Duration::from_millis(20));
        pool.run(64, &|_| {
            hits.fetch_add(1, SeqCst);
        });
        assert_eq!(hits.load(SeqCst), 128);
        drop(pool);
    }

    #[test]
    fn auto_thread_count_is_sane() {
        let n = auto_threads();
        assert!((1..=4).contains(&n), "{n}");
    }
}
