use std::env;
use std::error::Error;
use std::io;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

use rten_base::num::AsUsize;

/// A wrapper around the Rayon thread pool used to run models.
///
/// Explicit construction fails if the requested native threads cannot be
/// created. The implicit global pool retains Rayon's unsupported-target
/// behavior on platforms without native threads (eg. WebAssembly).
#[derive(Debug)]
pub struct ThreadPool {
    /// None only for the implicit unsupported-target fallback or during drop.
    pool: Option<rayon::ThreadPool>,
    workers: WorkerThreads,
}

#[derive(Debug, Default)]
struct WorkerThreads(Vec<JoinHandle<()>>);

impl WorkerThreads {
    fn join(handles: Vec<JoinHandle<()>>) -> thread::Result<()> {
        let mut failure = None;
        for handle in handles {
            if let Err(error) = handle.join()
                && failure.is_none()
            {
                failure = Some(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for WorkerThreads {
    fn drop(&mut self) {
        let handles = std::mem::take(&mut self.0);
        if handles
            .iter()
            .any(|handle| handle.thread().id() == thread::current().id())
        {
            // An asynchronous Rayon job may release the last pool owner.
            // Retain every native join on another thread until that job exits.
            let pending = Arc::new(Mutex::new(handles));
            let owned = Arc::clone(&pending);
            let reaper = thread::Builder::new()
                .name("rten-pool-reaper".into())
                .spawn(move || {
                    let handles = std::mem::take(
                        &mut *owned
                            .lock()
                            .expect("private thread retirement is not poisoned"),
                    );
                    if Self::join(handles).is_err() {
                        eprintln!("RTen private worker panicked during deferred retirement");
                        std::process::abort();
                    }
                });
            if let Err(error) = reaper {
                eprintln!("RTen could not retain self-owned worker retirement: {error}");
                std::process::abort();
            }
        } else if let Err(error) = Self::join(handles) {
            if thread::panicking() {
                eprintln!("RTen private worker also panicked during owner unwinding");
            } else {
                std::panic::resume_unwind(error);
            }
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        drop(self.pool.take());
        drop(std::mem::take(&mut self.workers));
    }
}

impl ThreadPool {
    /// Run a function in the thread pool.
    ///
    /// This corresponds to [`rayon::ThreadPool::install`], except on platforms
    /// where threading is not supported, where it just runs `op` directly.
    pub fn run<R: Send, Op: FnOnce() -> R + Send>(&self, op: Op) -> R {
        if let Some(pool) = self.pool.as_ref() {
            pool.install(op)
        } else {
            op()
        }
    }

    /// Create a thread pool with a given number of threads.
    ///
    /// Dropping the pool from outside its workers joins their native threads.
    /// A last owner released by an asynchronous job transfers those joins to
    /// a retirement thread, which can wait for the releasing worker itself.
    /// If construction fails, already-created workers are joined before the
    /// original error is returned; there is no smaller or implicit global pool.
    pub fn with_num_threads(num_threads: usize) -> Result<ThreadPool, rayon::ThreadPoolBuildError> {
        let mut workers = WorkerThreads::default();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|index| format!("rten-{}", index))
            .spawn_handler(|worker| {
                let mut builder = thread::Builder::new();
                if let Some(name) = worker.name() {
                    builder = builder.name(name.to_owned());
                }
                if let Some(stack_size) = worker.stack_size() {
                    builder = builder.stack_size(stack_size);
                }
                workers.0.push(builder.spawn(move || worker.run())?);
                Ok(())
            })
            .build();

        match pool {
            Ok(pool) => Ok(ThreadPool {
                pool: Some(pool),
                workers,
            }),
            Err(error) => {
                // Rayon requests termination when build fails. Join the actual
                // native threads before returning the construction error.
                drop(workers);
                Err(error)
            }
        }
    }
}

/// Return the optimal number of cores to use for maximum performance.
///
/// This may be less than the total number of cores on systems with heterogenous
/// cores (eg. a mix of performance and efficiency).
fn optimal_core_count() -> u32 {
    #[allow(unused_mut)]
    let mut core_count = num_cpus::get_physical().max(1) as u32;

    #[cfg(target_os = "macos")]
    {
        use rten_simd::isa_detection::macos::sysctl_int;
        if let Ok(perf_core_count) = sysctl_int(c"hw.perflevel0.physicalcpu") {
            core_count = core_count.clamp(1, perf_core_count as u32);
        }
    }

    core_count
}

/// Return the [Rayon][rayon] thread pool which is used to execute RTen models.
///
/// This differs from Rayon's default global thread pool in that it is tuned for
/// CPU rather than IO-bound work by choosing a thread count based on the number
/// of physical rather than logical cores.
///
/// The thread count can be overridden at the process level by setting the
/// `RTEN_NUM_THREADS` environment variable, whose value must be a number
/// between 1 and the logical core count.
///
/// The thread count can be overridden for each model run by configuring a
/// custom thread pool in [`RunOptions`](crate::RunOptions).
///
/// To run your own tasks in this thread pool, you can use
/// [`ThreadPool::run`].
///
/// [rayon]: https://github.com/rayon-rs/rayon
pub fn thread_pool() -> &'static ThreadPool {
    static THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();
    THREAD_POOL.get_or_init(|| {
        let physical_cpus = optimal_core_count();

        let num_threads = if let Some(threads_var) = env::var_os("RTEN_NUM_THREADS") {
            let requested_threads: Result<u32, _> = threads_var.to_string_lossy().parse();
            match requested_threads {
                Ok(n_threads) => n_threads.clamp(1, num_cpus::get() as u32),
                Err(_) => physical_cpus,
            }
        } else {
            physical_cpus
        };

        ThreadPool::with_num_threads(num_threads.as_usize()).unwrap_or_else(|error| {
            if error
                .source()
                .and_then(|source| source.downcast_ref::<io::Error>())
                .is_some_and(|source| source.kind() == io::ErrorKind::Unsupported)
            {
                ThreadPool {
                    pool: None,
                    workers: WorkerThreads::default(),
                }
            } else {
                panic!("RTen global thread pool creation failed: {error}");
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::optimal_core_count;

    #[test]
    fn test_optimal_core_count() {
        let max_cores = num_cpus::get_physical() as u32;
        let opt_cores = optimal_core_count();
        assert!(opt_cores >= 1 && opt_cores <= max_cores);
    }
}
