use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;

use rten::ThreadPool;

thread_local! {
    static RETIREMENT: RefCell<Option<Retirement>> = const { RefCell::new(None) };
}

struct Retirement {
    entered: mpsc::SyncSender<()>,
    release: mpsc::Receiver<()>,
    finished: Arc<AtomicBool>,
    completed: mpsc::Sender<()>,
}

impl Drop for Retirement {
    fn drop(&mut self) {
        self.entered.send(()).expect("native exit observer remains");
        self.release.recv().expect("native exit release remains");
        self.finished.store(true, Ordering::Release);
        self.completed
            .send(())
            .expect("native completion owner remains");
    }
}

fn exercise_drop(unwind: bool) {
    let pool = ThreadPool::with_num_threads(1).expect("native thread pool");
    let (entered, entry) = mpsc::sync_channel(0);
    let (release, released) = mpsc::sync_channel(0);
    let finished = Arc::new(AtomicBool::new(false));
    let worker_finished = Arc::clone(&finished);
    let (completed, completion) = mpsc::channel();
    pool.run(move || {
        RETIREMENT.with(|state| {
            *state.borrow_mut() = Some(Retirement {
                entered,
                release: released,
                finished: worker_finished,
                completed,
            });
        });
    });
    let (ready, readiness) = mpsc::sync_channel(0);
    let observer = thread::spawn(move || {
        ready.send(()).expect("native observer startup remains");
        entry.recv().expect("actual native TLS destructor begins");
        release
            .send(())
            .expect("actual native TLS destructor waits");
    });
    // Windows TLS destruction can hold the loader lock required by a new thread.
    readiness.recv().expect("native observer has entered Rust");
    let result = catch_unwind(AssertUnwindSafe(move || {
        if unwind {
            let _pool = pool;
            panic!("owned caller unwind");
        }
        drop(pool);
    }));
    let complete_when_drop_returned = finished.load(Ordering::Acquire);
    observer
        .join()
        .expect("native retirement observer completed");
    completion.recv().expect("native destructor completed");
    assert_eq!(result.is_err(), unwind);
    assert!(
        complete_when_drop_returned,
        "pool drop returned before its actual native thread-local destructor completed"
    );
}

#[test]
fn dropping_private_pool_joins_native_thread_destructors() {
    exercise_drop(false);
}

#[test]
fn caller_unwind_joins_native_thread_destructors() {
    exercise_drop(true);
}

#[test]
fn a_real_job_panic_propagates_without_disabling_the_owned_pool() {
    let pool = ThreadPool::with_num_threads(1).expect("native thread pool");
    let result = catch_unwind(AssertUnwindSafe(|| {
        pool.run(|| panic!("real inference job panic"));
    }));
    assert!(result.is_err());
    assert_eq!(pool.run(|| 6 * 7), 42);
    drop(pool);
}

#[test]
fn last_owner_dropped_on_its_own_worker_retires_without_self_join() {
    let pool = Arc::new(ThreadPool::with_num_threads(1).expect("native thread pool"));
    let (entered, entry) = mpsc::sync_channel(0);
    let (release, released) = mpsc::sync_channel(0);
    let (completed, completion) = mpsc::channel();
    let finished = Arc::new(AtomicBool::new(false));
    let worker_finished = Arc::clone(&finished);
    pool.run(move || {
        RETIREMENT.with(|state| {
            *state.borrow_mut() = Some(Retirement {
                entered,
                release: released,
                finished: worker_finished,
                completed,
            });
        });
    });
    let (start, started) = mpsc::channel();
    let (returned, result) = mpsc::channel();
    let last = Arc::clone(&pool);
    pool.run(|| {
        rayon::spawn(move || {
            started.recv().expect("external owner releases first");
            drop(last);
            returned.send(()).expect("self-drop observer remains");
        });
    });
    drop(pool);
    start.send(()).expect("actual asynchronous job remains");
    result
        .recv()
        .expect("last owner drop does not deadlock its worker");
    entry
        .recv()
        .expect("self-owned native worker starts retirement");
    release.send(()).expect("native TLS destructor remains");
    completion.recv().expect("native TLS retirement completed");
    assert!(finished.load(Ordering::Acquire));
}

#[cfg(target_os = "linux")]
#[test]
fn native_spawn_failure_reaps_partial_pool_and_returns_error() -> std::io::Result<()> {
    use std::fs;
    use std::io::{self, BufRead, BufReader, Read, Write};
    use std::process::{Command, Stdio};

    const CHILD: &str = "FOGSCRIB_THREAD_SPAWN_FAILURE_CHILD";
    const STACK_BYTES: usize = 16 * 1024 * 1024;
    if std::env::var_os(CHILD).is_some() {
        let tasks = || -> io::Result<std::collections::BTreeSet<String>> {
            fs::read_dir("/proc/self/task")?
                .map(|task| task.map(|task| task.file_name().to_string_lossy().into_owned()))
                .collect()
        };
        let before = tasks()?;
        let status = fs::read_to_string("/proc/self/status")?;
        let virtual_kib: u64 = status
            .lines()
            .find_map(|line| line.strip_prefix("VmSize:"))
            .and_then(|value| value.split_whitespace().next())
            .ok_or_else(|| io::Error::other("native virtual memory size missing"))?
            .parse()
            .map_err(io::Error::other)?;
        println!(
            "\nthread-pool-resource-ready {}",
            serde_json::json!({"pid":std::process::id(),"virtual_bytes":virtual_kib * 1024})
        );
        io::stdout().flush()?;
        io::stdin().read_exact(&mut [0])?;
        let result = catch_unwind(|| ThreadPool::with_num_threads(2))
            .expect("native resource failure must not panic");
        let error = result.expect_err("the native limit must reject the full private pool");
        let source = std::error::Error::source(&error)
            .and_then(|error| error.downcast_ref::<io::Error>())
            .expect("native spawn error must retain its I/O source");
        assert!(source.raw_os_error().is_some(), "{source}");
        println!("actual-private-pool-construction-error {error}");
        assert_eq!(
            tasks()?,
            before,
            "partially spawned pool retained native threads"
        );
        return Ok(());
    }

    let mut child = Command::new(std::env::current_exe()?)
        .args([
            "--exact",
            "native_spawn_failure_reaps_partial_pool_and_returns_error",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("RUST_MIN_STACK", STACK_BYTES.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let mut output = BufReader::new(child.stdout.take().expect("requested child stdout"));
    let configured = (|| -> io::Result<()> {
        let mut ready = None;
        for line in output.by_ref().lines() {
            let line = line?;
            if let Some(value) = line.strip_prefix("thread-pool-resource-ready ") {
                ready = Some(serde_json::from_str::<serde_json::Value>(value)?);
                break;
            }
        }
        let ready = ready.ok_or_else(|| io::Error::other("native child readiness missing"))?;
        assert_eq!(ready["pid"], child.id());
        let limit = ready["virtual_bytes"]
            .as_u64()
            .ok_or_else(|| io::Error::other("native child memory size missing"))?
            + (STACK_BYTES * 3 / 2) as u64;
        let python = std::env::var_os("FOGSCRIB_TEST_PYTHON")
            .ok_or_else(|| io::Error::other("explicit native Python required"))?;
        let status = Command::new(python)
            .args([
                "-c",
                "import resource,sys; resource.prlimit(int(sys.argv[1]),resource.RLIMIT_AS,(int(sys.argv[2]),int(sys.argv[2])))",
                &child.id().to_string(),
                &limit.to_string(),
            ])
            .status()?;
        if !status.success() {
            return Err(io::Error::other(
                "setting owned child's native address-space limit failed",
            ));
        }
        child
            .stdin
            .take()
            .expect("requested child stdin")
            .write_all(b"x")
    })();
    if configured.is_err() {
        child.kill()?;
    }
    let mut remaining = String::new();
    let drained = output.read_to_string(&mut remaining);
    let status = child.wait()?;
    configured?;
    drained?;
    assert!(status.success(), "{status}: {remaining}");
    Ok(())
}
