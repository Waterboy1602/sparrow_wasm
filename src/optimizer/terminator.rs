use log::warn;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Terminator {
    pub timeout: Option<Instant>,
    pub ctrlc: Arc<AtomicBool>,
}

impl Terminator {
    /// Creates a dummy that never terminates
    pub fn new_without_ctrlc() -> Self {
        Terminator {
            timeout: None,
            ctrlc: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Sets up the handler for Ctrl-C (only call once)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_with_ctrlc_handler() -> Self {
        let ctrlc = Arc::new(AtomicBool::new(false));
        let c = ctrlc.clone();

        ctrlc::set_handler(move || {
            warn!(" terminating...");
            c.store(true, Ordering::SeqCst);
        })
        .expect("Error setting Ctrl-C handler");

        Terminator {
            timeout: None,
            ctrlc,
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_with_ctrlc_handler() -> Self {
        warn!("Ctrl-C handler not available on wasm32 target. Using a dummy handler.");
        Terminator::new_without_ctrlc()
    }

    pub fn is_kill(&self) -> bool {
        self.timeout
            .map_or(false, |timeout| Instant::now() > timeout)
            || self.ctrlc.load(Ordering::SeqCst)
    }

    pub fn reset_ctrlc(&self) -> &Self {
        self.ctrlc.store(false, Ordering::SeqCst);
        self
    }

    /// Sets the timeout to a specific time in the future
    pub fn set_timeout_from_now(&mut self, timeout: Duration) -> &mut Self {
        self.timeout = Some(Instant::now() + timeout);
        self
    }

    pub fn clear_timeout(&mut self) -> &mut Self {
        self.timeout = None;
        self
    }
}
