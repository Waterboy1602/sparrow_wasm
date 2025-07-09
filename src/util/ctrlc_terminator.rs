use crate::util::terminator::Terminator;
use log::warn;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct CtrlCTerminator {
    pub timeout: Option<Instant>,
    pub ctrlc: Arc<AtomicBool>,
}

impl CtrlCTerminator {
    /// Sets up the handler for Ctrl-C (only call once)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Self {
        let ctrlc = Arc::new(AtomicBool::new(false));
        let c = ctrlc.clone();

        ctrlc::set_handler(move || {
            warn!(" terminating...");
            c.store(true, Ordering::SeqCst);
        })
        .expect("Error setting Ctrl-C handler");

        Self {
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
    }
}

impl Terminator for CtrlCTerminator {
    fn kill(&self) -> bool {
        self.timeout
            .map_or(false, |timeout| Instant::now() > timeout)
            || self.ctrlc.load(Ordering::SeqCst)
    }

    fn new_timeout(&mut self, timeout: Duration) {
        // Reset the Ctrl-C flag and set a new timeout
        self.ctrlc.store(false, Ordering::SeqCst);
        self.timeout = Some(Instant::now() + timeout);
    }

    fn timeout_at(&self) -> Option<Instant> {
        self.timeout
    }
}
