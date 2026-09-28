//! Fail-closed completion for the experimental graph harness. An error from
//! BOTH completion boundaries is not evidence that CUDA has stopped using memory.
//! Exit this process without unwinding instead of freeing graph dependencies.

use crate::kernels::cuda::driver::{Context, graph::Stream};
use anyhow::Result;

#[derive(Debug, PartialEq, Eq)]
enum Decision {
    Complete,
    ReturnStreamError,
    ExitWithoutUnwind,
}

fn decision(stream_ok: bool, context_ok: bool) -> Decision {
    match (stream_ok, context_ok) {
        (true, _) => Decision::Complete,
        (false, true) => Decision::ReturnStreamError,
        (false, false) => Decision::ExitWithoutUnwind,
    }
}

pub(super) fn drain(stream: &Stream<'_>, context: &Context) -> Result<()> {
    let stream_error = match stream.synchronize() {
        Ok(()) => return Ok(()),
        Err(error) => error,
    };
    let fallback = context.synchronize();
    match decision(false, fallback.is_ok()) {
        Decision::ReturnStreamError => {
            Err(stream_error.context("graph stream drain failed; context drain completed"))
        }
        Decision::ExitWithoutUnwind => {
            tracing::error!(stream_error = %format!("{stream_error:#}"),
                context_error = %format!("{:#}", fallback.unwrap_err()),
                "graph drain failed at both boundaries; exiting prototype harness without freeing CUDA owners");
            std::process::exit(70);
        }
        Decision::Complete => unreachable!("stream failure already established"),
    }
}

/// Retain upload storage through its completion callback, even on early return
/// or unwinding. `complete` may return an error only after a safe completion
/// boundary (the production drain exits without unwinding if none succeeds).
pub(super) struct Pending<T, F: FnMut() -> Result<()>> {
    pub(super) value: T,
    complete: F,
    drained: bool,
}

impl<T, F: FnMut() -> Result<()>> Pending<T, F> {
    pub(super) fn new(value: T, complete: F) -> Self {
        Self {
            value,
            complete,
            drained: false,
        }
    }

    pub(super) fn finish(&mut self) -> Result<()> {
        let result = (self.complete)();
        self.drained = true;
        result
    }
}

impl<T, F: FnMut() -> Result<()>> Drop for Pending<T, F> {
    fn drop(&mut self) {
        if !self.drained
            && let Err(error) = self.finish()
        {
            tracing::error!(error = %format!("{error:#}"), "graph pending-storage cleanup required context drain");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Decision, Pending, decision};
    use std::{cell::RefCell, rc::Rc};

    struct Storage(Rc<RefCell<Vec<&'static str>>>);
    impl Drop for Storage {
        fn drop(&mut self) {
            self.0.borrow_mut().push("release");
        }
    }

    #[test]
    fn pending_host_storage_is_retained_until_error_or_success_drain() {
        for (explicit_finish, success) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            let events = Rc::new(RefCell::new(Vec::new()));
            let storage = Storage(Rc::clone(&events));
            let mut pending = Pending::new(storage, || {
                assert!(
                    events.borrow().is_empty(),
                    "storage freed or duplicate drain"
                );
                events.borrow_mut().push("drain");
                if success {
                    Ok(())
                } else {
                    anyhow::bail!("stream error after successful fallback")
                }
            });
            if explicit_finish {
                assert_eq!(pending.finish().is_ok(), success);
            }
            drop(pending);
            assert_eq!(*events.borrow(), ["drain", "release"]);
        }
    }

    #[test]
    fn unwinding_drains_before_upload_storage_drops() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _pending = Pending::new(Storage(Rc::clone(&events)), || {
                events.borrow_mut().push("drain");
                Ok(())
            });
            panic!("simulated replay unwind");
        }));
        assert!(result.is_err());
        assert_eq!(*events.borrow(), ["drain", "release"]);
    }

    #[test]
    fn failed_drains_never_authorize_owner_release() {
        assert_eq!(decision(true, false), Decision::Complete);
        assert_eq!(decision(true, true), Decision::Complete);
        assert_eq!(decision(false, true), Decision::ReturnStreamError);
        assert_eq!(decision(false, false), Decision::ExitWithoutUnwind);
    }
}
