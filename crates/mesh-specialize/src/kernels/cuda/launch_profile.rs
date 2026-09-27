//! Per-launch CUDA event attribution for bounded diagnostic captures.

use super::driver::{Context, Event};
use anyhow::{Result, anyhow, ensure};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    ptr,
};

const MAX_LAUNCHES: u64 = 8192;
const PROFILE_SCOPE: &str = "per-launch synchronized CUDA events; instrumentation changes wall time; excludes host allocation/copies";

thread_local! {
    static ACTIVE_CAPTURE: RefCell<Option<Collector>> = const { RefCell::new(None) };
    static NEXT_CAPTURE_ID: Cell<u64> = const { Cell::new(1) };
}

/// Capture CUDA event durations while operation runs on context.
pub(super) fn capture<T>(
    context: &Context,
    operation: impl FnOnce() -> Result<T>,
) -> Result<(T, Value)> {
    capture_with_key(context_key(context), operation)
}

fn capture_with_key<T>(
    context_key: usize,
    operation: impl FnOnce() -> Result<T>,
) -> Result<(T, Value)> {
    let mut guard = CaptureGuard::start(context_key)?;
    let result = operation()?;
    let collector = take_capture(context_key, guard.capture_id)?;
    guard.disarm();
    ensure!(
        collector.started_count == collector.records.len() as u64,
        "CUDA launch profile ended with unfinished launch timers"
    );
    Ok((result, aggregate_records(&collector.records)))
}

/// Begin an event pair for one launch, or return None when no capture is active.
pub(super) fn begin<'a>(
    context: &'a Context,
    name: &str,
    grid: [u32; 3],
    block: [u32; 3],
    dynamic_shared_bytes: u32,
) -> Result<Option<LaunchTimer<'a>>> {
    let Some((capture_context, capture_id)) = active_capture()? else {
        return Ok(None);
    };
    ensure!(
        capture_context == context_key(context),
        "CUDA launch profile is active for a different context"
    );
    ensure_start_capacity(capture_id)?;

    let start =
        Event::new(context).map_err(|error| anyhow!(error).context("create launch start event"))?;
    let end =
        Event::new(context).map_err(|error| anyhow!(error).context("create launch end event"))?;
    start
        .record()
        .map_err(|error| anyhow!(error).context("record launch start event"))?;
    reserve_launch(capture_context, capture_id)?;

    Ok(Some(LaunchTimer {
        context,
        capture_id,
        name: name.to_owned(),
        grid,
        block,
        dynamic_shared_bytes,
        start,
        end,
    }))
}

/// Owns the event pair for one launch until it is finished or dropped.
pub(super) struct LaunchTimer<'ctx> {
    context: &'ctx Context,
    capture_id: u64,
    name: String,
    grid: [u32; 3],
    block: [u32; 3],
    dynamic_shared_bytes: u32,
    start: Event<'ctx>,
    end: Event<'ctx>,
}

impl LaunchTimer<'_> {
    /// Complete the event interval and add it to the active capture.
    pub(super) fn finish(self) -> Result<()> {
        self.end
            .record()
            .map_err(|error| anyhow!(error).context("record launch end event"))?;
        self.end
            .synchronize()
            .map_err(|error| anyhow!(error).context("synchronize launch end event"))?;
        let elapsed_ms = self
            .end
            .elapsed_since(&self.start)
            .map_err(|error| anyhow!(error).context("read launch event duration"))?;
        ensure!(
            elapsed_ms.is_finite() && elapsed_ms >= 0.0,
            "CUDA launch event duration must be finite and nonnegative, got {elapsed_ms}"
        );
        append_record(
            self.context,
            self.capture_id,
            LaunchRecord {
                kernel: self.name,
                grid: self.grid,
                block: self.block,
                dynamic_shared_bytes: self.dynamic_shared_bytes,
                elapsed_ms: f64::from(elapsed_ms),
            },
        )
    }
}

struct CaptureGuard {
    context_key: usize,
    capture_id: u64,
    active: bool,
}

impl CaptureGuard {
    fn start(context_key: usize) -> Result<Self> {
        let capture_id = next_capture_id()?;
        ACTIVE_CAPTURE.with(|active| {
            let mut active = active
                .try_borrow_mut()
                .map_err(|_| anyhow!("CUDA launch profile state is already borrowed"))?;
            ensure!(
                active.is_none(),
                "nested CUDA launch profile capture is not supported"
            );
            *active = Some(Collector::new(context_key, capture_id));
            Ok(Self {
                context_key,
                capture_id,
                active: true,
            })
        })
    }

    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        match ACTIVE_CAPTURE.try_with(|active| {
            if let Ok(mut active) = active.try_borrow_mut()
                && active.as_ref().is_some_and(|collector| {
                    collector.context_key == self.context_key
                        && collector.capture_id == self.capture_id
                })
            {
                active.take();
            }
        }) {
            Ok(()) | Err(_) => {}
        }
    }
}

struct Collector {
    context_key: usize,
    capture_id: u64,
    started_count: u64,
    records: Vec<LaunchRecord>,
}

impl Collector {
    fn new(context_key: usize, capture_id: u64) -> Self {
        Self {
            context_key,
            capture_id,
            started_count: 0,
            records: Vec::new(),
        }
    }

    fn ensure_start_capacity(&self) -> Result<()> {
        ensure!(
            self.started_count < MAX_LAUNCHES,
            "CUDA launch profile exceeded the {MAX_LAUNCHES}-launch limit"
        );
        Ok(())
    }
}

struct LaunchRecord {
    kernel: String,
    grid: [u32; 3],
    block: [u32; 3],
    dynamic_shared_bytes: u32,
    elapsed_ms: f64,
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct GroupKey {
    kernel: String,
    grid: [u32; 3],
    block: [u32; 3],
    dynamic_shared_bytes: u32,
}

#[derive(Clone, Copy)]
struct GroupStats {
    count: u64,
    total_gpu_ms: f64,
    min_gpu_ms: f64,
    max_gpu_ms: f64,
}

fn context_key(context: &Context) -> usize {
    ptr::from_ref(context) as usize
}

fn next_capture_id() -> Result<u64> {
    NEXT_CAPTURE_ID.with(|next| {
        let current = next.get();
        let following = current
            .checked_add(1)
            .ok_or_else(|| anyhow!("CUDA launch profile capture ID space exhausted"))?;
        next.set(following);
        Ok(current)
    })
}

fn active_capture() -> Result<Option<(usize, u64)>> {
    ACTIVE_CAPTURE.with(|active| {
        let active = active
            .try_borrow()
            .map_err(|_| anyhow!("CUDA launch profile state is already borrowed"))?;
        Ok(active
            .as_ref()
            .map(|collector| (collector.context_key, collector.capture_id)))
    })
}

fn ensure_start_capacity(capture_id: u64) -> Result<()> {
    ACTIVE_CAPTURE.with(|active| {
        let active = active
            .try_borrow()
            .map_err(|_| anyhow!("CUDA launch profile state is already borrowed"))?;
        let collector = active
            .as_ref()
            .filter(|collector| collector.capture_id == capture_id)
            .ok_or_else(|| anyhow!("CUDA launch profile capture ended before launch began"))?;
        collector.ensure_start_capacity()
    })
}

fn reserve_launch(context_key: usize, capture_id: u64) -> Result<()> {
    ACTIVE_CAPTURE.with(|active| {
        let mut active = active
            .try_borrow_mut()
            .map_err(|_| anyhow!("CUDA launch profile state is already borrowed"))?;
        let collector = active
            .as_mut()
            .filter(|collector| {
                collector.context_key == context_key && collector.capture_id == capture_id
            })
            .ok_or_else(|| anyhow!("CUDA launch profile capture changed during launch"))?;
        collector.ensure_start_capacity()?;
        collector.started_count += 1;
        Ok(())
    })
}

fn append_record(context: &Context, capture_id: u64, record: LaunchRecord) -> Result<()> {
    ACTIVE_CAPTURE.with(|active| {
        let mut active = active
            .try_borrow_mut()
            .map_err(|_| anyhow!("CUDA launch profile state is already borrowed"))?;
        let collector = active
            .as_mut()
            .filter(|collector| {
                collector.context_key == context_key(context) && collector.capture_id == capture_id
            })
            .ok_or_else(|| anyhow!("CUDA launch timer finished outside its capture"))?;
        collector.records.push(record);
        Ok(())
    })
}

fn take_capture(context_key: usize, capture_id: u64) -> Result<Collector> {
    ACTIVE_CAPTURE.with(|active| {
        let mut active = active
            .try_borrow_mut()
            .map_err(|_| anyhow!("CUDA launch profile state is already borrowed"))?;
        let collector = active
            .take()
            .ok_or_else(|| anyhow!("CUDA launch profile capture was cleared unexpectedly"))?;
        ensure!(
            collector.context_key == context_key && collector.capture_id == capture_id,
            "CUDA launch profile capture identity changed during operation"
        );
        Ok(collector)
    })
}

fn aggregate_records(records: &[LaunchRecord]) -> Value {
    let mut groups = BTreeMap::<GroupKey, GroupStats>::new();
    let mut total_gpu_ms = 0.0;
    for record in records {
        total_gpu_ms += record.elapsed_ms;
        let key = GroupKey {
            kernel: record.kernel.clone(),
            grid: record.grid,
            block: record.block,
            dynamic_shared_bytes: record.dynamic_shared_bytes,
        };
        groups
            .entry(key)
            .and_modify(|stats| {
                stats.count += 1;
                stats.total_gpu_ms += record.elapsed_ms;
                stats.min_gpu_ms = stats.min_gpu_ms.min(record.elapsed_ms);
                stats.max_gpu_ms = stats.max_gpu_ms.max(record.elapsed_ms);
            })
            .or_insert(GroupStats {
                count: 1,
                total_gpu_ms: record.elapsed_ms,
                min_gpu_ms: record.elapsed_ms,
                max_gpu_ms: record.elapsed_ms,
            });
    }

    let mut groups = groups.into_iter().collect::<Vec<_>>();
    groups.sort_by(|left, right| {
        right
            .1
            .total_gpu_ms
            .total_cmp(&left.1.total_gpu_ms)
            .then_with(|| left.0.cmp(&right.0))
    });
    let group_values = groups
        .into_iter()
        .map(|(key, stats)| {
            json!({
                "kernel": key.kernel,
                "grid": key.grid,
                "block": key.block,
                "dynamic_shared_bytes": key.dynamic_shared_bytes,
                "count": stats.count,
                "total_gpu_ms": stats.total_gpu_ms,
                "min_gpu_ms": stats.min_gpu_ms,
                "max_gpu_ms": stats.max_gpu_ms,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "launch_count": records.len() as u64,
        "total_gpu_ms": total_gpu_ms,
        "groups": group_values,
        "scope": PROFILE_SCOPE,
        "throughput_claim": false,
    })
}

#[cfg(test)]
mod tests {
    use super::{Collector, LaunchRecord, MAX_LAUNCHES, aggregate_records, capture_with_key};
    use anyhow::{Result, anyhow};
    use serde_json::json;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn aggregation_groups_full_launch_shape_and_sorts_by_total_time() {
        let records = [
            record("alpha", [1, 1, 1], [32, 1, 1], 0, 1.0),
            record("alpha", [1, 1, 1], [32, 1, 1], 0, 3.0),
            record("alpha", [1, 1, 1], [32, 1, 1], 16, 5.0),
            record("beta", [2, 1, 1], [32, 1, 1], 0, 2.0),
        ];
        let report = aggregate_records(&records);
        assert_eq!(report["launch_count"], json!(4));
        assert_eq!(report["total_gpu_ms"], json!(11.0));
        assert_eq!(report["groups"][0]["kernel"], json!("alpha"));
        assert_eq!(report["groups"][0]["dynamic_shared_bytes"], json!(16));
        assert_eq!(report["groups"][0]["count"], json!(1));
        assert_eq!(report["groups"][0]["total_gpu_ms"], json!(5.0));
        assert_eq!(report["groups"][1]["count"], json!(2));
        assert_eq!(report["groups"][1]["total_gpu_ms"], json!(4.0));
        assert_eq!(report["groups"][1]["min_gpu_ms"], json!(1.0));
        assert_eq!(report["groups"][1]["max_gpu_ms"], json!(3.0));
        assert_eq!(report["groups"][2]["kernel"], json!("beta"));
        assert_eq!(report["groups"][2]["grid"], json!([2, 1, 1]));
    }

    #[test]
    fn launch_limit_rejects_an_additional_start() {
        let mut collector = Collector::new(1, 1);
        collector.started_count = MAX_LAUNCHES;
        assert!(collector.ensure_start_capacity().is_err());
    }

    #[test]
    fn nested_capture_rejection_preserves_the_outer_capture() -> Result<()> {
        let (_, report) = capture_with_key(11, || {
            assert!(capture_with_key(22, || Ok(())).is_err());
            super::ACTIVE_CAPTURE.with(|active| {
                let active = active.borrow();
                assert_eq!(active.as_ref().map(|capture| capture.context_key), Some(11));
            });
            Ok(())
        })?;
        assert_eq!(report["launch_count"], json!(0));
        Ok(())
    }

    #[test]
    fn capture_clears_state_after_operation_error() {
        let error = capture_with_key::<()>(33, || Err(anyhow!("operation failed")));
        assert!(error.is_err());
        assert!(super::active_capture().unwrap().is_none());
    }

    #[test]
    fn capture_clears_state_after_unwind() {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = capture_with_key::<()>(44, || panic!("operation panicked"));
        }));
        assert!(result.is_err());
        assert!(super::active_capture().unwrap().is_none());
    }

    fn record(
        kernel: &str,
        grid: [u32; 3],
        block: [u32; 3],
        dynamic_shared_bytes: u32,
        elapsed_ms: f64,
    ) -> LaunchRecord {
        LaunchRecord {
            kernel: kernel.to_owned(),
            grid,
            block,
            dynamic_shared_bytes,
            elapsed_ms,
        }
    }
}
