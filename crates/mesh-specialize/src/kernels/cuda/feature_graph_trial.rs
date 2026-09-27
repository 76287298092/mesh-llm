//! Fixed-address graph replay and capture guard qualification.
use super::driver::{Buffer, Context, Module, graph::Stream};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(crate) fn run(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    let module = Module::load(&ctx, ptx)?;
    let a = Buffer::new(&ctx, 514)?;
    let b = Buffer::new(&ctx, 514)?;
    let out = Buffer::new(&ctx, 514)?;
    let function = module.function("residual_add_bf16")?;
    let mut stream = Stream::new(&ctx)?;
    a.upload(
        &vec![0x3f80_u16; 257]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    b.upload(
        &vec![0x4000_u16; 257]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    let mut pointers = [a.pointer(), b.pointer(), out.pointer()];
    let mut count = 257_u32;
    let mut args = pointers
        .iter_mut()
        .map(|v| (v as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.push((&mut count as *mut u32).cast());
    let capture = stream.begin_capture()?;
    ensure!(stream.is_capturing(), "capture state not set");
    ensure!(
        Buffer::new(&ctx, 4).is_err(),
        "allocation accepted during capture"
    );
    ensure!(
        ctx.synchronize().is_err(),
        "context sync accepted during capture"
    );
    // SAFETY: Buffer extents and argument storage satisfy the residual-add ABI. This
    // call must be rejected by the capture guard before CUDA sees a default-stream launch.
    ensure!(
        unsafe { function.launch([2, 1, 1], [256, 1, 1], 0, &mut args) }.is_err(),
        "default launch accepted during capture"
    );
    capture.abort()?;
    ensure!(!stream.is_capturing(), "abort left capture active");
    drop(stream.begin_capture()?);
    ensure!(!stream.is_capturing(), "dropped capture left state active");
    stream.abort_capture()?;
    let capture = stream.begin_capture()?;
    // SAFETY: All buffers and module outlive graph/exec; arguments are valid for this
    // bounded 257-value kernel. No referenced allocation moves or is freed during replay.
    unsafe {
        function.launch_on_stream(&stream, [2, 1, 1], [256, 1, 1], 0, &mut args)?;
    }
    // SAFETY: Captured module/allocations remain live until graph/exec drops after sync.
    let graph = unsafe { capture.finish()? };
    // SAFETY: Same retained module and fixed allocations remain live for the executable.
    let exec = unsafe { graph.instantiate()? };
    for _ in 0..3 {
        // SAFETY: The captured arguments remain valid, and this is the owning stream context.
        unsafe {
            exec.launch(&stream)?;
        }
    }
    stream.synchronize()?;
    let mut output = vec![0; 514];
    out.download(&mut output)?;
    ensure!(
        output
            .as_chunks::<2>()
            .0
            .iter()
            .all(|v| u16::from_le_bytes(*v) == 0x4040),
        "graph replay differs from 1+2 oracle"
    );
    drop(exec);
    drop(graph);
    Ok(
        json!({"kind":"fixed-address-graph-check","all_passed":true,"device":ctx.info(),
        "replays":3,"elements":257,"allocation_sync_default_launch_rejected":true,
        "explicit_and_dropped_capture_abort_passed":true,
        "scope":"bounded graph ownership and exact replay only; no model or throughput qualification"}),
    )
}
