//! Raw-bit old/new exponential probe. No approximate or host-generated output substitution.
use super::driver::{Buffer, Context, Function, Module};
use anyhow::Result;
use serde_json::{Value, json};
use std::ffi::c_void;

const GUARD: usize = 64;
const REPEATS: usize = 3;

fn download(buffer: &Buffer<'_>) -> Result<Vec<u8>> {
    let mut bytes = vec![0; buffer.len()];
    buffer.download(&mut bytes)?;
    Ok(bytes)
}

fn output(ctx: &Context, count: usize) -> Result<Buffer<'_>> {
    let b = Buffer::new(ctx, count * 8 + 2 * GUARD)?;
    b.upload(&vec![0xa5; b.len()])?;
    Ok(b)
}

fn snapshot(buffer: &Buffer<'_>) -> Result<(Vec<u64>, bool)> {
    let bytes = download(buffer)?;
    let end = bytes.len() - GUARD;
    let guards = bytes[..GUARD]
        .iter()
        .chain(&bytes[end..])
        .all(|&x| x == 0xa5);
    let words = bytes[GUARD..end]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect();
    Ok((words, guards))
}

fn execute(
    ctx: &Context,
    function: &Function<'_, '_>,
    input: &Buffer<'_>,
    out: &Buffer<'_>,
    count: usize,
) -> Result<()> {
    let mut count = u32::try_from(count)?;
    let mut pointers = [input.pointer(), out.pointer() + GUARD as u64];
    let mut args = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast::<c_void>())
        .collect::<Vec<_>>();
    args.push((&mut count as *mut u32).cast());
    // SAFETY: Count covers aligned input and guarded output payloads, with no aliasing.
    // All buffers stay live until the following drain, even after launch failure.
    let launched =
        unsafe { function.launch([count.div_ceil(256), 1, 1], [256, 1, 1], 0, &mut args) };
    let sync = ctx.synchronize();
    launched?;
    sync
}

fn comparison(inputs: &[u64], control: &[u64], candidate: &[u64]) -> Value {
    let mismatches = control
        .iter()
        .zip(candidate)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let row = |i: usize| {
        json!({"index":i,"input_bits":format!("{:016x}",inputs[i]),
        "control_bits":format!("{:016x}",control[i]),"candidate_bits":format!("{:016x}",candidate[i])})
    };
    json!({"all_passed":control.len()==inputs.len() && candidate.len()==inputs.len() && mismatches.is_empty(),
        "fp64_bit_differences":mismatches.len(),
        "first_differences":mismatches.iter().take(16).map(|&i| row(i)).collect::<Vec<_>>(),
        "special_inputs":(inputs.len()-19..inputs.len()).map(row).collect::<Vec<_>>()})
}

pub(crate) fn standalone(ptx: &str, device: i32) -> Result<Value> {
    let ctx = Context::new(device)?;
    let info = ctx.info();
    anyhow::ensure!(
        (info.major, info.minor) == (12, 0),
        "exponential bit trial requires SM120"
    );
    let module = Module::load(&ctx, ptx)?;
    let mut report = run(&ctx, &module)?;
    report["kind"] = json!("fp64-exponential-unrolled-bit-comparison-v1");
    report["device"] = json!(info);
    Ok(report)
}

pub(super) fn run(ctx: &Context, module: &Module<'_>) -> Result<Value> {
    let inputs = crate::exponential_unrolled_reference::input_bits();
    let bytes = inputs
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    let input = Buffer::new(ctx, bytes.len())?;
    input.upload(&bytes)?;
    let control = output(ctx, inputs.len())?;
    let candidate = output(ctx, inputs.len())?;
    let baseline = module.function("exponential_control_bits")?;
    let unrolled = module.function("exponential_unrolled_bits")?;
    execute(ctx, &baseline, &input, &control, inputs.len())?;
    execute(ctx, &unrolled, &input, &candidate, inputs.len())?;
    let (old, old_guard) = snapshot(&control)?;
    let (new, new_guard) = snapshot(&candidate)?;
    let comparison = comparison(&inputs, &old, &new);
    let mut repeat_passed = true;
    for _ in 0..REPEATS {
        execute(ctx, &baseline, &input, &control, inputs.len())?;
        execute(ctx, &unrolled, &input, &candidate, inputs.len())?;
        let (old_repeat, guard_a) = snapshot(&control)?;
        let (new_repeat, guard_b) = snapshot(&candidate)?;
        repeat_passed &= old_repeat == old && new_repeat == new && guard_a && guard_b;
    }
    let unchanged = download(&input)? == bytes;
    Ok(
        json!({"inputs":inputs.len(),"comparison":comparison,"guards_intact":old_guard && new_guard,
        "repeats":REPEATS,"repeat_bits_and_guards_passed":repeat_passed,"input_bits_unchanged":unchanged,
        "resources":{"control":baseline.resources()?,"candidate":unrolled.resources()?},
        "all_passed":comparison["all_passed"]==true && old_guard && new_guard && repeat_passed && unchanged,
        "scope":"raw FP64 device helper equivalence, not attention or model qualification"}),
    )
}
