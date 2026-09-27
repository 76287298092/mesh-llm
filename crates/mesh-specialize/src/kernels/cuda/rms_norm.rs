//! Host-side CUDA execution and measurement for the RMSNorm kernel.

use super::driver::{self, Buffer, Context, Event, Function, Module};
use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

const ABSOLUTE_TOLERANCE: f32 = 2e-6;
const RELATIVE_TOLERANCE: f32 = 2e-6;
const WARMUP_LAUNCHES: usize = 10;
const TIMING_BATCHES: usize = 5;
const LAUNCHES_PER_BATCH: usize = 100;

pub(super) fn cases(context: &Context, module: &Module<'_>) -> Result<Vec<Value>> {
    let function = module.function("rms_norm_f32")?;
    let resources = function.resources()?;
    let fixtures = super::super::rms_norm_fixtures::fixtures().map_err(anyhow::Error::msg)?;
    fixtures
        .iter()
        .map(|fixture| run_fixture(context, &function, fixture, resources))
        .collect()
}

fn run_fixture(
    context: &Context,
    function: &Function<'_, '_>,
    fixture: &super::super::rms_norm_fixtures::Fixture,
    resources: driver::FunctionResources,
) -> Result<Value> {
    let input_words: Vec<_> = fixture.input.iter().map(|value| value.to_bits()).collect();
    let weight_words: Vec<_> = fixture.weight.iter().map(|value| value.to_bits()).collect();
    let output_words = vec![f32::NAN.to_bits(); fixture.expected.len()];
    let input = super::upload_words(context, &input_words)?;
    let weight = super::upload_words(context, &weight_words)?;
    let output = super::upload_words(context, &output_words)?;
    let (free_bytes_with_allocations, _) = context.memory()?;

    launch_once(function, &input, &weight, &output, fixture)?;
    context.synchronize()?;
    let actual = download_f32(&output, fixture.expected.len())?;
    let (max_abs_error, mismatches) = compare(&actual, &fixture.expected)?;
    let (event_ms, event_us_per_launch) =
        time_launches(context, function, &input, &weight, &output, fixture)?;
    let payload_bytes = (input_words.len() + weight_words.len() + output_words.len()) * 4;
    let samples: Vec<_> = actual
        .iter()
        .zip(&fixture.expected)
        .take(8)
        .enumerate()
        .map(|(index, (actual, expected))| {
            json!({"index": index, "actual": actual, "expected": expected})
        })
        .collect();

    Ok(json!({
        "kernel": "rms_norm_f32",
        "fixture": fixture.name,
        "rows": fixture.rows,
        "width": fixture.width,
        "epsilon": fixture.epsilon,
        "tolerance": {
            "absolute": ABSOLUTE_TOLERANCE,
            "relative": RELATIVE_TOLERANCE
        },
        "elements": actual.len(),
        "max_abs_error": max_abs_error,
        "mismatches": mismatches,
        "all_finite": true,
        "passed": mismatches == 0,
        "samples": samples,
        "resources": resources,
        "payload_bytes": payload_bytes,
        "free_bytes_with_allocations": free_bytes_with_allocations,
        "event_ms": event_ms,
        "event_us_per_launch": event_us_per_launch,
        "warmup_launches": WARMUP_LAUNCHES,
        "launches_per_batch": LAUNCHES_PER_BATCH,
        "timing_note": "Data remains resident on the device. Event batches include host submission and default-stream gaps; uploads are excluded. This is not a tokens-per-second measurement."
    }))
}

fn launch_once(
    function: &Function<'_, '_>,
    input: &Buffer<'_>,
    weight: &Buffer<'_>,
    output: &Buffer<'_>,
    fixture: &super::super::rms_norm_fixtures::Fixture,
) -> Result<()> {
    let mut input_pointer = input.pointer();
    let mut weight_pointer = weight.pointer();
    let mut output_pointer = output.pointer();
    let mut width = u32::try_from(fixture.width)?;
    let mut epsilon = fixture.epsilon;
    let mut args = [
        (&mut input_pointer as *mut u64).cast::<c_void>(),
        (&mut weight_pointer as *mut u64).cast::<c_void>(),
        (&mut output_pointer as *mut u64).cast::<c_void>(),
        (&mut width as *mut u32).cast::<c_void>(),
        (&mut epsilon as *mut f32).cast::<c_void>(),
    ];
    let rows = u32::try_from(fixture.rows)?;
    // SAFETY: The kernel has exactly these five argument types and sizes. Its grid gives each
    // row one 256-thread block; buffers and their device allocations stay live through sync.
    unsafe { function.launch([rows, 1, 1], [256, 1, 1], 0, &mut args) }
}

fn download_f32(output: &Buffer<'_>, elements: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0; elements * 4];
    output.download(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect())
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<(f32, usize)> {
    ensure!(
        actual.len() == expected.len(),
        "RMSNorm output length differs from fixture"
    );
    let mut max_abs_error = 0.0_f32;
    let mut mismatches = 0;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            bail!("RMSNorm output or reference contains a nonfinite value");
        }
        let error = (actual - expected).abs();
        let tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * expected.abs();
        max_abs_error = max_abs_error.max(error);
        mismatches += usize::from(error > tolerance);
    }
    Ok((max_abs_error, mismatches))
}

fn time_launches(
    context: &Context,
    function: &Function<'_, '_>,
    input: &Buffer<'_>,
    weight: &Buffer<'_>,
    output: &Buffer<'_>,
    fixture: &super::super::rms_norm_fixtures::Fixture,
) -> Result<(Vec<f32>, Vec<f32>)> {
    for _ in 0..WARMUP_LAUNCHES {
        launch_once(function, input, weight, output, fixture)?;
    }
    context.synchronize()?;

    let start = Event::new(context)?;
    let end = Event::new(context)?;
    let mut event_ms = Vec::with_capacity(TIMING_BATCHES);
    let mut event_us_per_launch = Vec::with_capacity(TIMING_BATCHES);
    for _ in 0..TIMING_BATCHES {
        start.record()?;
        for _ in 0..LAUNCHES_PER_BATCH {
            launch_once(function, input, weight, output, fixture)?;
        }
        end.record()?;
        end.synchronize()?;
        let elapsed_ms = end.elapsed_since(&start)?;
        event_ms.push(elapsed_ms);
        event_us_per_launch.push(elapsed_ms * 1000.0 / LAUNCHES_PER_BATCH as f32);
    }
    Ok((event_ms, event_us_per_launch))
}
