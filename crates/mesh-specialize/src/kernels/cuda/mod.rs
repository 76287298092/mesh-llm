//! Linux CUDA Driver API instruction qualification.

pub(super) mod attention;
mod attention_core;
mod attention_finish;
mod attention_gate;
mod attention_prepare;

mod bf16_trial;
mod causal_conv4;
mod driver;
pub(super) mod embedding_norm;
mod fp8_exact_trial;
pub(super) mod fp8_mlp_trial;
mod gdn_output;
mod gdn_prepare;
mod gdn_recurrent;
mod gemm;
pub(super) mod instructions;
mod launch_profile;
mod nvfp4_exact_trial;
pub(super) mod projections;
pub(super) mod residency;
mod residency_entry;
mod resident_activation;
mod resident_attention;
mod resident_attention_core;
mod resident_attention_gate;
mod resident_attention_prepare;
pub(super) mod resident_attention_trial;
mod resident_bf16;
mod resident_conv;
mod resident_embedding;
mod resident_fp8;
mod resident_gdn;
mod resident_gdn_core;
pub(super) mod resident_gdn_trial;
mod resident_head;
mod resident_layer_diagnostic;
mod resident_mlp;
mod resident_model;
pub(super) mod resident_model_bench;
pub(super) mod resident_model_profile;
pub(super) mod resident_model_trial;
mod resident_norm;
mod resident_nvfp4;
mod resident_projection;
mod resident_state;
mod resident_weights;
#[allow(
    dead_code,
    reason = "Experimental workspace awaiting operator integration and GPU qualification"
)]
mod resident_workspace;
mod residual_norm;
mod rms_norm;
mod silu_trial;
pub(super) mod workloads;

use super::{fixtures, nvfp4_layout};
use anyhow::{Result, anyhow, ensure};
use driver::{Buffer, Context, Event, Function, Module};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(super) fn device_probe(device: i32) -> Result<Value> {
    let context = Context::new(device)?;
    let info = context.info();
    let (free, total) = context.memory()?;
    ensure!(
        info.major > 0 && info.minor >= 0 && info.driver_version > 0,
        "CUDA returned invalid selected-device properties"
    );
    Ok(json!({
        "ordinal": info.ordinal,
        "uuid": info.uuid,
        "compute_arch": format!("sm_{}{}", info.major, info.minor),
        "driver_api_version": info.driver_version,
        "total_memory_bytes": total,
        "free_memory_bytes": free,
    }))
}

pub(super) fn run(ptx: &str, device: i32) -> Result<Value> {
    ensure!(ptx.contains(".target sm_120a"), "probe must target sm_120a");
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "probe requires SM120, got {}.{}",
        info.major,
        info.minor
    );
    context_isolation_check(&context, device)?;
    let before = context.memory()?;
    let module = Module::load(&context, ptx)?;
    let function = module.function("probe_nvfp4_mma")?;
    let resources = function.resources()?;
    let logical_cases = fixtures::fixtures().map_err(anyhow::Error::msg)?;
    let mut results = Vec::new();
    for case in &logical_cases {
        for selector_a in 0..2 {
            for selector_b in 0..4 {
                results.push(run_case(&context, &function, case, selector_a, selector_b)?);
            }
        }
    }
    context.synchronize()?;
    let after = context.memory()?;
    Ok(json!({
        "schema_version": 1,
        "kind": "single-warp-nvfp4-instruction-qualification",
        "device_ordinal": device,
        "device": info,
        "context_isolation_passed": true,
        "resources": resources,
        "jit_log": module.jit_log(),
        "memory_before_module": {"free_bytes": before.0, "total_bytes": before.1},
        "memory_after_cases": {"free_bytes": after.0, "total_bytes": after.1},
        "cases": results,
        "all_passed": results.iter().all(|case| case["passed"] == true),
        "model_prefill_tokens_per_second": null,
        "model_decode_tokens_per_second": null,
        "model_context_tokens": null,
        "timing_note": "CUDA event duration includes a single launch and host submission gaps; not a throughput benchmark"
    }))
}

fn context_isolation_check(first: &Context, device: i32) -> Result<()> {
    let first_buffer = Buffer::new(first, 4)?;
    let second = Context::new(device)?;
    let second_buffer = Buffer::new(&second, 4)?;
    first_buffer.upload(&[1, 3, 9, 3])?;
    second_buffer.upload(&[5, 0, 9, 0])?;
    let mut read = [0; 4];
    first_buffer.download(&mut read)?;
    ensure!(read == [1, 3, 9, 3], "first context buffer changed");
    second_buffer.download(&mut read)?;
    ensure!(read == [5, 0, 9, 0], "second context buffer changed");
    let first_event = Event::new(first)?;
    let second_event = Event::new(&second)?;
    ensure!(
        first_event.elapsed_since(&second_event).is_err(),
        "foreign-context events were accepted"
    );
    Ok(())
}

fn upload_words<'a>(context: &'a Context, words: &[u32]) -> Result<Buffer<'a>> {
    let bytes: Vec<_> = words.iter().flat_map(|value| value.to_ne_bytes()).collect();
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(&bytes)?;
    Ok(buffer)
}

fn run_case(
    context: &Context,
    function: &Function<'_, '_>,
    case: &fixtures::Fixture,
    mut selector_a: u16,
    mut selector_b: u16,
) -> Result<Value> {
    let packed = nvfp4_layout::pack(&case.a, &case.b, &case.sa, &case.sb, selector_a, selector_b)
        .map_err(anyhow::Error::msg)?;
    let a = upload_words(context, &packed.a)?;
    let b = upload_words(context, &packed.b)?;
    let sa = upload_words(context, &packed.scale_a)?;
    let sb = upload_words(context, &packed.scale_b)?;
    let output = upload_words(context, &[f32::NAN.to_bits(); 128])?;
    let mut pointers = [
        a.pointer(),
        b.pointer(),
        sa.pointer(),
        sb.pointer(),
        output.pointer(),
    ];
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast())
        .collect();
    args.push((&mut selector_a as *mut u16).cast());
    args.push((&mut selector_b as *mut u16).cast());
    let start = Event::new(context)?;
    let end = Event::new(context)?;
    let allocated = context.memory()?;
    start.record()?;
    // SAFETY: Exact seven-argument signature, one complete warp; allocations and
    // host argument storage live until the end event has synchronized below.
    unsafe {
        function.launch([1, 1, 1], [32, 1, 1], 0, &mut args)?;
    }
    end.record()?;
    end.synchronize()?;
    let elapsed_ms = end.elapsed_since(&start)?;
    let mut bytes = vec![0; 128 * 4];
    output.download(&mut bytes)?;
    let raw: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect();
    let actual = nvfp4_layout::unpack_output(&raw).map_err(anyhow::Error::msg)?;
    let (max_abs_error, mismatches) = compare(&actual, &case.expected)?;
    Ok(json!({
        "fixture": case.name, "selector_a": selector_a, "selector_b": selector_b,
        "elements": actual.len(), "max_abs_error": max_abs_error,
        "mismatches": mismatches, "passed": mismatches == 0,
        "event_ms": elapsed_ms,
        "allocation_payload_bytes": (128 + 64 + 32 + 32 + 128) * 4,
        "free_bytes_with_case_allocations": allocated.0,
        "expected": case.expected, "actual": actual
    }))
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<(f32, usize)> {
    ensure!(actual.len() == expected.len(), "output lengths differ");
    let mut max_error = 0.0_f32;
    let mut mismatches = 0;
    for (&a, &e) in actual.iter().zip(expected) {
        if !a.is_finite() || !e.is_finite() {
            return Err(anyhow!("nonfinite numerical output"));
        }
        let error = (a - e).abs();
        max_error = max_error.max(error);
        // Fixtures use powers-of-two scales and half-integer FP4 values, making
        // their small dot products exactly representable in f32.
        mismatches += usize::from(error != 0.0);
    }
    Ok((max_error, mismatches))
}

mod nvfp4_quantize;

mod nvfp4_linear;

mod mlp;
mod mlp_activation;
mod residual_add;

mod resident_mtp;
pub(super) mod resident_mtp_trial;
mod resident_speculation;

pub(super) mod feature_projection_trial;

pub(super) mod feature_attention_trial;

pub(super) mod feature_graph_trial;

mod fp8_projection_audit;

pub(super) mod feature_gdn_replay_trial;

pub(super) mod resident_recovery;
