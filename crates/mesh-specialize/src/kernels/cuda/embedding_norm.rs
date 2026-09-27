//! Execute the embedding/norm kernel and check independent scalar results.
use super::driver::{Buffer, Context, Function, Module};
use crate::entry_reference::{self, EntryReference};
use crate::kernels::EmbeddingNormInput;
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::ffi::c_void;

pub(in crate::kernels) fn run(ptx: &str, device: i32, input: &EmbeddingNormInput) -> Result<Value> {
    ensure!(
        ptx.contains(".target sm_120a"),
        "entry kernel must target SM120a"
    );
    ensure!(
        (1..=16).contains(&input.batches.len()),
        "invalid entry batch count"
    );
    // Validate all host extents and token IDs before exposing pointers to the GPU.
    let references: Vec<_> = input
        .batches
        .iter()
        .map(|tokens| {
            entry_reference::embedding_norm(
                &input.table,
                tokens,
                &input.weight,
                input.width,
                input.epsilon,
            )
        })
        .collect::<Result<_>>()?;
    let context = Context::new(device)?;
    let info = context.info();
    ensure!(
        (info.major, info.minor) == (12, 0),
        "entry kernel requires SM120"
    );
    let before = context.memory()?;
    let module = Module::load(&context, ptx)?;
    let function = module.function("embedding_norm_bf16")?;
    let table = upload(&context, &input.table)?;
    let weight = upload(&context, &input.weight)?;
    let mut cases = Vec::new();
    for (tokens, reference) in input.batches.iter().zip(&references) {
        cases.push(run_case(
            &context, &function, &table, &weight, input, tokens, reference,
        )?);
    }
    let after = context.memory()?;
    Ok(
        json!({"schema_version":1,"kind":"qwen-embedding-input-norm-trial",
        "device":info,"resources":function.resources()?,"jit_log":module.jit_log(),
        "all_passed":cases.iter().all(|case|case["passed"]==true),"cases":cases,
        "weight_payload_bytes":input.table.len()+input.weight.len(),
        "memory_before_allocations":{"free_bytes":before.0,"total_bytes":before.1},
        "memory_with_weights":{"free_bytes":after.0,"total_bytes":after.1},
        "timing_collected":false,"full_model_executed":false}),
    )
}

fn upload<'a>(context: &'a Context, bytes: &[u8]) -> Result<Buffer<'a>> {
    let buffer = Buffer::new(context, bytes.len())?;
    buffer.upload(bytes)?;
    Ok(buffer)
}

fn run_case(
    context: &Context,
    function: &Function<'_, '_>,
    table: &Buffer<'_>,
    weight: &Buffer<'_>,
    input: &EmbeddingNormInput,
    tokens: &[u32],
    reference: &EntryReference,
) -> Result<Value> {
    let token_bytes: Vec<_> = tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect();
    let token_buffer = upload(context, &token_bytes)?;
    let count = reference.residual.len();
    let residual = upload(context, &vec![0xa5; count * 2])?;
    let normalized = upload(context, &vec![0xa5; count * 2])?;
    let unrounded = upload(context, &vec![0xff; count * 4])?;
    let buffers = [
        table,
        &token_buffer,
        weight,
        &residual,
        &normalized,
        &unrounded,
    ];
    launch(function, &buffers, tokens.len(), input.width, input.epsilon)?;
    context.synchronize()?;
    let mut residual_bytes = vec![0; count * 2];
    let mut normalized_bytes = vec![0; count * 2];
    let mut unrounded_bytes = vec![0; count * 4];
    residual.download(&mut residual_bytes)?;
    normalized.download(&mut normalized_bytes)?;
    unrounded.download(&mut unrounded_bytes)?;
    let residual: Vec<_> = residual_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    let normalized: Vec<_> = normalized_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    let unrounded: Vec<_> = unrounded_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    compare(tokens, &residual, &normalized, &unrounded, reference)
}

pub(super) fn launch(
    function: &Function<'_, '_>,
    buffers: &[&Buffer<'_>; 6],
    rows: usize,
    width: usize,
    epsilon: f32,
) -> Result<()> {
    let mut pointers = buffers.map(|buffer| buffer.pointer());
    let mut width = u32::try_from(width)?;
    let mut epsilon = epsilon;
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|pointer| (pointer as *mut u64).cast())
        .collect();
    args.push((&mut width as *mut u32).cast());
    args.push((&mut epsilon as *mut f32).cast());
    // SAFETY: Six correctly sized device pointers, u32 width and f32 epsilon match
    // the kernel ABI. Host validation establishes extents and all token bounds.
    // All allocations outlive this launch and the caller synchronizes before use.
    unsafe { function.launch([u32::try_from(rows)?, 1, 1], [256, 1, 1], 0, &mut args) }
}

fn compare(
    tokens: &[u32],
    residual: &[u16],
    normalized: &[u16],
    actual: &[f32],
    reference: &EntryReference,
) -> Result<Value> {
    ensure!(
        actual.len() == reference.unrounded.len()
            && residual.len() == actual.len()
            && normalized.len() == actual.len(),
        "entry output extent mismatch"
    );
    let mut maximum_error = 0.0_f32;
    let mut numerical_mismatches = 0;
    let mut rounding_mismatches = 0;
    let mut bf16_differences = 0;
    let mut bf16_over_one_ulp = 0;
    for (index, (&actual, &expected)) in actual.iter().zip(&reference.unrounded).enumerate() {
        ensure!(
            actual.is_finite() && expected.is_finite(),
            "nonfinite entry result"
        );
        let error = (actual - expected).abs();
        maximum_error = maximum_error.max(error);
        numerical_mismatches += usize::from(error > 2e-6 + 2e-6 * expected.abs());
        rounding_mismatches +=
            usize::from(normalized[index] != entry_reference::round_bf16(actual));
        let difference = normalized[index].abs_diff(reference.normalized[index]);
        bf16_differences += usize::from(difference != 0);
        bf16_over_one_ulp += usize::from(difference > 1);
    }
    let residual_mismatches = residual
        .iter()
        .zip(&reference.residual)
        .filter(|(a, b)| a != b)
        .count();
    Ok(
        json!({"tokens":tokens,"elements":actual.len(),"max_abs_error":maximum_error,
        "absolute_tolerance":2e-6,"relative_tolerance":2e-6,
        "residual_mismatches":residual_mismatches,"numerical_mismatches":numerical_mismatches,
        "rounding_mismatches":rounding_mismatches,"bf16_reference_differences":bf16_differences,
        "bf16_reference_over_one_ulp":bf16_over_one_ulp,
        "passed":residual_mismatches==0 && numerical_mismatches==0 && rounding_mismatches==0 && bf16_over_one_ulp==0,
        "reference":"independent f64 sum, f32 normalization, BF16 RNE; at most one BF16 ULP at rounding boundaries"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn comparison_detects_corrupt_residual_rounding_and_numerics() {
        let reference = EntryReference {
            residual: vec![0x3f80],
            normalized: vec![0x3f80],
            unrounded: vec![1.0],
        };
        assert_eq!(
            compare(&[0], &[0x3f80], &[0x3f80], &[1.0], &reference).unwrap()["passed"],
            true
        );
        assert_eq!(
            compare(&[0], &[0x4000], &[0x3f80], &[1.0], &reference).unwrap()["passed"],
            false
        );
        assert_eq!(
            compare(&[0], &[0x3f80], &[0x3f81], &[1.0], &reference).unwrap()["passed"],
            false
        );
        assert_eq!(
            compare(&[0], &[0x3f80], &[0x4000], &[2.0], &reference).unwrap()["passed"],
            false
        );
        assert!(compare(&[0], &[0x3f80], &[0x7fc0], &[f32::NAN], &reference).is_err());
    }
}
