use super::super::{
    driver::{Context, Module},
    resident_native_mtp::{NativeMtpParentBinding, NativeMtpQ8Binding, ResidentNativeMtp},
};
use super::{
    SparseInput, compare_sparse,
    fixture::{Candidate, K, ROWS},
    launch::{self, BoundRequest},
    reference_diagnostics, sparse_input,
};
mod dense_fixture;
use crate::{
    native_mtp_q8_sliced_k_fc_reference,
    packages::qwen3_8_27b::{native_mtp_views::Q8MatrixView, native_source::NativeModelSource},
};
use anyhow::{Context as _, Result, ensure};
use dense_fixture::DensePattern;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;

const READBACK_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy)]
struct CaseSpec {
    candidate: Candidate,
    pattern: InputPattern,
}

#[derive(Clone, Copy)]
enum InputPattern {
    Sparse(SparseInput),
    Dense(DensePattern),
}

impl InputPattern {
    const fn name(self) -> &'static str {
        match self {
            Self::Sparse(pattern) => pattern.name(),
            Self::Dense(pattern) => pattern.name(),
        }
    }

    fn input(self, tokens: usize) -> Result<Vec<u16>> {
        match self {
            Self::Sparse(pattern) => sparse_input(pattern, tokens),
            Self::Dense(pattern) => dense_fixture::input(pattern, tokens),
        }
    }
}

struct FcExecutor<'request, 'owner, 'ctx> {
    context: &'request Context,
    module: &'request Module<'ctx>,
    binding: &'request NativeMtpQ8Binding<'owner, 'ctx>,
    object: &'request [u8],
    view: &'request Q8MatrixView,
}

pub(super) fn run(artifact: &Path, ptx: &str, device: i32) -> Result<Value> {
    let mut source = NativeModelSource::open(artifact)?;
    let source_view = source.native_mtp_views()?.fc.clone();
    let context = Context::new(device)?;
    ensure!(context.info().major >= 8, "Q8 FC requires SM80+");
    let module = Module::load(&context, ptx)?;
    let resident = ResidentNativeMtp::load(&context, &mut source)?;
    let view = &resident.views().fc;
    ensure!(source_view == *view, "FC source and resident views differ");
    let binding = resident.q8(view)?;
    ensure!(
        std::ptr::eq(binding.view(), view),
        "FC binding lost saved view"
    );
    validate_full_fc(view)?;
    let parent = binding.parent();
    let parent_extent = view
        .scale_bits
        .offset
        .checked_add(view.scale_bits.bytes)
        .context("FC parent extent overflow")?;
    ensure!(
        parent.object_id() == view.object_id,
        "FC binding has wrong parent"
    );
    ensure!(
        parent.bytes() == parent_extent,
        "FC physical parent extent mismatch"
    );
    let mut object = Vec::new();
    object
        .try_reserve_exact(usize::try_from(parent.bytes())?)
        .context("FC CPU oracle parent allocation failed")?;
    let copied = source.copy_native_mtp_parent(parent.object_id(), &mut object)?;
    ensure!(copied == parent.bytes(), "short packed FC parent copy");
    let source_sha256 = hex::encode(Sha256::digest(&object));
    ensure!(
        source_sha256 == parent.sha256(),
        "FC source parent hash mismatch"
    );
    let resident_sha256 = hash_resident_parent(parent)?;
    ensure!(
        resident_sha256 == parent.sha256(),
        "FC resident parent hash mismatch"
    );
    let executor = FcExecutor {
        context: &context,
        module: &module,
        binding: &binding,
        object: &object,
        view,
    };
    let mut cases = Vec::with_capacity(8);
    for candidate in [Candidate::C4, Candidate::C8] {
        for pattern in [
            InputPattern::Sparse(SparseInput::GroupSweep),
            InputPattern::Sparse(SparseInput::LastK),
            InputPattern::Dense(DensePattern::AlternatingUnit),
            InputPattern::Dense(DensePattern::SignedDyadic),
        ] {
            let spec = CaseSpec { candidate, pattern };
            let result = run_case(&executor, spec);
            cases.push(json!({
                "entry": candidate.entry(), "tokens": candidate.tokens(),
                "fixture": pattern.name(),
                "result": match result {
                    Ok(report) => report,
                    Err(error) => json!({"all_passed": false, "error": format!("{error:#}")}),
                },
            }));
        }
    }
    Ok(json!({
        "schema_version": 1,
        "kind": "native-mtp-q8-fc-resident-qualification-v1",
        "all_passed": cases.iter().all(|case| case["result"]["all_passed"] == true),
        "device": context.info(), "jit_log": module.jit_log(),
        "grid": [320, 1, 1], "block": [256, 1, 1], "shape": [ROWS, K],
        "physical_parent": {
            "object_id": parent.object_id(), "bytes": parent.bytes(),
            "expected_sha256": parent.sha256(), "source_copy_sha256": source_sha256,
            "resident_readback_sha256": resident_sha256,
            "source_hash_matches": true, "resident_hash_matches": true,
            "codes": {"offset": view.codes.offset, "bytes": view.codes.bytes},
            "scales": {"offset": view.scale_bits.offset, "bytes": view.scale_bits.bytes},
            "saved_view_matches": true, "identity_row_map": true,
        },
        "source_verification": source.verification_report(),
        "cases": cases,
        "native_mtp_admitted": false, "model_executable": false,
        "timing_claim": false,
    }))
}

fn validate_full_fc(view: &Q8MatrixView) -> Result<()> {
    let code_bytes = ROWS.checked_mul(K).context("FC code extent overflow")?;
    let scale_count = ROWS
        .checked_mul(K / super::FC_GROUP)
        .context("FC scale count overflow")?;
    let scale_bytes = scale_count
        .checked_mul(2)
        .context("FC scale extent overflow")?;
    let scale_offset = code_bytes
        .checked_add((256 - code_bytes % 256) % 256)
        .context("FC scale offset overflow")?;
    ensure!(view.shape == [ROWS, K], "native FC shape mismatch");
    ensure!(
        view.padded_k == K && view.group_size == super::FC_GROUP,
        "native FC layout mismatch"
    );
    ensure!(
        view.source_rows.iter().copied().eq(0..ROWS),
        "native FC row map must be identity"
    );
    ensure!(
        view.codes.offset == 0 && view.codes.bytes == u64::try_from(code_bytes)?,
        "native FC code plane mismatch"
    );
    ensure!(
        view.scale_bits.offset == u64::try_from(scale_offset)?
            && view.scale_bits.bytes == u64::try_from(scale_bytes)?
            && view.scale_count == scale_count,
        "native FC scale plane mismatch"
    );
    Ok(())
}

fn hash_resident_parent(parent: &NativeMtpParentBinding<'_, '_>) -> Result<String> {
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(READBACK_BYTES)
        .context("FC readback scratch allocation failed")?;
    scratch.resize(READBACK_BYTES, 0);
    let mut digest = Sha256::new();
    let mut offset = 0_u64;
    while offset < parent.bytes() {
        let length = usize::try_from((parent.bytes() - offset).min(u64::try_from(scratch.len())?))?;
        let chunk = &mut scratch[..length];
        parent.read_range(offset, chunk)?;
        digest.update(&*chunk);
        offset = offset
            .checked_add(u64::try_from(length)?)
            .context("FC readback position overflow")?;
    }
    Ok(hex::encode(digest.finalize()))
}

fn run_case(executor: &FcExecutor<'_, '_, '_>, spec: CaseSpec) -> Result<Value> {
    let input = spec.pattern.input(spec.candidate.tokens())?;
    let expected =
        native_mtp_q8_sliced_k_fc_reference::run(executor.object, executor.view, &input)?;
    let launched = launch::run_bound(BoundRequest {
        context: executor.context,
        module: executor.module,
        binding: executor.binding,
        candidate: spec.candidate,
        input_bf16: &input,
    })?;
    let mut report = compare_sparse(&expected, &launched.outputs, spec.candidate.tokens())?;
    if matches!(spec.pattern, InputPattern::Dense(_)) && matches!(spec.candidate, Candidate::C8) {
        let distinct = dense_fixture::columns_are_pairwise_distinct(&input, K)
            && dense_fixture::columns_are_pairwise_distinct(&expected.output_bf16, ROWS);
        report["five_columns_distinct"] = json!(distinct);
        report["all_passed"] = json!(report["all_passed"] == true && distinct);
    }
    report["kernel_resources"] = json!(launched.resources);
    report["reference"] = reference_diagnostics(&expected);
    Ok(report)
}
